"""Agent search with a monotonic deadline, durable evidence, and explicit rollback."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from servepilot.benchmark.evaluator import CandidateLaunchFailed
from servepilot.benchmark.runner import make_spec
from servepilot.exceptions import ConfigurationError, ServePilotError
from servepilot.logging import redact_secrets
from servepilot.optimization.agent import AgentTool, PiAgent
from servepilot.optimization.budget import BudgetExpired, TimeBudget
from servepilot.optimization.correctness import CorrectnessVerifier
from servepilot.optimization.schemas import (
    CorrectnessResult,
    ExperimentProposal,
    ExperimentResult,
    RunDefinition,
    VerificationDecision,
    canonical_json,
    utc_now,
)
from servepilot.optimization.store import ExperimentStore
from servepilot.optimization.verifier import DeterministicVerifier
from servepilot.schemas.benchmark import BenchmarkResult, BenchmarkSpec


class OptimizationSession(Protocol):
    base_url: str
    served_model_name: str
    metadata: dict[str, Any]

    async def benchmark(self, spec: BenchmarkSpec) -> BenchmarkResult: ...
    async def profile(self) -> None: ...
    async def close(self) -> None: ...


class ExperimentBackend(Protocol):
    async def materialize(self, proposal: ExperimentProposal) -> ExperimentProposal: ...
    def validate(self, proposal: ExperimentProposal) -> None: ...
    async def open(self, proposal: ExperimentProposal, number: int) -> OptimizationSession: ...
    async def close(self) -> None: ...
    async def agent_tools(self, incumbent: ExperimentResult | None) -> list[AgentTool]: ...


@dataclass(frozen=True)
class OptimizationOutcome:
    best: ExperimentResult | None
    baseline: ExperimentResult | None
    stopping_reason: str
    elapsed_seconds: float
    experiment_count: int


class Optimizer:
    def __init__(
        self,
        *,
        definition: RunDefinition,
        store: ExperimentStore,
        budget: TimeBudget,
        backend: ExperimentBackend,
        agent: PiAgent,
        baselines: list[ExperimentProposal],
        plateau_seconds: float | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.definition = definition.model_copy(deep=True)
        self.store = store
        self.budget = budget
        self.backend = backend
        self.agent = agent
        self.baselines = baselines
        self.plateau_seconds = plateau_seconds
        self.progress = progress or (lambda message: None)
        self.verifier = DeterministicVerifier(definition)
        self.correctness = CorrectnessVerifier(definition.correctness)
        self.stop_requested = asyncio.Event()
        self._active: asyncio.Task[Any] | None = None
        self._reference: dict[str, str] | None = None
        self._baseline_metric: BenchmarkResult | None = None
        self._best: ExperimentResult | None = None
        self._baseline: ExperimentResult | None = None
        self._last_improvement = time.monotonic()

    def request_stop(self) -> None:
        self.stop_requested.set()
        if self._active is not None:
            self._active.cancel()

    def _recover(self) -> None:
        for event in self.store.pending():
            payload = event["payload"]
            # A result may have been fsynced just before the journal update was interrupted.
            path = self.store.experiment_dir(payload["id"]) / "result.json"
            if path.exists():
                result = ExperimentResult.model_validate_json(path.read_bytes())
                self.store.finish(result)
                continue
            self.store.finish(
                ExperimentResult(
                    id=payload["id"],
                    proposal=ExperimentProposal.model_validate(payload["proposal"]),
                    status="interrupted",
                    started_at=datetime.fromisoformat(event["at"]),
                    elapsed_seconds=0,
                    decision=VerificationDecision(
                        eligible=False,
                        reasons=["controller interrupted before verification completed"],
                    ),
                    error="interrupted previous run; runtime cleanup is required before resuming",
                )
            )
        results = self.store.results()
        for result in results:
            self._remember_reference(result)
            if result.decision.accepted:
                self._best = result

    def _remember_reference(self, result: ExperimentResult) -> None:
        if self._reference is None and result.correctness is not None and result.correctness.passed:
            self._reference = self.correctness.references(result.correctness)
        expected = (
            len(self.definition.policy.concurrency_levels) * self.definition.policy.repetitions
        )
        if (
            self._baseline is None
            and result.status in ("verified", "rejected")
            and len(result.benchmarks) == expected
            and all(not self.verifier._invalid(measurement) for measurement in result.benchmarks)
        ):
            self._baseline = result
            self._baseline_metric = result.benchmarks[0]

    async def _evaluate(self, proposal: ExperimentProposal) -> ExperimentResult:
        self.backend.validate(proposal)
        number = self.store.begin(proposal)
        started = utc_now()
        clock_start = time.monotonic()
        self.progress(f"Experiment {number}: {proposal.hypothesis}")
        session: OptimizationSession | None = None
        correctness: CorrectnessResult | None = None
        benchmarks: list[BenchmarkResult] = []
        runtime: dict[str, Any] = {}
        error: str | None = None
        status: str = "failed"
        decision = VerificationDecision(eligible=False, reasons=["verification did not complete"])
        propagate_cancel = False

        async def measure() -> None:
            nonlocal session, correctness, runtime, status, decision
            session = await self.backend.open(proposal, number)
            runtime = session.metadata
            correctness = await self.correctness.run(
                session.base_url, session.served_model_name, reference=self._reference
            )
            if not correctness.passed:
                status = "rejected"
                decision = VerificationDecision(
                    eligible=False, reasons=["correctness verification failed"]
                )
                return
            policy = self.definition.policy
            for concurrency in policy.concurrency_levels:
                for repetition in range(policy.repetitions):
                    self.progress(
                        f"Experiment {number}: concurrency {concurrency}, trial {repetition + 1}/{policy.repetitions}"
                    )
                    spec = make_spec(
                        self.definition.workload,
                        concurrency=concurrency,
                        num_requests=max(policy.requests_per_trial, concurrency * 2),
                        seed=policy.seed,
                        request_rate=self.definition.workload.target_request_rate,
                        timeout_seconds=min(
                            policy.request_timeout_seconds, max(0.001, self.budget.remaining)
                        ),
                        label=f"experiment-{number}-c{concurrency}-trial{repetition + 1}",
                    )
                    result = await session.benchmark(spec)
                    benchmarks.append(result)
                    self.store.artifact(
                        f"benchmark-results/{spec.label}.json", canonical_json(result).encode()
                    )
            if self._baseline_metric is None and all(
                not self.verifier._invalid(result) for result in benchmarks
            ):
                self._baseline_metric = benchmarks[0]
            decision = self.verifier.decide(
                correctness, benchmarks, incumbent=self._best, baseline=self._baseline_metric
            )
            status = "verified" if decision.eligible else "rejected"
            if proposal.profile:
                try:
                    self.progress(f"Experiment {number}: capturing a separate diagnostic profile")
                    await session.profile()
                except Exception as exc:
                    runtime["profile"] = {"status": "failed", "error": redact_secrets(str(exc))}

        try:
            self._active = asyncio.create_task(self.budget.run(measure))
            await self._active
        except BudgetExpired as exc:
            if status in ("verified", "rejected"):
                runtime["profile"] = {
                    "status": "interrupted",
                    "reason": "search budget expired after performance verification",
                }
            else:
                status, error = "interrupted", str(exc)
        except asyncio.CancelledError:
            if status in ("verified", "rejected"):
                runtime["profile"] = {
                    "status": "interrupted",
                    "reason": "optimization stopped after performance verification",
                }
            else:
                status, error = "interrupted", "optimization stopped before verification completed"
            propagate_cancel = not self.stop_requested.is_set()
        except Exception as exc:
            detail = exc.render() if isinstance(exc, ServePilotError) else str(exc)
            error = redact_secrets(f"{type(exc).__name__}: {detail}")
            if isinstance(exc, CandidateLaunchFailed):
                runtime["launch_failure"] = exc.failure.model_dump(mode="json")
        finally:
            self._active = None
            # Cleanup is outside the search budget. Never continue on contaminated GPUs.
            cleanup_error = None
            try:
                if session is not None:
                    await session.close()
                await self.backend.close()
            except Exception as exc:
                cleanup_error = exc
                status = "failed"
                error = f"runtime cleanup failed: {redact_secrets(str(exc))}"
                decision = VerificationDecision(eligible=False, reasons=[error])
            if status in ("failed", "interrupted"):
                decision = VerificationDecision(
                    eligible=False, reasons=[error or "experiment failed"]
                )
            result = ExperimentResult(
                id=number,
                proposal=proposal,
                status=status,  # type: ignore[arg-type]
                started_at=started,
                elapsed_seconds=time.monotonic() - clock_start,
                correctness=correctness,
                benchmarks=benchmarks,
                decision=decision,
                error=error,
                runtime=runtime,
            )
            self.store.finish(result)
            self._remember_reference(result)
            if cleanup_error is not None:
                raise ConfigurationError(
                    "optimization stopped because runtime cleanup could not be verified"
                ) from cleanup_error
        if propagate_cancel:
            raise asyncio.CancelledError
        if result.decision.accepted:
            self._best = result
            self._last_improvement = time.monotonic()
            self.progress(
                f"Experiment {number} accepted at concurrency {decision.recommended_concurrency}"
            )
        else:
            self.progress(
                f"Experiment {number} {result.status}: {error or '; '.join(decision.reasons)}"
            )
        return result

    async def _propose(self) -> ExperimentProposal | None:
        proposed: list[ExperimentProposal] = []

        async def submit(arguments: dict[str, Any]) -> dict[str, Any]:
            if proposed:
                raise ConfigurationError("submit only one experiment per planning turn")
            proposal = await self.backend.materialize(ExperimentProposal.model_validate(arguments))
            self.backend.validate(proposal)
            proposed.append(proposal)
            return {
                "submitted": True,
                "note": "The controller will independently evaluate this proposal. End your turn now.",
            }

        async def history(arguments: dict[str, Any]) -> dict[str, Any]:
            offset = int(arguments.get("offset", 0))
            if offset < 0:
                raise ValueError("offset must be non-negative")
            results = self.store.results()
            return {
                "total": len(results),
                "experiments": [r.model_dump(mode="json") for r in results[offset : offset + 10]],
            }

        async def inspect(arguments: dict[str, Any]) -> dict[str, Any]:
            return self.definition.hardware.model_dump(mode="json")

        tools = [
            AgentTool(
                "submit_experiment",
                "Submit one complete experiment proposal for independent verification.",
                ExperimentProposal.model_json_schema(),
                submit,
            ),
            AgentTool(
                "experiment_history",
                "Read immutable experiment results, 10 at a time.",
                {
                    "type": "object",
                    "properties": {"offset": {"type": "integer", "minimum": 0}},
                    "additionalProperties": False,
                },
                history,
            ),
            AgentTool(
                "inspect_cluster",
                "Read the controller's GPU and topology inventory.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                inspect,
            ),
            *await self.backend.agent_tools(self._best),
        ]
        results = self.store.results()
        context = {
            "task": "Use evidence and your tools to prepare and submit one new inference experiment. Then end your turn.",
            "model": self.definition.model.model_dump(mode="json"),
            "hardware": self.definition.hardware.model_dump(mode="json"),
            "workload": self.definition.workload.model_dump(mode="json"),
            "evaluation_policy": self.definition.policy.model_dump(mode="json"),
            "remaining_seconds": self.budget.remaining,
            "runtime_inheritance": "Set parent_experiment to an accepted experiment id to retain its code changes. Omit it to start from the prepared base engine.",
            "current_best": self._best.model_dump(mode="json") if self._best else None,
            "baseline": self._baseline.model_dump(mode="json") if self._baseline else None,
            "available_baseline_plans": [p.plan.model_dump(mode="json") for p in self.baselines],
            "recent_experiments": [
                {
                    "id": r.id,
                    "hypothesis": r.proposal.hypothesis,
                    "status": r.status,
                    "decision": r.decision.model_dump(mode="json"),
                    "error": r.error,
                }
                for r in results[-30:]
            ],
            "history_count": len(results),
        }
        turn = await self.agent.turn(context, tools, self.budget)
        sequence = (
            len([event for event in self.store.events() if event["kind"] == "agent_turn"]) + 1
        )
        relative = f"experiments/agent-turn-{sequence:04d}.json"
        digest = self.store.artifact(
            relative, canonical_json({"events": turn.events, "text": turn.text}).encode()
        )
        self.store.append(
            "agent_turn", {"artifact": relative, "sha256": digest, "submitted": bool(proposed)}
        )
        return proposed[0] if proposed else None

    def _has_baseline(self) -> bool:
        return self._baseline is not None

    async def run(self) -> OptimizationOutcome:
        self._recover()
        self.store.append(
            "search_started",
            {"budget_seconds": self.budget.seconds, "remaining_seconds": self.budget.remaining},
        )
        reason = "budget_expired"
        try:
            if not self._has_baseline():
                for baseline in self.baselines:
                    if self.budget.expired or self.stop_requested.is_set():
                        break
                    await self._evaluate(baseline)
                    if self._has_baseline():
                        break
            empty_turns = 0
            while not self.budget.expired and not self.stop_requested.is_set():
                if (
                    self.plateau_seconds is not None
                    and time.monotonic() - self._last_improvement >= self.plateau_seconds
                ):
                    reason = "plateau"
                    break
                self.progress(
                    f"Pi planning the next experiment; {self.budget.remaining:.0f}s remaining"
                )
                self._active = asyncio.create_task(self._propose())
                try:
                    proposal = await self._active
                except BudgetExpired:
                    break
                except asyncio.CancelledError:
                    if not self.stop_requested.is_set():
                        raise
                    break
                finally:
                    self._active = None
                if proposal is None:
                    empty_turns += 1
                    if empty_turns >= 3:
                        reason = "agent_did_not_propose"
                        break
                    continue
                empty_turns = 0
                await self._evaluate(proposal)
            if self.stop_requested.is_set():
                reason = "manual_stop"
        except Exception as exc:
            reason = "controller_error"
            self.store.append(
                "controller_error", {"error": redact_secrets(f"{type(exc).__name__}: {exc}")}
            )
            raise
        finally:
            await self.backend.close()
            self.store.append(
                "search_stopped",
                {
                    "reason": reason,
                    "elapsed_seconds": self.budget.elapsed,
                    "best_experiment": self._best.id if self._best else None,
                },
            )
        return OptimizationOutcome(
            best=self._best,
            baseline=self._baseline,
            stopping_reason=reason,
            elapsed_seconds=self.budget.elapsed,
            experiment_count=len(self.store.results()),
        )
