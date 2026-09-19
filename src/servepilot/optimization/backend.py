"""Connect trusted experiments to the existing launch, router, and benchmark pipeline."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from typing import Any

from servepilot.benchmark.evaluator import (
    CLEANUP_MEMORY_TIMEOUT_SECONDS,
    CLEANUP_MEMORY_TOLERANCE_BYTES,
    LaunchedSession,
    LaunchingEvaluator,
)
from servepilot.benchmark.runner import BenchmarkRunner
from servepilot.benchmark.workload import BenchmarkRequest
from servepilot.engines.process import Launcher
from servepilot.engines.registry import EngineRegistry
from servepilot.exceptions import ConfigurationError
from servepilot.hardware.base import HardwareProvider
from servepilot.models.tokenizer import TokenCounter
from servepilot.optimization.agent import AgentTool
from servepilot.optimization.schemas import (
    ExperimentProposal,
    ExperimentResult,
    RunDefinition,
    canonical_json,
)
from servepilot.optimization.store import ExperimentStore
from servepilot.optimization.validation import validate_proposal
from servepilot.optimization.workload import ReplayEntry, replay_requests
from servepilot.runtime.ports import PortAllocator
from servepilot.schemas.benchmark import BenchmarkResult, BenchmarkSpec, RequestBenchmarkResult


class Session:
    def __init__(self, launched: LaunchedSession, metadata: dict[str, Any]) -> None:
        self.launched = launched
        self.base_url = launched.server.base_url
        self.served_model_name = launched.served_model_name
        self.metadata = metadata
        self.profiler: Callable[[], Awaitable[dict[str, Any]]] | None = None

    async def benchmark(self, spec: BenchmarkSpec) -> BenchmarkResult:
        return await self.launched.benchmark(spec)

    async def profile(self) -> None:
        self.metadata["profile"] = (
            await self.profiler()
            if self.profiler
            else {"status": "unavailable", "reason": "no GPU profiler for the fake testing engine"}
        )

    async def close(self) -> None:
        self.metadata["logs"] = [
            {
                "replica": replica.id,
                "stdout": replica.process.stdout_tail(),
                "stderr": replica.process.stderr_tail(),
            }
            for replica in self.launched.replica_set.replicas
            if replica.process is not None
        ]
        await self.launched.close()


class LaunchBackend:
    """Shared mechanics. Production supplies an isolated container launcher; tests use fakes."""

    def __init__(
        self,
        *,
        definition: RunDefinition,
        store: ExperimentStore,
        registry: EngineRegistry,
        launcher: Launcher,
        hardware: HardwareProvider,
        tokenizer: TokenCounter,
        ports: PortAllocator,
        startup_timeout: float = 1200,
    ) -> None:
        self.definition = definition
        self.store = store
        self.registry = registry
        self.launcher = launcher
        self.hardware = hardware
        self.tokenizer = tokenizer
        self.ports = ports
        self.startup_timeout = startup_timeout
        self._sessions: list[Session] = []
        self.runtime_model = definition.model
        self._memory_baseline: dict[int, int] = {}

    def validate(self, proposal: ExperimentProposal) -> None:
        validate_proposal(proposal, self.definition, self.registry)

    async def materialize(self, proposal: ExperimentProposal) -> ExperimentProposal:
        return proposal

    async def prepare(self, proposal: ExperimentProposal, number: int) -> dict[str, Any]:
        raise NotImplementedError

    def _observations(
        self,
        spec: BenchmarkSpec,
        requests: list[BenchmarkRequest],
        results: list[RequestBenchmarkResult],
    ) -> None:
        self.store.artifact(
            f"benchmark-results/{spec.label}-requests.json",
            canonical_json(
                {
                    "spec": spec.model_dump(mode="json"),
                    "tokenizer_exact": self.tokenizer.exact,
                    "requests": [asdict(request) for request in requests],
                    "results": [result.model_dump(mode="json") for result in results],
                }
            ).encode(),
        )

    async def open(self, proposal: ExperimentProposal, number: int) -> Session:
        samples = await asyncio.to_thread(self.hardware.sample, proposal.plan.gpu_ids)
        self._memory_baseline = {
            s.index: s.memory_used_bytes for s in samples if s.memory_used_bytes is not None
        }
        if set(self._memory_baseline) != set(proposal.plan.gpu_ids):
            raise ConfigurationError("cannot establish pre-experiment GPU memory usage")
        metadata = await self.prepare(proposal, number)
        evaluator = LaunchingEvaluator(
            registry=self.registry,
            launcher=self.launcher,
            ports=self.ports,
            hardware_provider=self.hardware,
            hardware=self.definition.hardware,
            model=self.runtime_model,
            workload=self.definition.workload,
            tokenizer=self.tokenizer,
            trust_remote_code=self.definition.trust_remote_code,
            startup_timeout=self.startup_timeout,
            stagger_seconds=0,
            # This backend verifies memory after every owned container has stopped.
            verify_gpu_cleanup=False,
        )
        launched = await evaluator.open(proposal.plan)
        if not isinstance(launched, LaunchedSession):
            raise TypeError("expected a launched inference session")
        entries = [ReplayEntry.model_validate(entry) for entry in self.definition.requests]
        launched.runner = BenchmarkRunner(
            tokenizer=self.tokenizer,
            workload=self.definition.workload,
            hardware=self.hardware,
            requests=replay_requests(entries, self.tokenizer) if entries else None,
            # An editable engine cannot self-report inflated token counts to improve its score.
            trust_server_usage=False,
            on_observations=self._observations,
        )
        metadata.update(
            engine_version=launched.engine_version,
            commands=[
                replica.spec.redacted_display_command for replica in launched.replica_set.replicas
            ],
            tokenizer_exact=self.tokenizer.exact,
        )
        session = Session(launched, metadata)
        self._sessions.append(session)
        return session

    async def close(self) -> None:
        sessions, self._sessions = self._sessions, []
        try:
            for session in sessions:
                await session.close()
        finally:
            await self.launcher.shutdown_all()
        if self._memory_baseline:
            deadline = time.monotonic() + CLEANUP_MEMORY_TIMEOUT_SECONDS
            while True:
                samples = await asyncio.to_thread(self.hardware.sample, list(self._memory_baseline))
                used = {
                    sample.index: sample.memory_used_bytes
                    for sample in samples
                    if sample.memory_used_bytes is not None
                }
                if set(used) != set(self._memory_baseline):
                    raise ConfigurationError(
                        "cannot verify GPU cleanup: telemetry is missing devices"
                    )
                if all(
                    used[index] - self._memory_baseline[index] <= CLEANUP_MEMORY_TOLERANCE_BYTES
                    for index in used
                ):
                    self._memory_baseline = {}
                    break
                if time.monotonic() >= deadline:
                    raise ConfigurationError(
                        "GPU memory did not return to its pre-experiment level; optimization stopped"
                    )
                await asyncio.sleep(1)

    async def agent_tools(self, incumbent: ExperimentResult | None) -> list[AgentTool]:
        return []


class FakeLaunchBackend(LaunchBackend):
    """Only usable with the explicit testing engine. Never a production fallback."""

    def validate(self, proposal: ExperimentProposal) -> None:
        super().validate(proposal)
        if proposal.plan.engine != "fake":
            raise ConfigurationError("fake backend only accepts the explicit fake testing engine")
        if proposal.changes.files or proposal.changes.commands or proposal.changes.environment:
            raise ConfigurationError("runtime code edits need the isolated container backend")

    async def prepare(self, proposal: ExperimentProposal, number: int) -> dict[str, Any]:
        return {
            "testing_only": True,
            "runtime": "fake",
            "warning": "Simulated inference; these are not GPU performance measurements.",
        }
