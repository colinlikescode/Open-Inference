"""Deterministic, fail-closed selection. No scores from Pi or the engine are accepted."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence

from servepilot.optimization.schemas import (
    CorrectnessResult,
    ExperimentResult,
    RunDefinition,
    VerificationDecision,
    fingerprint,
)
from servepilot.planner.scoring import slo_violations
from servepilot.schemas.benchmark import BenchmarkResult
from servepilot.schemas.workload import Objective


class DeterministicVerifier:
    def __init__(self, definition: RunDefinition) -> None:
        self.definition = definition.model_copy(deep=True)
        self._fingerprint = fingerprint(self.definition)

    def check_contract(self) -> None:
        if fingerprint(self.definition) != self._fingerprint:
            raise RuntimeError("evaluation contract was mutated during optimization")

    def _invalid(self, result: BenchmarkResult) -> list[str]:
        policy = self.definition.policy
        reasons = []
        if result.spec.seed != policy.seed:
            reasons.append("benchmark seed changed")
        if result.spec.num_requests != max(policy.requests_per_trial, result.concurrency * 2):
            reasons.append("benchmark sample count changed")
        if result.spec.streaming != self.definition.workload.streaming:
            reasons.append("benchmark streaming mode changed")
        if (
            result.total_requests != result.spec.num_requests
            or result.successful_requests != result.total_requests
        ):
            reasons.append("benchmark has missing or failed requests")
        if result.failed_requests != 0 or result.error_rate != 0:
            reasons.append("benchmark contains request failures")
        values = [result.duration_seconds, result.output_tokens_per_second, result.latency_p95_ms]
        values.extend(v for v in (result.ttft_p95_ms, result.tpot_p95_ms) if v is not None)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            reasons.append("benchmark has missing, non-finite or non-positive performance metrics")
        return reasons

    def _utility(self, result: BenchmarkResult, baseline: BenchmarkResult | None) -> float:
        objective = self.definition.workload.objective
        if objective == Objective.THROUGHPUT:
            return result.output_tokens_per_second
        latency = result.ttft_p95_ms
        if latency is None or latency <= 0:
            return 0
        if objective == Objective.LATENCY:
            return 1 / latency
        base_tps = baseline.output_tokens_per_second if baseline is not None else 1
        base_latency = baseline.ttft_p95_ms if baseline is not None else 1
        return math.sqrt(
            (result.output_tokens_per_second / max(base_tps, 1e-9))
            * ((base_latency or 1) / latency)
        )

    def decide(
        self,
        correctness: CorrectnessResult | None,
        results: Sequence[BenchmarkResult],
        *,
        incumbent: ExperimentResult | None = None,
        baseline: BenchmarkResult | None = None,
    ) -> VerificationDecision:
        self.check_contract()
        reasons = []
        suite = self.definition.correctness
        if (
            correctness is None
            or not correctness.passed
            or correctness.suite_fingerprint != fingerprint(suite)
        ):
            reasons.append("all required correctness checks must pass against the fixed suite")
        elif not all(
            any(
                o.name == case.name and o.passed and o.streaming == mode
                for o in correctness.observations
            )
            for case in suite.cases
            for mode in ([False, True] if suite.verify_streaming else [False])
        ):
            reasons.append("correctness result is missing required cases")
        policy = self.definition.policy
        groups: dict[int, list[BenchmarkResult]] = defaultdict(list)
        for result in results:
            groups[result.concurrency].append(result)
            reasons.extend(self._invalid(result))
        if set(groups) != set(policy.concurrency_levels):
            reasons.append("capacity measurements do not cover the fixed concurrency levels")
        if any(len(group) != policy.repetitions for group in groups.values()):
            reasons.append("each load level requires the configured confirmation repetitions")
        if reasons:
            return VerificationDecision(eligible=False, reasons=list(dict.fromkeys(reasons)))

        eligible: dict[int, float] = {}
        for concurrency, group in groups.items():
            violations = []
            for result in group:
                violations.extend(
                    slo_violations(result, self.definition.workload.latency_constraints)
                )
                floor = policy.minimum_output_tokens_per_second
                if floor is not None and result.output_tokens_per_second < floor:
                    violations.append("throughput is below the configured minimum")
                if self._utility(result, baseline) <= 0:
                    violations.append("objective requires unavailable latency metrics")
            if not violations:
                # Use the worst confirmation run; a single lucky measurement cannot win.
                eligible[concurrency] = min(self._utility(r, baseline) for r in group)
            else:
                reasons.extend(f"concurrency {concurrency}: {v}" for v in dict.fromkeys(violations))
        if not eligible:
            return VerificationDecision(
                eligible=False, reasons=reasons or ["no verified capacity point"]
            )
        concurrency = max(eligible, key=lambda c: (eligible[c], -c))
        score = eligible[concurrency]
        previous = incumbent.decision.score if incumbent is not None else None
        improvement = score / previous - 1 if previous is not None and previous > 0 else None
        accepted = incumbent is None or (
            improvement is not None and improvement >= policy.minimum_improvement
        )
        if not accepted:
            reasons.append(f"improvement does not reach {policy.minimum_improvement:.1%}")
        return VerificationDecision(
            eligible=True,
            accepted=accepted,
            score=score,
            improvement_fraction=improvement,
            recommended_concurrency=concurrency,
            maximum_slo_concurrency=max(eligible),
            reasons=reasons,
        )

    @staticmethod
    def representative(results: Sequence[BenchmarkResult], concurrency: int) -> BenchmarkResult:
        group = [r for r in results if r.concurrency == concurrency]
        if not group:
            raise ValueError("no measurement at selected concurrency")
        middle = statistics.median(r.output_tokens_per_second for r in group)
        return min(group, key=lambda r: abs(r.output_tokens_per_second - middle))
