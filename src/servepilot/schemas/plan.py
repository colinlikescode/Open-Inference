"""Candidate plan, planning result and selection schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from servepilot.schemas.benchmark import BenchmarkResult, CandidateFailure, ParetoPoint
from servepilot.schemas.workload import Objective


class EngineName(StrEnum):
    VLLM = "vllm"
    SGLANG = "sglang"
    # Testing-only adapter that launches ServePilot's fake OpenAI backend as a real subprocess.
    FAKE = "fake"


class CandidateViability(StrEnum):
    UNKNOWN = "unknown"
    ESTIMATED_VIABLE = "estimated_viable"
    ESTIMATED_IMPOSSIBLE = "estimated_impossible"
    LAUNCH_VALIDATED = "launch_validated"
    LAUNCH_FAILED = "launch_failed"


class KVEstimate(BaseModel):
    """KV-cache memory estimate for one model/topology."""

    bytes_per_token_total: int | None = None
    bytes_per_token_per_gpu: int | None = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    explanation: str = ""


class MemoryEstimate(BaseModel):
    """Static per-GPU memory breakdown for one candidate. All values are bytes per GPU."""

    device_total_bytes: int
    device_free_bytes: int
    memory_fraction: float
    engine_budget_bytes: int

    weights_bytes: int | None
    activations_bytes: int
    cuda_graph_bytes: int
    communication_bytes: int
    engine_overhead_bytes: int
    safety_reserve_bytes: int

    kv_cache_bytes_available: int | None
    kv_bytes_per_token_per_gpu: int | None
    kv_confidence: Literal["high", "medium", "low", "unknown"] = "unknown"

    estimated_kv_tokens: int | None = None
    estimated_max_concurrency_p50: int | None = None
    estimated_max_concurrency_p95: int | None = None

    fits: bool
    confidence: Literal["high", "medium", "low", "unknown"] = "medium"
    shortfall_bytes: int = 0
    weight_source: str = "unknown"
    notes: list[str] = Field(default_factory=list)

    @property
    def fixed_bytes(self) -> int:
        return (
            (self.weights_bytes or 0)
            + self.activations_bytes
            + self.cuda_graph_bytes
            + self.communication_bytes
            + self.engine_overhead_bytes
        )


class PrefillConfig(BaseModel):
    """SGLang prefill workers paired with the plan's decode worker group."""

    gpu_groups: list[list[int]] = Field(min_length=1)
    tensor_parallel_size: int = Field(ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    memory_fraction: float = Field(default=0.8, gt=0, lt=1)
    transfer_backend: Literal["nixl", "mooncake"] = "nixl"


class CandidatePlan(BaseModel):
    """One concrete serving topology the tuner may launch."""

    id: str
    engine: EngineName

    gpu_groups: list[list[int]]

    tensor_parallel_size: int = Field(ge=1)
    data_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    replica_count: int = Field(ge=1)

    expert_parallel_size: int | None = None
    expert_parallel_enabled: bool = False
    dp_attention_enabled: bool = False

    # "ray" when a replica spans machines and the engine's Ray executor must be used.
    distributed_backend: str | None = None
    # Node ids (one per replica) when planning against a Ray cluster; None on a single machine.
    replica_nodes: list[list[str]] | None = None

    context_length: int = Field(ge=16)

    # Workload objective the plan is tuned for; adapters map it to engine performance modes.
    # Not part of the structural identity.
    objective: Objective | None = None

    memory_fraction: float | None = None

    max_concurrency: int | None = None
    max_num_seqs: int | None = None
    max_running_requests: int | None = None

    chunked_prefill_enabled: bool | None = None
    chunked_prefill_size: int | None = None

    kv_cache_dtype: str | None = None

    engine_args: dict[str, Any] = Field(default_factory=dict)
    prefill: PrefillConfig | None = None

    estimated_memory: MemoryEstimate | None = None

    rationale: list[str] = Field(default_factory=list)

    viability: CandidateViability = CandidateViability.UNKNOWN
    heuristic_rank: int | None = None

    @property
    def gpu_ids(self) -> list[int]:
        return [
            g
            for group in [*self.gpu_groups, *(self.prefill.gpu_groups if self.prefill else [])]
            for g in group
        ]

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_ids)

    @property
    def spans_nodes(self) -> bool:
        return bool(self.replica_nodes) and any(len(set(n)) > 1 for n in self.replica_nodes or [])

    def structural_key(self) -> str:
        """Identity ignoring tunable knobs; used to de-duplicate equivalent topologies."""
        return (
            f"{self.engine}|tp={self.tensor_parallel_size}|pp={self.pipeline_parallel_size}"
            f"|dp={self.data_parallel_size}|rep={self.replica_count}"
            f"|ep={self.expert_parallel_size if self.expert_parallel_enabled else 0}"
            f"|dpa={int(self.dp_attention_enabled)}|ctx={self.context_length}"
            f"|groups={self.gpu_groups}|backend={self.distributed_backend or ''}"
            f"|prefill={self.prefill.model_dump_json() if self.prefill else ''}"
        )

    def label(self) -> str:
        parts = [f"TP{self.tensor_parallel_size}"]
        if self.pipeline_parallel_size > 1:
            parts.append(f"PP{self.pipeline_parallel_size}")
        if self.expert_parallel_enabled and self.expert_parallel_size:
            parts.append(f"EP{self.expert_parallel_size}")
        if self.data_parallel_size > 1:
            parts.append(f"DP{self.data_parallel_size}")
        base = "×".join(parts)
        rep = f"{self.replica_count} replica" + ("s" if self.replica_count != 1 else "")
        suffix = f", {self.distributed_backend}" if self.distributed_backend else ""
        if self.prefill:
            suffix += f", PD {len(self.prefill.gpu_groups)} prefill × TP{self.prefill.tensor_parallel_size}"
        return f"{base} × {rep} ({self.engine.value}{suffix})"

    def with_updates(self, **updates: Any) -> CandidatePlan:
        return self.model_copy(update=updates, deep=True)


class ExcludedCandidate(BaseModel):
    """A candidate that was pruned statically, with a human-readable reason."""

    description: str
    reason: str
    engine: EngineName | None = None
    tensor_parallel_size: int | None = None
    replica_count: int | None = None
    plan: CandidatePlan | None = None


class PlanConstraints(BaseModel):
    """User overrides that constrain (not disable) the planner search."""

    engine: EngineName | None = None
    tensor_parallel_size: int | None = Field(default=None, ge=1)
    replica_count: int | None = Field(default=None, ge=1)
    gpu_ids: list[int] | None = None
    context_length: int | None = Field(default=None, ge=16)
    max_concurrency: int | None = Field(default=None, ge=1)
    memory_headroom: float | None = Field(default=None, ge=0.0, lt=1.0)
    memory_fraction: float | None = Field(default=None, gt=0.0, lt=1.0)
    kv_cache_dtype: str | None = None
    allow_busy_gpus: bool = False
    allow_context_override: bool = False
    trust_remote_code: bool = False
    served_model_name: str | None = None
    revision: str | None = None
    extra_engine_args: dict[str, Any] = Field(default_factory=dict)


class PlanningResult(BaseModel):
    """Output of the static planner."""

    candidates: list[CandidatePlan] = Field(default_factory=list)
    excluded: list[ExcludedCandidate] = Field(default_factory=list)
    selected_gpu_ids: list[int] = Field(default_factory=list)
    estimated_minimum_tp: int | None = None
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def viable(self) -> list[CandidatePlan]:
        return [
            c
            for c in self.candidates
            if c.viability in (CandidateViability.ESTIMATED_VIABLE, CandidateViability.UNKNOWN)
        ]


class CandidateEvaluation(BaseModel):
    """Everything observed about one candidate during tuning."""

    plan: CandidatePlan
    stage: str
    status: Literal["pending", "benchmarked", "failed", "skipped"] = "pending"
    results: list[BenchmarkResult] = Field(default_factory=list)
    failure: CandidateFailure | None = None
    launch_seconds: float | None = None
    engine_version: str | None = None
    runtime_metadata: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @property
    def best_result(self) -> BenchmarkResult | None:
        scored = [r for r in self.results if r.score is not None]
        if scored:
            return max(scored, key=lambda r: r.score or 0.0)
        return self.results[-1] if self.results else None


class SelectedPlan(BaseModel):
    """The plan ServePilot will serve, together with the evidence that selected it."""

    plan: CandidatePlan
    objective: Objective
    benchmarked: bool
    slo_satisfied: bool | None = None
    final_result: BenchmarkResult | None = None
    rationale: list[str] = Field(default_factory=list)
    engine_version: str | None = None
    equivalent_commands: list[str] = Field(default_factory=list)
    pareto_front: list[ParetoPoint] = Field(default_factory=list)
    source: Literal["tuned", "cached", "heuristic"] = "tuned"

    def summary(self) -> dict[str, Any]:
        p = self.plan
        out: dict[str, Any] = {
            "engine": p.engine.value,
            "tensor_parallel_size": p.tensor_parallel_size,
            "pipeline_parallel_size": p.pipeline_parallel_size,
            "data_parallel_size": p.data_parallel_size,
            "replica_count": p.replica_count,
            "distributed_backend": p.distributed_backend,
            "gpu_groups": p.gpu_groups,
            "context_length": p.context_length,
            "memory_fraction": p.memory_fraction,
            "max_concurrency": p.max_concurrency,
            "benchmarked": self.benchmarked,
            "source": self.source,
        }
        if self.final_result is not None:
            r = self.final_result
            out["benchmark_summary"] = {
                "concurrency": r.spec.concurrency,
                "output_tokens_per_second": r.output_tokens_per_second,
                "request_throughput": r.request_throughput,
                "ttft_p95_ms": r.ttft_p95_ms,
                "tpot_p95_ms": r.tpot_p95_ms,
                "latency_p95_ms": r.latency_p95_ms,
                "error_rate": r.error_rate,
            }
        return out
