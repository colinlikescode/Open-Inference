"""Benchmark and failure schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

BenchmarkEndpoint = Literal["chat", "completions"]
BenchmarkMode = Literal["closed", "open"]


class FailureType(StrEnum):
    OOM = "oom"
    MODEL_UNSUPPORTED = "model_unsupported"
    INVALID_ARGUMENT = "invalid_argument"
    ENGINE_CRASH = "engine_crash"
    STARTUP_TIMEOUT = "startup_timeout"
    PORT_CONFLICT = "port_conflict"
    CUDA_ERROR = "cuda_error"
    NCCL_ERROR = "nccl_error"
    MODEL_DOWNLOAD_ERROR = "model_download_error"
    AUTH_ERROR = "auth_error"
    UNKNOWN = "unknown"


class CandidateFailure(BaseModel):
    """Why a candidate could not be launched or benchmarked."""

    type: FailureType = FailureType.UNKNOWN
    message: str
    stderr_tail: str = ""
    stdout_tail: str = ""
    exit_code: int | None = None
    stage: str | None = None


class BenchmarkSpec(BaseModel):
    """Exactly how one benchmark run was executed (persisted for reproducibility)."""

    concurrency: int = Field(ge=1)
    num_requests: int = Field(ge=1)
    seed: int
    streaming: bool = True
    endpoint: BenchmarkEndpoint = "chat"
    mode: BenchmarkMode = "closed"
    request_rate: float | None = None
    input_tokens_p50: int
    input_tokens_p95: int
    output_tokens_p50: int
    output_tokens_p95: int
    warmup_requests: int = 0
    request_timeout_seconds: float = 300.0
    label: str | None = None


class RequestBenchmarkResult(BaseModel):
    """Measurements for one benchmark request."""

    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    started_at: float
    first_token_at: float | None = None
    completed_at: float | None = None
    ttft_ms: float | None = None
    e2e_latency_ms: float | None = None
    tpot_ms: float | None = None
    error: str | None = None
    status_code: int | None = None
    request_index: int | None = None
    output_text: str | None = None


class GPUBenchmarkMetrics(BaseModel):
    mean_gpu_utilization: float | None = None
    peak_gpu_utilization: float | None = None
    peak_memory_bytes: int | None = None
    mean_power_watts: float | None = None
    sample_count: int = 0


class BenchmarkResult(BaseModel):
    """Aggregate result of one benchmark run against one candidate."""

    candidate_id: str
    spec: BenchmarkSpec

    total_requests: int
    successful_requests: int
    failed_requests: int

    duration_seconds: float

    request_throughput: float

    input_tokens_per_second: float
    output_tokens_per_second: float
    total_tokens_per_second: float

    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None

    tpot_p50_ms: float | None = None
    tpot_p95_ms: float | None = None
    tpot_p99_ms: float | None = None

    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    error_rate: float

    gpu_metrics: GPUBenchmarkMetrics = Field(default_factory=GPUBenchmarkMetrics)

    score: float | None = None
    errors_sample: list[str] = Field(default_factory=list)

    @property
    def concurrency(self) -> int:
        return self.spec.concurrency

    def short(self) -> str:
        return (
            f"c={self.spec.concurrency} out={self.output_tokens_per_second:,.0f} tok/s "
            f"p95={self.latency_p95_ms:,.0f} ms err={self.error_rate:.1%}"
        )


class ParetoPoint(BaseModel):
    candidate_id: str
    concurrency: int
    output_tokens_per_second: float
    latency_p95_ms: float
    on_front: bool = True
