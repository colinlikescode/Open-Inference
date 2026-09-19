"""Persisted contracts. Agent proposals never contain scores or acceptance decisions."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from servepilot.schemas.benchmark import BenchmarkResult
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan
from servepilot.schemas.workload import WorkloadProfile


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AgentConfig(Contract):
    """Secrets are resolved from the environment, never stored in run artifacts."""

    executable: str = "pi"
    base_url: str = "http://127.0.0.1:4000/v1"
    model: str
    api_key_env: str = "LITELLM_API_KEY"
    context_window: int = Field(default=128000, ge=4096)
    max_tokens: int = Field(default=16384, ge=1024)
    turn_timeout_seconds: float = Field(default=300, gt=0)

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("LiteLLM base_url must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("put credentials in api_key_env, not in the LiteLLM URL")
        return value.rstrip("/")

    @field_validator("api_key_env")
    @classmethod
    def validate_env(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("api_key_env must name an environment variable")
        return value


class RuntimeFile(Contract):
    path: str
    content: str

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("runtime file path must be relative and cannot traverse directories")
        if any(part.startswith(".") for part in path.parts) or str(path) == ".":
            raise ValueError("hidden paths are not runtime artifacts")
        return str(path)


class RuntimeChanges(Contract):
    """Replayed inside an isolated engine image, never on the controller host."""

    files: list[RuntimeFile] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)
    artifacts: list[RuntimeFile] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("environment")
    @classmethod
    def engine_environment(cls, values: dict[str, str]) -> dict[str, str]:
        for key, value in values.items():
            if not re.fullmatch(
                r"(?:NCCL|GLOO|UCX|VLLM|SGLANG|TORCH|PYTORCH|TRITON)_[A-Z0-9_]+", key
            ) or any(word in key for word in ("TOKEN", "SECRET", "API_KEY", "PROFILER")):
                raise ValueError(
                    f"runtime environment variable {key!r} is not an allowed engine tuning setting"
                )
            if "\x00" in value or "\n" in value:
                raise ValueError("runtime environment values must be single-line strings")
        return values

    @model_validator(mode="after")
    def unique_paths(self) -> RuntimeChanges:
        if len({f.path for f in self.files}) != len(self.files):
            raise ValueError("runtime file paths must be unique")
        return self


class ExperimentProposal(Contract):
    hypothesis: str = Field(min_length=1, max_length=12000)
    plan: CandidatePlan
    changes: RuntimeChanges = Field(default_factory=RuntimeChanges)
    parent_experiment: int | None = Field(default=None, ge=1)
    profile: bool = Field(
        default=False,
        description="Capture a separate GPU profiler trace after the scored benchmark trials. Traces are diagnostic and never used as scored measurements.",
    )


class CorrectnessCase(Contract):
    name: str = Field(min_length=1)
    request: dict[str, Any]
    endpoint: Literal["chat", "completions"] = "chat"
    expected: str | None = None
    comparison: Literal["exact", "numeric", "similarity"] = "exact"
    absolute_tolerance: float = Field(default=0, ge=0)
    relative_tolerance: float = Field(default=0, ge=0)
    minimum_similarity: float = Field(default=1, gt=0, le=1)

    @model_validator(mode="after")
    def validate_request(self) -> CorrectnessCase:
        field = "messages" if self.endpoint == "chat" else "prompt"
        if not self.request.get(field):
            raise ValueError(f"correctness case requires {field}")
        forbidden = {"model", "stream", "stream_options", "n"} & self.request.keys()
        if forbidden:
            raise ValueError(f"verifier owns request fields: {sorted(forbidden)}")
        if self.request.get("temperature", 0) != 0:
            raise ValueError("correctness cases must use temperature 0")
        if self.request.get("max_tokens", 64) < 1:
            raise ValueError("max_tokens must be positive")
        return self


class CorrectnessSuite(Contract):
    cases: list[CorrectnessCase] = Field(min_length=1)
    verify_streaming: bool = True
    timeout_seconds: float = Field(default=120, gt=0)

    @model_validator(mode="after")
    def names_unique(self) -> CorrectnessSuite:
        if len({c.name for c in self.cases}) != len(self.cases):
            raise ValueError("correctness case names must be unique")
        return self


class CorrectnessObservation(Contract):
    name: str
    streaming: bool
    passed: bool
    expected: str | None
    actual: str | None = None
    reason: str | None = None


class CorrectnessResult(Contract):
    suite_fingerprint: str
    observations: list[CorrectnessObservation]
    baseline_reference: bool = False

    @property
    def passed(self) -> bool:
        return bool(self.observations) and all(o.passed for o in self.observations)


class EvaluationPolicy(Contract):
    seed: int = 1234
    requests_per_trial: int = Field(default=64, ge=2)
    repetitions: int = Field(default=2, ge=2)
    concurrency_levels: list[int] = Field(default_factory=lambda: [8, 16, 32, 64])
    minimum_output_tokens_per_second: float | None = Field(default=None, gt=0)
    minimum_improvement: float = Field(default=0.02, ge=0, lt=1)
    request_timeout_seconds: float = Field(default=300, gt=0)

    @field_validator("concurrency_levels")
    @classmethod
    def valid_levels(cls, value: list[int]) -> list[int]:
        if not value or any(c < 1 or c > 65536 for c in value):
            raise ValueError("concurrency_levels must contain positive bounded values")
        return sorted(set(value))


class RunDefinition(Contract):
    schema_version: Literal[1] = 1
    model: ModelProfile
    hardware: HardwareSnapshot
    workload: WorkloadProfile
    policy: EvaluationPolicy = Field(default_factory=EvaluationPolicy)
    correctness: CorrectnessSuite
    agent: AgentConfig
    engine_images: dict[str, str] = Field(default_factory=dict)
    engine_versions: dict[str, str] = Field(default_factory=dict)
    runtime_config: dict[str, Any] = Field(default_factory=dict)
    nodes: dict[str, Any] = Field(default_factory=dict)
    requests: list[dict[str, Any]] = Field(default_factory=list)
    trust_remote_code: bool = False
    created_at: datetime = Field(default_factory=utc_now)


class VerificationDecision(Contract):
    eligible: bool
    accepted: bool = False
    reasons: list[str] = Field(default_factory=list)
    score: float | None = None
    improvement_fraction: float | None = None
    recommended_concurrency: int | None = None
    maximum_slo_concurrency: int | None = None


class ExperimentResult(Contract):
    id: int = Field(ge=1)
    proposal: ExperimentProposal
    status: Literal["verified", "rejected", "failed", "interrupted"]
    started_at: datetime
    finished_at: datetime = Field(default_factory=utc_now)
    elapsed_seconds: float = Field(ge=0)
    correctness: CorrectnessResult | None = None
    benchmarks: list[BenchmarkResult] = Field(default_factory=list)
    decision: VerificationDecision
    error: str | None = None
    runtime: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)


class DeploymentRecipe(Contract):
    schema_version: Literal[1] = 1
    run_fingerprint: str
    experiment: int = Field(ge=1)
    definition: RunDefinition
    result: ExperimentResult
    files: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def verified(self) -> DeploymentRecipe:
        if self.run_fingerprint != fingerprint(self.definition):
            raise ValueError("recipe run fingerprint does not match its definition")
        if self.result.id != self.experiment or not self.result.decision.accepted:
            raise ValueError("recipe must identify an accepted experiment")
        if self.result.status != "verified" or not self.result.decision.eligible:
            raise ValueError("recipe cannot deploy an unverified or infeasible experiment")
        if self.result.correctness is None or not self.result.correctness.passed:
            raise ValueError("recipe is missing passed correctness evidence")
        return self
