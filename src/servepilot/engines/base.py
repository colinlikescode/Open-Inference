"""Common inference-engine interface.

An adapter translates a :class:`CandidatePlan` into a concrete process launch, knows how to tell
when its server is ready, and classifies failures from logs. All engine-specific CLI knowledge
lives in adapters so the planner and tuner stay engine agnostic.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, Field

from servepilot.constants import DEFAULT_READINESS_POLL_INTERVAL_SECONDS
from servepilot.engines.failures import classify_failure
from servepilot.logging import redact_secrets
from servepilot.schemas.benchmark import CandidateFailure, FailureType
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, EngineName

if TYPE_CHECKING:
    from servepilot.engines.process import ProcessHandle

# Environment variables that must never appear in displayed commands/logs.
SENSITIVE_ENV_KEYS = {
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "OPENAI_API_KEY",
    "SERVEPILOT_API_KEY",
    "VLLM_API_KEY",
}
# Variables forwarded from ServePilot's environment to engine processes.
FORWARDED_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HF_ENDPOINT",
    "TRANSFORMERS_CACHE",
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "VIRTUAL_ENV",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "GLOO_SOCKET_IFNAME",
    "RAY_ADDRESS",
    "PYTHONUNBUFFERED",
)


class SupportResult(BaseModel):
    supported: bool
    confidence: Literal["high", "medium", "low"] = "medium"
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LaunchSpec(BaseModel):
    """Normalised description of one engine server process."""

    executable: str
    args: list[str]
    env: dict[str, str]
    cwd: str | None = None
    host: str
    port: int
    gpu_ids: list[int]
    redacted_display_command: str
    replica_id: str = "replica-0"
    node_id: str | None = None
    node_ip: str | None = None
    readiness_path: str = "/v1/models"
    health_path: str = "/health"

    @property
    def base_url(self) -> str:
        host = self.node_ip or self.host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    @property
    def command(self) -> list[str]:
        return [self.executable, *self.args]


def engine_environment(python: str, gpu_ids: list[int] | None) -> dict[str, str]:
    """Environment for an engine process: forwarded variables plus the engine's own ``bin`` on PATH.

    Engines frequently live in a dedicated virtual environment; their console scripts (``ninja``
    for JIT kernels, ``ray``, ...) must be discoverable even though ServePilot runs elsewhere.
    """
    env = {k: v for k, v in os.environ.items() if k in FORWARDED_ENV_KEYS}
    # Do not resolve symlinks: a venv's python points at the base interpreter, and we want
    # the venv's own bin directory (where ninja, ray and friends are installed).
    bin_dir = str(Path(python).absolute().parent)
    path = env.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        env["PATH"] = f"{bin_dir}{os.pathsep}{path}" if path else bin_dir
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_ids)
    return env


def redacted_command(command: list[str], env: dict[str, str]) -> str:
    """Shell-quoted command line with sensitive environment values removed."""
    parts = []
    for key in sorted(env):
        if key in SENSITIVE_ENV_KEYS:
            parts.append(f"{key}=***")
        elif key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER") or key.startswith(
            ("VLLM_", "SGLANG_")
        ):
            parts.append(f"{key}={shlex.quote(env[key])}")
    parts.extend(shlex.quote(c) for c in command)
    return redact_secrets(" ".join(parts))


@dataclass
class ReadinessResult:
    ready: bool
    elapsed_seconds: float
    failure: CandidateFailure | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceEngine(ABC):
    """Base class for engine adapters."""

    engine_name: EngineName

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def version(self) -> str | None: ...

    @abstractmethod
    def supports(self, model: ModelProfile, plan: CandidatePlan) -> SupportResult: ...

    @abstractmethod
    def build_launch_spec(
        self,
        model: ModelProfile,
        plan: CandidatePlan,
        *,
        replica_index: int,
        host: str,
        port: int,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        node: tuple[str, str] | None = None,
        local_gpu_ids: list[int] | None = None,
    ) -> LaunchSpec: ...

    # -------------------------------------------------------------- capabilities
    def supports_expert_parallel(self, model: ModelProfile) -> bool:
        return False

    def supports_dp_attention(self, model: ModelProfile) -> bool:
        return False

    def supports_pipeline_parallel(self) -> bool:
        return False

    def supports_ray_backend(self) -> bool:
        return False

    def supports_native_backend(self) -> bool:
        return False

    def default_max_num_seqs(self) -> int:
        return 256

    def unavailable_hints(self) -> list[str]:
        return []

    # -------------------------------------------------------------- readiness
    async def wait_until_ready(
        self,
        spec: LaunchSpec,
        process: ProcessHandle | None,
        timeout_seconds: float,
        poll_interval: float = DEFAULT_READINESS_POLL_INTERVAL_SECONDS,
    ) -> ReadinessResult:
        """Poll the engine's HTTP endpoints until they answer, or the process dies/times out."""
        start = time.monotonic()
        base = spec.base_url
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if process is not None and not process.is_running():
                    code = process.returncode
                    failure = self.classify_failure(
                        code, process.stdout_tail(), process.stderr_tail()
                    )
                    failure.stage = "startup"
                    return ReadinessResult(False, time.monotonic() - start, failure)
                try:
                    resp = await client.get(base + spec.readiness_path)
                    if 200 <= resp.status_code < 300:
                        metadata: dict[str, Any] = {}
                        try:
                            metadata = self.extract_runtime_metadata(resp.json())
                        except ValueError:
                            metadata = {}
                        return ReadinessResult(True, time.monotonic() - start, metadata=metadata)
                except (httpx.HTTPError, OSError):
                    pass
                if time.monotonic() - start > timeout_seconds:
                    tail_out = process.stdout_tail() if process is not None else ""
                    tail_err = process.stderr_tail() if process is not None else ""
                    return ReadinessResult(
                        False,
                        time.monotonic() - start,
                        CandidateFailure(
                            type=FailureType.STARTUP_TIMEOUT,
                            message=f"{self.name()} did not become ready within {timeout_seconds:.0f}s",
                            stdout_tail=tail_out,
                            stderr_tail=tail_err,
                            exit_code=None,
                            stage="startup",
                        ),
                    )
                await asyncio.sleep(poll_interval)

    def classify_failure(
        self, exit_code: int | None, stdout_tail: str, stderr_tail: str
    ) -> CandidateFailure:
        return classify_failure(exit_code, stdout_tail, stderr_tail, engine=self.name())

    def extract_runtime_metadata(self, models_payload: Any) -> dict[str, Any]:
        """Extract useful facts from the ``/v1/models`` payload (max model length etc.)."""
        out: dict[str, Any] = {}
        if isinstance(models_payload, dict):
            data = models_payload.get("data")
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict):
                    if "id" in first:
                        out["served_model_id"] = first["id"]
                    if "max_model_len" in first:
                        out["max_model_len"] = first["max_model_len"]
        return out

    def equivalent_command(self, spec: LaunchSpec) -> str:
        return spec.redacted_display_command
