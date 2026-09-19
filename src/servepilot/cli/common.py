"""Shared CLI plumbing: option parsing into configuration, workspace construction, output helpers."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console

from servepilot.cache.store import CacheStore
from servepilot.engines.base import InferenceEngine
from servepilot.engines.process import Launcher, LocalLauncher
from servepilot.engines.registry import (
    EngineRegistry,
    build_registry,
    select_engines,
)
from servepilot.exceptions import ConfigurationError, ServePilotError
from servepilot.hardware.base import HardwareProvider, get_hardware_provider
from servepilot.logging import get_logger
from servepilot.models.auth import hf_token as hf_token
from servepilot.models.inspector import ModelInspector
from servepilot.runtime.ports import PortAllocator
from servepilot.runtime.state import RuntimeStateStore
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import EngineName, PlanConstraints
from servepilot.schemas.workload import WorkloadProfile
from servepilot.settings import ServePilotConfig, ServePilotSettings, load_config

log = get_logger(__name__)
T = TypeVar("T")


# --------------------------------------------------------------------------- state shared via ctx
@dataclass
class CLIState:
    verbosity: int = 0
    log_format: str = "human"
    console: Console = field(default_factory=Console)
    err_console: Console = field(default_factory=lambda: Console(stderr=True))
    settings: ServePilotSettings = field(default_factory=ServePilotSettings)


def get_state(ctx: typer.Context) -> CLIState:
    state = ctx.find_root().obj
    if not isinstance(state, CLIState):
        state = CLIState()
        ctx.find_root().obj = state
    return state


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run ``coro`` on uvloop when available.

    The router proxy, the benchmark load generator and the engine supervisors all share this one
    event loop; libuv makes its socket I/O several times cheaper than the default selector loop.
    """
    try:
        import uvloop
    except ImportError:  # pragma: no cover - only absent on Windows
        return asyncio.run(coro)
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        return runner.run(coro)


def emit_json(payload: Any) -> None:
    """Write machine-readable JSON to stdout (progress/logs always go to stderr)."""
    sys.stdout.write(json.dumps(payload, indent=2, default=str) + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------- flags shared by plan/tune/serve
BASICS = "Basics"
TRAFFIC = "Traffic details"
HARDWARE = "Existing hardware"
ADVANCED = "Advanced"

ModelArg = Annotated[
    str | None,
    typer.Argument(help="Hugging Face model id or local model directory.", show_default=False),
]
ConfigOpt = Annotated[
    Path | None,
    typer.Option(
        "--config", "-c", help="YAML config file. Flags override it.", rich_help_panel=BASICS
    ),
]
EngineOpt = Annotated[
    str | None, typer.Option("--engine", help="auto | vllm | sglang", rich_help_panel=BASICS)
]
ObjectiveOpt = Annotated[
    str | None,
    typer.Option("--objective", help="throughput | latency | balanced", rich_help_panel=BASICS),
]
ProfileOpt = Annotated[
    str | None,
    typer.Option(
        "--profile", help="chat | long-context | decode-heavy | custom", rich_help_panel=BASICS
    ),
]
ExpectedConcurrencyOpt = Annotated[
    int | None,
    typer.Option(
        "--expected-concurrency",
        help="How many requests at once you expect. Needed for good latency tuning.",
        rich_help_panel=BASICS,
    ),
]
GPUsOpt = Annotated[
    str | None,
    typer.Option(
        "--gpus", help="GPU indices to use, like 0,1,2,3. Default: all.", rich_help_panel=BASICS
    ),
]
JSONOpt = Annotated[
    bool, typer.Option("--json", help="Machine-readable JSON on stdout.", rich_help_panel=BASICS)
]
HostOpt = Annotated[
    str | None,
    typer.Option(
        "--host", help="Bind address for the API (default 127.0.0.1).", rich_help_panel=BASICS
    ),
]
PortOpt = Annotated[
    int | None,
    typer.Option("--port", help="Port for the API (default 8000).", rich_help_panel=BASICS),
]

InputTokensOpt = Annotated[
    int | None,
    typer.Option(
        "--input-tokens", help="Typical prompt length in tokens (p50).", rich_help_panel=TRAFFIC
    ),
]
InputTokensP95Opt = Annotated[
    int | None,
    typer.Option(
        "--input-tokens-p95", help="Long prompt length in tokens (p95).", rich_help_panel=TRAFFIC
    ),
]
OutputTokensOpt = Annotated[
    int | None,
    typer.Option(
        "--output-tokens", help="Typical reply length in tokens (p50).", rich_help_panel=TRAFFIC
    ),
]
OutputTokensP95Opt = Annotated[
    int | None,
    typer.Option(
        "--output-tokens-p95", help="Long reply length in tokens (p95).", rich_help_panel=TRAFFIC
    ),
]
ContextLengthOpt = Annotated[
    int | None,
    typer.Option(
        "--context-length", help="Longest context to serve, in tokens.", rich_help_panel=TRAFFIC
    ),
]
RequestRateOpt = Annotated[
    float | None,
    typer.Option(
        "--request-rate",
        help="Requests per second for open-loop latency benchmarks.",
        rich_help_panel=TRAFFIC,
    ),
]
MaxTTFTOpt = Annotated[
    float | None,
    typer.Option(
        "--max-p95-ttft", help="Limit: p95 time to first token, in ms.", rich_help_panel=TRAFFIC
    ),
]
MaxLatencyOpt = Annotated[
    float | None,
    typer.Option(
        "--max-p95-latency", help="Limit: p95 end-to-end latency, in ms.", rich_help_panel=TRAFFIC
    ),
]
MaxTPOTOpt = Annotated[
    float | None,
    typer.Option(
        "--max-p95-tpot", help="Limit: p95 time per output token, in ms.", rich_help_panel=TRAFFIC
    ),
]
NoStreamOpt = Annotated[
    bool,
    typer.Option(
        "--no-stream", help="Benchmark with non-streaming requests.", rich_help_panel=TRAFFIC
    ),
]

RayAddressOpt = Annotated[
    str | None,
    typer.Option(
        "--ray-address",
        help="Use a Ray cluster you already started, like 'auto' or 10.0.0.1:6379.",
        rich_help_panel=HARDWARE,
    ),
]

TPOpt = Annotated[
    int | None,
    typer.Option("--tp", help="Force the tensor-parallel size.", rich_help_panel=ADVANCED),
]
ReplicasOpt = Annotated[
    int | None,
    typer.Option("--replicas", help="Force the number of model copies.", rich_help_panel=ADVANCED),
]
MaxConcurrencyOpt = Annotated[
    int | None,
    typer.Option(
        "--max-concurrency",
        help="Cap the requests one replica runs at once.",
        rich_help_panel=ADVANCED,
    ),
]
MemoryHeadroomOpt = Annotated[
    float | None,
    typer.Option(
        "--memory-headroom",
        help="Share of GPU memory to keep free (default 0.08).",
        rich_help_panel=ADVANCED,
    ),
]
MemoryFractionOpt = Annotated[
    float | None,
    typer.Option(
        "--memory-fraction", help="Force the engine memory fraction.", rich_help_panel=ADVANCED
    ),
]
AllowBusyOpt = Annotated[
    bool,
    typer.Option(
        "--allow-busy-gpus",
        help="Do not warn when other processes already use the GPUs.",
        rich_help_panel=ADVANCED,
    ),
]
AllowContextOverrideOpt = Annotated[
    bool,
    typer.Option(
        "--allow-context-override",
        help="Allow --context-length beyond the model's own maximum.",
        rich_help_panel=ADVANCED,
    ),
]
TrustRemoteCodeOpt = Annotated[
    bool,
    typer.Option(
        "--trust-remote-code",
        help="Pass trust_remote_code to the engine and tokenizer.",
        rich_help_panel=ADVANCED,
    ),
]
RevisionOpt = Annotated[
    str | None,
    typer.Option(
        "--revision", help="Model revision (branch, tag or commit).", rich_help_panel=ADVANCED
    ),
]
ServedModelNameOpt = Annotated[
    str | None,
    typer.Option(
        "--served-model-name",
        help="Public model name in /v1/models (default: the model id).",
        rich_help_panel=ADVANCED,
    ),
]
KVCacheDtypeOpt = Annotated[
    str | None,
    typer.Option(
        "--kv-cache-dtype",
        help="KV cache dtype for the engine, like fp8.",
        rich_help_panel=ADVANCED,
    ),
]
StartupTimeoutOpt = Annotated[
    float | None,
    typer.Option(
        "--startup-timeout",
        help="Seconds to wait for an engine to come up.",
        rich_help_panel=ADVANCED,
    ),
]


@dataclass
class PlanFlags:
    model: str | None
    config: Path | None = None
    engine: str | None = None
    objective: str | None = None
    profile: str | None = None
    input_tokens: int | None = None
    input_tokens_p95: int | None = None
    output_tokens: int | None = None
    output_tokens_p95: int | None = None
    context_length: int | None = None
    expected_concurrency: int | None = None
    request_rate: float | None = None
    max_p95_ttft: float | None = None
    max_p95_latency: float | None = None
    max_p95_tpot: float | None = None
    no_stream: bool = False
    gpus: str | None = None
    tp: int | None = None
    replicas: int | None = None
    max_concurrency: int | None = None
    memory_headroom: float | None = None
    memory_fraction: float | None = None
    allow_busy_gpus: bool = False
    allow_context_override: bool = False
    trust_remote_code: bool = False
    revision: str | None = None
    served_model_name: str | None = None
    kv_cache_dtype: str | None = None
    ray_address: str | None = None
    host: str | None = None
    port: int | None = None
    startup_timeout: float | None = None

    def overrides(self) -> dict[str, Any]:
        gpus: list[int] | None = None
        if self.gpus:
            try:
                gpus = sorted({int(x) for x in self.gpus.split(",") if x.strip()})
            except ValueError as exc:
                raise ConfigurationError(
                    f"--gpus must be comma-separated integers, got {self.gpus!r}"
                ) from exc
        if self.engine is not None and self.engine not in ("auto", "vllm", "sglang", "fake"):
            raise ConfigurationError(f"--engine must be auto, vllm or sglang (got {self.engine!r})")
        if self.objective is not None and self.objective not in (
            "throughput",
            "latency",
            "balanced",
        ):
            raise ConfigurationError(
                f"--objective must be throughput, latency or balanced (got {self.objective!r})"
            )
        custom_tokens = any(
            v is not None
            for v in (
                self.input_tokens,
                self.input_tokens_p95,
                self.output_tokens,
                self.output_tokens_p95,
            )
        )
        profile_name = self.profile
        if profile_name is None and custom_tokens:
            profile_name = "custom"
        # A p50 override without a p95 override scales the tail proportionally (4× / 3×, matching presets).
        in_p95 = self.input_tokens_p95
        if in_p95 is None and self.input_tokens is not None:
            in_p95 = self.input_tokens * 4
        out_p95 = self.output_tokens_p95
        if out_p95 is None and self.output_tokens is not None:
            out_p95 = self.output_tokens * 3
        return {
            "model": self.model,
            "engine": self.engine,
            "objective": self.objective,
            "profile.name": profile_name,
            "profile.input_tokens_p50": self.input_tokens,
            "profile.input_tokens_p95": in_p95,
            "profile.output_tokens_p50": self.output_tokens,
            "profile.output_tokens_p95": out_p95,
            "profile.max_context_tokens": self.context_length,
            "profile.expected_concurrency": self.expected_concurrency,
            "profile.target_request_rate": self.request_rate,
            "profile.max_p95_ttft_ms": self.max_p95_ttft,
            "profile.max_p95_latency_ms": self.max_p95_latency,
            "profile.max_p95_tpot_ms": self.max_p95_tpot,
            "profile.streaming": False if self.no_stream else None,
            "hardware.gpus": gpus,
            "hardware.memory_headroom": self.memory_headroom,
            "hardware.allow_busy_gpus": True if self.allow_busy_gpus else None,
            "constraints.tp": self.tp,
            "constraints.replicas": self.replicas,
            "constraints.context_length": self.context_length,
            "constraints.max_concurrency": self.max_concurrency,
            "constraints.memory_fraction": self.memory_fraction,
            "constraints.allow_context_override": True if self.allow_context_override else None,
            "model_options.trust_remote_code": True if self.trust_remote_code else None,
            "model_options.revision": self.revision,
            "model_options.kv_cache_dtype": self.kv_cache_dtype,
            "server.host": self.host,
            "server.port": self.port,
            "server.served_model_name": self.served_model_name,
            "tuning.startup_timeout_seconds": self.startup_timeout,
        }


def build_config(flags: PlanFlags) -> ServePilotConfig:
    config = load_config(flags.config).merged(flags.overrides())
    if not config.model:
        raise ConfigurationError(
            "a model is required: pass MODEL or set `model:` in the config file"
        )
    return config


def constraints_from_config(config: ServePilotConfig) -> PlanConstraints:
    c = config.constraints
    engine = None if config.engine == "auto" else EngineName(config.engine)
    return PlanConstraints(
        engine=engine,
        tensor_parallel_size=c.tp,
        replica_count=c.replicas,
        gpu_ids=None if config.hardware.gpus == "auto" else list(config.hardware.gpus),
        context_length=c.context_length,
        max_concurrency=c.max_concurrency,
        memory_headroom=config.hardware.memory_headroom,
        memory_fraction=c.memory_fraction,
        kv_cache_dtype=config.model_options.kv_cache_dtype,
        allow_busy_gpus=config.hardware.allow_busy_gpus,
        allow_context_override=c.allow_context_override,
        trust_remote_code=config.model_options.trust_remote_code,
        served_model_name=config.server.served_model_name,
        revision=config.model_options.revision,
        extra_engine_args=dict(c.engine_args),
    )


# --------------------------------------------------------------------------- workspace
@dataclass
class Workspace:
    settings: ServePilotSettings
    config: ServePilotConfig
    hardware_provider: HardwareProvider
    hardware: HardwareSnapshot
    model: ModelProfile
    workload: WorkloadProfile
    registry: EngineRegistry
    engines: list[InferenceEngine]
    constraints: PlanConstraints
    cache: CacheStore
    state_store: RuntimeStateStore
    ports: PortAllocator
    launcher: Launcher
    ray_address: str | None = None

    @property
    def served_model_name(self) -> str:
        return self.config.server.served_model_name or self.model.model_id


def make_hardware_provider(
    settings: ServePilotSettings, ray_address: str | None
) -> HardwareProvider:
    if ray_address:
        from servepilot.cluster.ray_provider import RayHardwareProvider

        return RayHardwareProvider(ray_address)
    return get_hardware_provider(settings)


def make_launcher(ray_address: str | None, on_line: Any = None) -> Launcher:
    if ray_address:
        from servepilot.cluster.ray_launcher import RayLauncher

        return RayLauncher(ray_address)
    return LocalLauncher(on_line=on_line)


def build_workspace(flags: PlanFlags, state: CLIState, *, on_engine_line: Any = None) -> Workspace:
    settings = state.settings
    config = build_config(flags)
    assert config.model is not None
    if flags.ray_address and "RAY_ADDRESS" not in os.environ:
        os.environ["RAY_ADDRESS"] = flags.ray_address
    provider = make_hardware_provider(settings, flags.ray_address)
    hardware = provider.snapshot()
    inspector = ModelInspector(token=hf_token())
    model = inspector.inspect(config.model, revision=config.model_options.revision)
    workload = config.build_workload()
    registry = build_registry(settings)
    engines = select_engines(registry, config.engine)
    constraints = constraints_from_config(config)
    return Workspace(
        settings=settings,
        config=config,
        hardware_provider=provider,
        hardware=hardware,
        model=model,
        workload=workload,
        registry=registry,
        engines=engines,
        constraints=constraints,
        cache=CacheStore(settings.cache_dir),
        state_store=RuntimeStateStore(settings.state_dir),
        ports=PortAllocator(settings.backend_port_start, settings.backend_port_end),
        launcher=make_launcher(flags.ray_address, on_engine_line),
        ray_address=flags.ray_address,
    )


def fail(exc: ServePilotError, console: Console) -> None:
    console.print(f"[bold red]error:[/] {exc.render()}", highlight=False)
