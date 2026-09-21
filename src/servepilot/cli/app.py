"""Typer application: every ``servepilot`` subcommand."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, TypeVar

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from servepilot import __version__
from servepilot.cli import doctor as doctor_checks
from servepilot.cli.common import (
    AllowBusyOpt,
    AllowContextOverrideOpt,
    CLIState,
    ConfigOpt,
    ContextLengthOpt,
    EngineOpt,
    ExpectedConcurrencyOpt,
    GPUsOpt,
    HostOpt,
    InputTokensOpt,
    InputTokensP95Opt,
    JSONOpt,
    KVCacheDtypeOpt,
    MaxConcurrencyOpt,
    MaxLatencyOpt,
    MaxTPOTOpt,
    MaxTTFTOpt,
    MemoryFractionOpt,
    MemoryHeadroomOpt,
    ModelArg,
    NoStreamOpt,
    ObjectiveOpt,
    OutputTokensOpt,
    OutputTokensP95Opt,
    PlanFlags,
    PortOpt,
    ProfileOpt,
    RayAddressOpt,
    ReplicasOpt,
    RequestRateOpt,
    RevisionOpt,
    ServedModelNameOpt,
    StartupTimeoutOpt,
    TPOpt,
    TrustRemoteCodeOpt,
    Workspace,
    build_workspace,
    emit_json,
    fail,
    get_state,
    hf_token,
    make_hardware_provider,
    run_async,
)
from servepilot.cli.optimize_cmd import register as register_optimization
from servepilot.cli.pipeline import (
    equivalent_commands,
    lookup_cache,
    run_plan,
    run_tune,
)
from servepilot.cli.render import (
    render_hardware,
    render_model,
    render_pareto,
    render_plan,
    render_results_table,
    render_selected,
    render_workload,
)
from servepilot.constants import DEFAULT_PUBLIC_PORT, ExitCode
from servepilot.engines.registry import build_registry
from servepilot.exceptions import (
    ConfigurationError,
    EngineUnavailableError,
    RuntimeStateError,
    ServePilotError,
)
from servepilot.logging import configure_logging, get_logger
from servepilot.models.inspector import ModelInspector
from servepilot.planner.planner import Planner
from servepilot.runtime.replicas import ReplicaSet
from servepilot.runtime.router import ReplicaRouter
from servepilot.runtime.supervisor import Deployment
from servepilot.schemas.plan import SelectedPlan

log = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])

app = typer.Typer(
    name="servepilot",
    help="Open Sandbox: optimize and serve models on your existing NVIDIA GPUs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)
inspect_app = typer.Typer(
    help="Inspect hardware or a model without launching anything.", no_args_is_help=True
)
cache_app = typer.Typer(help="Manage cached tuning results.", no_args_is_help=True)
app.add_typer(inspect_app, name="inspect")
app.add_typer(cache_app, name="cache")


# --------------------------------------------------------------------------- error handling
def handle_errors(func: F) -> F:
    """Render ServePilot errors cleanly and map them to exit codes; tracebacks only with -vv."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx: typer.Context | None = next(
            (a for a in args if isinstance(a, typer.Context)), None
        ) or kwargs.get("ctx")
        state = get_state(ctx) if ctx is not None else CLIState()
        try:
            return func(*args, **kwargs)
        except ServePilotError as exc:
            fail(exc, state.err_console)
            if state.verbosity >= 2:
                state.err_console.print_exception()
            raise typer.Exit(code=int(exc.exit_code)) from None
        except KeyboardInterrupt:
            state.err_console.print("\n[yellow]interrupted[/]")
            raise typer.Exit(code=int(ExitCode.INTERRUPTED)) from None
        except typer.Exit:
            raise
        except Exception as exc:
            state.err_console.print(f"[bold red]unexpected error:[/] {type(exc).__name__}: {exc}")
            if state.verbosity >= 2:
                state.err_console.print_exception()
            else:
                state.err_console.print("[dim]re-run with -vv for a traceback[/]")
            raise typer.Exit(code=int(ExitCode.UNEXPECTED_ERROR)) from None

    return wrapper  # type: ignore[return-value]


@app.callback()
def main_callback(
    ctx: typer.Context,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose", "-v", count=True, help="Increase log verbosity (-v info, -vv debug)."
        ),
    ] = 0,
    log_format: Annotated[str, typer.Option("--log-format", help="human | json")] = "human",
) -> None:
    if log_format not in ("human", "json"):
        raise typer.BadParameter("--log-format must be 'human' or 'json'")
    configure_logging(verbose, log_format)  # type: ignore[arg-type]
    state = CLIState(verbosity=verbose, log_format=log_format)
    ctx.obj = state


def _flags(**kwargs: Any) -> PlanFlags:
    return PlanFlags(**kwargs)


def _engine_line_printer(state: CLIState) -> Callable[[str, str], None] | None:
    if state.verbosity < 2:
        return None

    def printer(stream: str, line: str) -> None:
        state.err_console.print(f"[dim]engine {stream}:[/] {line}", markup=False, highlight=False)

    return printer


# --------------------------------------------------------------------------- version / doctor
@app.command()
def version(json_output: JSONOpt = False) -> None:
    """Print the ServePilot version."""
    if json_output:
        emit_json({"version": __version__})
    else:
        typer.echo(f"servepilot {__version__}")


@app.command()
@handle_errors
def doctor(
    ctx: typer.Context,
    port: Annotated[
        int, typer.Option("--port", help="Public port to check.")
    ] = DEFAULT_PUBLIC_PORT,
    ray_address: RayAddressOpt = None,
    nodes: Annotated[
        Path | None,
        typer.Option("--nodes", help="Check existing SSH GPU workers from this controller."),
    ] = None,
    json_output: JSONOpt = False,
) -> None:
    """Verify the environment: OS, Python, NVIDIA/NVML, engines, Hugging Face access, ports."""
    state = get_state(ctx)
    if nodes is not None:
        if ray_address:
            raise ConfigurationError("choose --nodes or --ray-address")
        from servepilot.cluster.diagnostics import diagnose
        from servepilot.cluster.inventory import NodeInventory

        diagnostics = run_async(diagnose(NodeInventory.load(nodes)))
        if json_output:
            emit_json(diagnostics)
        else:
            state.console.print(f"Open Sandbox cluster checks: {diagnostics['status']}")
            for check in diagnostics["checks"]:
                state.console.print(
                    f"  {check.get('node', 'controller')}: {check['check']} — {check['detail']}"
                )
            state.console.print("Controller connectivity and pairwise TCP bandwidth checks passed.")
        return
    provider = make_hardware_provider(state.settings, ray_address)
    registry = build_registry(state.settings)
    checks = doctor_checks.run_all(provider, registry, state.settings, public_port=port)
    overall = doctor_checks.overall_status(checks)
    if json_output:
        emit_json({"status": overall, "checks": [c.__dict__ for c in checks]})
    else:
        console = state.console
        console.print("[bold]ServePilot doctor[/]\n")
        section = None
        icons = {
            "ok": "[green]✓[/]",
            "warn": "[yellow]![/]",
            "fail": "[red]✗[/]",
            "info": "[blue]i[/]",
        }
        for c in checks:
            if c.section != section:
                section = c.section
                console.print(f"[bold]{section}[/]")
            line = f"  {icons[c.status]} {c.label}"
            if c.detail:
                line += f"  [dim]{c.detail}[/]"
            console.print(line)
            for hint in c.hints:
                console.print(f"      → {hint}")
        console.print()
        if overall == "ok":
            console.print("[green]ServePilot is ready.[/]")
        elif overall == "warn":
            console.print("[yellow]ServePilot can run, but review the warnings above.[/]")
        else:
            console.print(
                "[red]ServePilot cannot serve models until the failures above are fixed.[/]"
            )
    if overall == "fail":
        raise typer.Exit(code=int(ExitCode.ENVIRONMENT_ERROR))


# --------------------------------------------------------------------------- inspect
@inspect_app.command("hardware")
@handle_errors
def inspect_hardware(
    ctx: typer.Context,
    ray_address: RayAddressOpt = None,
    json_output: JSONOpt = False,
    nodes: Annotated[
        Path | None, typer.Option("--nodes", help="Inspect existing SSH GPU workers.")
    ] = None,
) -> None:
    """Show GPUs, memory and interconnect topology."""
    state = get_state(ctx)
    if nodes is not None:
        if ray_address:
            raise ConfigurationError("choose --nodes or --ray-address")
        from servepilot.cluster.inventory import NodeInventory
        from servepilot.cluster.ssh_provider import SSHHardwareProvider

        snap = SSHHardwareProvider(NodeInventory.load(nodes)).snapshot()
    else:
        snap = make_hardware_provider(state.settings, ray_address).snapshot()
    if json_output:
        emit_json(snap.model_dump(mode="json"))
    else:
        render_hardware(state.console, snap)


@inspect_app.command("model")
@handle_errors
def inspect_model(
    ctx: typer.Context,
    model: Annotated[str, typer.Argument(help="Hugging Face model id or local directory.")],
    revision: RevisionOpt = None,
    json_output: JSONOpt = False,
) -> None:
    """Show the normalized model profile (architecture, dtype, weight size, KV shape)."""
    state = get_state(ctx)
    profile = ModelInspector(token=hf_token()).inspect(model, revision=revision)
    if json_output:
        emit_json(profile.model_dump(mode="json"))
    else:
        render_model(state.console, profile)


# --------------------------------------------------------------------------- plan
@app.command()
@handle_errors
def plan(
    ctx: typer.Context,
    model: ModelArg = None,
    config: ConfigOpt = None,
    engine: EngineOpt = None,
    objective: ObjectiveOpt = None,
    profile: ProfileOpt = None,
    input_tokens: InputTokensOpt = None,
    input_tokens_p95: InputTokensP95Opt = None,
    output_tokens: OutputTokensOpt = None,
    output_tokens_p95: OutputTokensP95Opt = None,
    context_length: ContextLengthOpt = None,
    expected_concurrency: ExpectedConcurrencyOpt = None,
    request_rate: RequestRateOpt = None,
    max_p95_ttft: MaxTTFTOpt = None,
    max_p95_latency: MaxLatencyOpt = None,
    max_p95_tpot: MaxTPOTOpt = None,
    no_stream: NoStreamOpt = False,
    gpus: GPUsOpt = None,
    tp: TPOpt = None,
    replicas: ReplicasOpt = None,
    max_concurrency: MaxConcurrencyOpt = None,
    memory_headroom: MemoryHeadroomOpt = None,
    memory_fraction: MemoryFractionOpt = None,
    allow_busy_gpus: AllowBusyOpt = False,
    allow_context_override: AllowContextOverrideOpt = False,
    trust_remote_code: TrustRemoteCodeOpt = False,
    revision: RevisionOpt = None,
    kv_cache_dtype: KVCacheDtypeOpt = None,
    ray_address: RayAddressOpt = None,
    json_output: JSONOpt = False,
) -> None:
    """Show which layouts can work on existing GPUs without launching inference."""
    state = get_state(ctx)
    flags = _flags(**{k: v for k, v in locals().items() if k in PlanFlags.__dataclass_fields__})
    ws = build_workspace(flags, state)
    result, explanation = run_plan(ws)
    if json_output:
        emit_json(
            {
                "hardware": ws.hardware.model_dump(mode="json"),
                "model": ws.model.model_dump(mode="json"),
                "workload": ws.workload.model_dump(mode="json"),
                "planning": result.model_dump(mode="json"),
                "explanation": explanation,
            }
        )
        return
    console = state.console
    render_hardware(console, ws.hardware)
    console.print()
    render_model(console, ws.model)
    console.print()
    render_workload(console, ws.workload)
    render_plan(console, result, explanation)


# --------------------------------------------------------------------------- tune
@app.command()
@handle_errors
def tune(
    ctx: typer.Context,
    model: ModelArg = None,
    config: ConfigOpt = None,
    engine: EngineOpt = None,
    objective: ObjectiveOpt = None,
    profile: ProfileOpt = None,
    input_tokens: InputTokensOpt = None,
    input_tokens_p95: InputTokensP95Opt = None,
    output_tokens: OutputTokensOpt = None,
    output_tokens_p95: OutputTokensP95Opt = None,
    context_length: ContextLengthOpt = None,
    expected_concurrency: ExpectedConcurrencyOpt = None,
    request_rate: RequestRateOpt = None,
    max_p95_ttft: MaxTTFTOpt = None,
    max_p95_latency: MaxLatencyOpt = None,
    max_p95_tpot: MaxTPOTOpt = None,
    no_stream: NoStreamOpt = False,
    gpus: GPUsOpt = None,
    tp: TPOpt = None,
    replicas: ReplicasOpt = None,
    max_concurrency: MaxConcurrencyOpt = None,
    memory_headroom: MemoryHeadroomOpt = None,
    memory_fraction: MemoryFractionOpt = None,
    allow_busy_gpus: AllowBusyOpt = False,
    allow_context_override: AllowContextOverrideOpt = False,
    trust_remote_code: TrustRemoteCodeOpt = False,
    revision: RevisionOpt = None,
    served_model_name: ServedModelNameOpt = None,
    kv_cache_dtype: KVCacheDtypeOpt = None,
    ray_address: RayAddressOpt = None,
    startup_timeout: StartupTimeoutOpt = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the full tuning record (JSON) to this file."),
    ] = None,
    show_pareto: Annotated[
        bool, typer.Option("--show-pareto", help="Print the throughput/latency Pareto frontier.")
    ] = False,
    resume: Annotated[
        bool, typer.Option("--resume", help="Reuse completed candidates from an interrupted run.")
    ] = False,
    json_output: JSONOpt = False,
) -> None:
    """Launch and benchmark each layout that can work, save the winner, exit. Nothing keeps running."""
    state = get_state(ctx)
    flags = _flags(**{k: v for k, v in locals().items() if k in PlanFlags.__dataclass_fields__})
    ws = build_workspace(flags, state, on_engine_line=_engine_line_printer(state))
    console = state.err_console if json_output else state.console
    planning, explanation = run_plan(ws)
    render_hardware(console, ws.hardware)
    console.print()
    render_model(console, ws.model)
    console.print()
    render_workload(console, ws.workload)
    render_plan(console, planning, explanation)
    run = run_async(
        run_tune(ws, planning, console=console, resume=resume or ws.config.tuning.resume)
    )
    console.print()
    render_results_table(console, run.outcome.evaluations)
    console.print()
    render_selected(console, run.outcome.winner, commands=run.outcome.winner.equivalent_commands)
    if show_pareto:
        console.print()
        render_pareto(console, run.outcome.winner.pareto_front)
    console.print(f"\nSaved tuning profile: {run.path}")
    if output is not None:
        output.write_text(
            json.dumps(run.record.model_dump(mode="json"), indent=2, default=str), encoding="utf-8"
        )
        console.print(f"Wrote tuning record to {output}")
    if json_output:
        emit_json(run.record.model_dump(mode="json"))


# --------------------------------------------------------------------------- serve
@app.command()
@handle_errors
def serve(
    ctx: typer.Context,
    model: ModelArg = None,
    config: ConfigOpt = None,
    engine: EngineOpt = None,
    objective: ObjectiveOpt = None,
    profile: ProfileOpt = None,
    input_tokens: InputTokensOpt = None,
    input_tokens_p95: InputTokensP95Opt = None,
    output_tokens: OutputTokensOpt = None,
    output_tokens_p95: OutputTokensP95Opt = None,
    context_length: ContextLengthOpt = None,
    expected_concurrency: ExpectedConcurrencyOpt = None,
    request_rate: RequestRateOpt = None,
    max_p95_ttft: MaxTTFTOpt = None,
    max_p95_latency: MaxLatencyOpt = None,
    max_p95_tpot: MaxTPOTOpt = None,
    no_stream: NoStreamOpt = False,
    gpus: GPUsOpt = None,
    tp: TPOpt = None,
    replicas: ReplicasOpt = None,
    max_concurrency: MaxConcurrencyOpt = None,
    memory_headroom: MemoryHeadroomOpt = None,
    memory_fraction: MemoryFractionOpt = None,
    allow_busy_gpus: AllowBusyOpt = False,
    allow_context_override: AllowContextOverrideOpt = False,
    trust_remote_code: TrustRemoteCodeOpt = False,
    revision: RevisionOpt = None,
    served_model_name: ServedModelNameOpt = None,
    kv_cache_dtype: KVCacheDtypeOpt = None,
    ray_address: RayAddressOpt = None,
    host: HostOpt = None,
    port: PortOpt = None,
    startup_timeout: StartupTimeoutOpt = None,
    retune: Annotated[
        bool, typer.Option("--retune", help="Ignore any cached tuning result and tune again.")
    ] = False,
    no_tune: Annotated[
        bool,
        typer.Option(
            "--no-tune", help="Serve the heuristic plan without benchmarking (clearly labelled)."
        ),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be launched and exit.")
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Reuse completed candidates from an interrupted tuning run."),
    ] = False,
    json_output: JSONOpt = False,
) -> None:
    """Tune (or reuse a saved result), then serve the model at one OpenAI-compatible URL."""
    state = get_state(ctx)
    flags = _flags(**{k: v for k, v in locals().items() if k in PlanFlags.__dataclass_fields__})
    ws = build_workspace(flags, state, on_engine_line=_engine_line_printer(state))
    console = state.err_console if json_output else state.console
    if not dry_run:
        # Refuse before tuning: a live deployment would both block the port later and share
        # the GPUs with the candidates being measured.
        _refuse_if_deployment_live(ws)
    console.print(f"[bold]ServePilot {__version__}[/]\n")
    console.print("Inspecting hardware...")
    render_hardware(console, ws.hardware)
    console.print("\nInspecting model...")
    render_model(console, ws.model)
    console.print()
    render_workload(console, ws.workload)
    console.print("\nGenerating candidates...")
    planning, explanation = run_plan(ws)
    render_plan(console, planning, explanation)

    selected = _select_plan(
        ws,
        planning,
        console,
        retune=retune,
        no_tune=no_tune,
        resume=resume or ws.config.tuning.resume,
        use_cache=ws.config.tuning.use_cache,
        dry_run=dry_run,
    )
    if not selected.equivalent_commands:
        selected.equivalent_commands = equivalent_commands(ws, selected)
    console.print()
    render_selected(console, selected, commands=selected.equivalent_commands)

    if dry_run:
        engine_obj = ws.registry.require(selected.plan.engine)
        replica_set = ReplicaSet(
            plan=selected.plan,
            model=ws.model,
            engine=engine_obj,
            launcher=ws.launcher,
            ports=ws.ports,
            hardware=ws.hardware,
            host=ws.config.server.host,
            served_model_name=ws.served_model_name,
            trust_remote_code=ws.constraints.trust_remote_code,
            router=ReplicaRouter(),
        )
        specs = replica_set.specs()
        console.print("\n[bold]Dry run[/] – nothing was launched.")
        table = Table(title="Planned launches", expand=False)
        table.add_column("Replica")
        table.add_column("GPUs")
        table.add_column("Node")
        table.add_column("Port", justify="right")
        for spec in specs:
            table.add_row(
                spec.replica_id, str(spec.gpu_ids), spec.node_ip or "local", str(spec.port)
            )
        console.print(table)
        console.print(f"Public endpoint: http://{ws.config.server.host}:{ws.config.server.port}/v1")
        if json_output:
            emit_json(
                {
                    "selected": selected.model_dump(mode="json"),
                    "launches": [s.model_dump(mode="json", exclude={"env"}) for s in specs],
                    "public_endpoint": f"http://{ws.config.server.host}:{ws.config.server.port}/v1",
                }
            )
        return

    engine_obj = ws.registry.require(selected.plan.engine)
    deployment = Deployment(
        selected=selected,
        model=ws.model,
        engine=engine_obj,
        launcher=ws.launcher,
        ports=ws.ports,
        hardware=ws.hardware,
        hardware_provider=ws.hardware_provider,
        state_store=ws.state_store,
        host=ws.config.server.host,
        port=ws.config.server.port,
        served_model_name=ws.served_model_name,
        trust_remote_code=ws.constraints.trust_remote_code,
        max_queue_depth=ws.config.server.max_queue_depth,
        startup_timeout=ws.config.tuning.startup_timeout_seconds,
        on_event=lambda msg: console.print(f"[dim]{msg}[/]"),
    )
    _refuse_if_deployment_live(ws)  # tuning may have taken a while; check again before binding
    console.print(f"\nStarting {selected.plan.replica_count} replica(s)... (Ctrl-C to stop)")

    async def _serve() -> None:
        await deployment.run()

    if json_output:
        emit_json(
            {
                "selected": selected.model_dump(mode="json"),
                "public_endpoint": f"http://{ws.config.server.host}:{ws.config.server.port}/v1",
            }
        )
    run_async(_serve())


def _refuse_if_deployment_live(ws: Workspace) -> None:
    existing = ws.state_store.read()
    if existing is not None and ws.state_store.is_live(existing):
        raise RuntimeStateError(
            f"a ServePilot deployment is already running (pid {existing.servepilot_pid}, port {existing.public_port}).",
            hints=["Run `servepilot stop` first, or use a different SERVEPILOT_STATE_DIR."],
        )


def _select_plan(
    ws: Workspace,
    planning: Any,
    console: Console,
    *,
    retune: bool,
    no_tune: bool,
    resume: bool,
    use_cache: bool,
    dry_run: bool = False,
) -> SelectedPlan:
    if no_tune:
        console.print("\n[yellow]--no-tune:[/] serving the heuristic plan without benchmarking.")
        return Planner.heuristic_selection(planning, ws.workload)
    if use_cache and not retune:
        lookup = lookup_cache(ws, planning)
        if lookup.record is not None and lookup.validation is not None:
            if lookup.validation.valid and lookup.record.winner is not None:
                console.print(
                    f"\nUsing cached tuning result {lookup.key} from {lookup.record.created_at:%Y-%m-%d %H:%M} UTC."
                )
                cached = lookup.record.winner.model_copy(deep=True)
                cached.source = "cached"
                return cached
            console.print("\nCached tuning result cannot be reused:")
            for reason in lookup.validation.reasons:
                console.print(f"  - {reason}")
    if dry_run:
        console.print(
            "\n[yellow]Dry run:[/] showing the heuristic plan; tuning runs when serving starts."
        )
        return Planner.heuristic_selection(planning, ws.workload)
    if not ws.config.tuning.enabled:
        console.print(
            "\n[yellow]tuning disabled in config:[/] serving the heuristic plan without benchmarking."
        )
        return Planner.heuristic_selection(planning, ws.workload)
    console.print("\nBenchmarking candidates...")
    run = run_async(run_tune(ws, planning, console=console, resume=resume))
    console.print()
    render_results_table(console, run.outcome.evaluations)
    console.print(f"Saved tuning profile: {run.path}")
    return run.outcome.winner


# --------------------------------------------------------------------------- benchmark
@app.command()
@handle_errors
def benchmark(
    ctx: typer.Context,
    url: Annotated[
        str,
        typer.Argument(help="Base URL of an OpenAI-compatible server, e.g. http://127.0.0.1:8000"),
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name to request (default: first entry of /v1/models)."),
    ] = None,
    tokenizer_model: Annotated[
        str | None,
        typer.Option(
            "--tokenizer",
            help="Model id/path whose tokenizer generates prompts (default: --model).",
        ),
    ] = None,
    profile: ProfileOpt = None,
    input_tokens: InputTokensOpt = None,
    input_tokens_p95: InputTokensP95Opt = None,
    output_tokens: OutputTokensOpt = None,
    output_tokens_p95: OutputTokensP95Opt = None,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Concurrent requests.")] = 16,
    num_requests: Annotated[int, typer.Option("--num-requests", help="Total requests.")] = 64,
    request_rate: RequestRateOpt = None,
    no_stream: NoStreamOpt = False,
    completions: Annotated[
        bool, typer.Option("--completions", help="Use /v1/completions instead of chat.")
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="OPENAI_API_KEY", help="Bearer token for the server."),
    ] = None,
    no_ignore_eos: Annotated[
        bool,
        typer.Option(
            "--no-ignore-eos",
            help="Do not send the ignore_eos extension (for non-vLLM/SGLang servers).",
        ),
    ] = False,
    seed: Annotated[int, typer.Option("--seed")] = 1234,
    json_output: JSONOpt = False,
) -> None:
    """Benchmark any OpenAI-compatible endpoint with ServePilot's workload generator."""
    from servepilot.benchmark.runner import BenchmarkRunner, make_spec
    from servepilot.cli.common import build_config
    from servepilot.schemas.model import ModelProfile

    state = get_state(ctx)
    console = state.err_console if json_output else state.console
    flags = PlanFlags(
        model=model or "benchmark-target",
        profile=profile,
        input_tokens=input_tokens,
        input_tokens_p95=input_tokens_p95,
        output_tokens=output_tokens,
        output_tokens_p95=output_tokens_p95,
        request_rate=request_rate,
        no_stream=no_stream,
    )
    workload = build_config(flags).build_workload()
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    if model is None:
        try:
            data = httpx.get(f"{base}/v1/models", timeout=10.0).json()
            model = data["data"][0]["id"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ConfigurationError(
                f"could not discover a model name from {base}/v1/models: {exc}",
                hints=["Pass --model explicitly."],
            ) from exc
    assert model is not None
    tok_source = tokenizer_model or model
    try:
        profile_obj = ModelInspector(token=hf_token()).inspect(tok_source)
    except ServePilotError:
        profile_obj = ModelProfile(model_id=tok_source)
    from servepilot.models.tokenizer import load_tokenizer

    tokenizer = load_tokenizer(profile_obj, token=hf_token())
    spec = make_spec(
        workload,
        concurrency=concurrency,
        num_requests=num_requests,
        seed=seed,
        label="cli",
        request_rate=request_rate,
        endpoint="completions" if completions else "chat",
    )
    runner = BenchmarkRunner(
        tokenizer=tokenizer, workload=workload, api_key=api_key, ignore_eos=not no_ignore_eos
    )
    console.print(
        f"Benchmarking {base} model={model} concurrency={concurrency} requests={num_requests} tokenizer={tokenizer.name}"
    )
    result = run_async(runner.run(base, model, spec, candidate_id="external"))
    if json_output:
        emit_json(result.model_dump(mode="json"))
        return
    table = Table(title="Benchmark result", show_header=False)
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right")
    rows = [
        ("requests (ok/failed)", f"{result.successful_requests}/{result.failed_requests}"),
        ("duration", f"{result.duration_seconds:.1f} s"),
        ("request throughput", f"{result.request_throughput:.2f} req/s"),
        ("output tokens/s", f"{result.output_tokens_per_second:,.0f}"),
        ("total tokens/s", f"{result.total_tokens_per_second:,.0f}"),
        (
            "TTFT p50/p95/p99",
            " / ".join(
                f"{v:,.0f} ms" if v is not None else "-"
                for v in (result.ttft_p50_ms, result.ttft_p95_ms, result.ttft_p99_ms)
            ),
        ),
        (
            "TPOT p50/p95/p99",
            " / ".join(
                f"{v:.1f} ms" if v is not None else "-"
                for v in (result.tpot_p50_ms, result.tpot_p95_ms, result.tpot_p99_ms)
            ),
        ),
        (
            "latency p50/p95/p99",
            " / ".join(
                f"{v:,.0f} ms"
                for v in (result.latency_p50_ms, result.latency_p95_ms, result.latency_p99_ms)
            ),
        ),
        ("error rate", f"{result.error_rate:.1%}"),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    if result.errors_sample:
        console.print("errors: " + "; ".join(result.errors_sample))


# --------------------------------------------------------------------------- status / stop
@app.command()
@handle_errors
def status(ctx: typer.Context, json_output: JSONOpt = False) -> None:
    """Show the running deployment (from runtime state and the live /status endpoint)."""
    from servepilot.runtime.state import RuntimeStateStore

    state = get_state(ctx)
    from servepilot.optimization.controller_state import ControllerState

    controller = ControllerState(state.settings.state_dir).read()
    if controller and controller["running"]:
        if json_output:
            emit_json(controller)
        else:
            state.console.print(
                f"Open Sandbox {controller['phase']}: {controller['message']}\nOutput: {controller['output']}\nPID: {controller['pid']}"
            )
        return
    store = RuntimeStateStore(state.settings.state_dir)
    rt = store.read()
    if rt is None:
        if json_output:
            emit_json({"running": False})
        else:
            state.console.print("No ServePilot deployment is recorded.")
        return
    live = store.is_live(rt)
    payload: dict[str, Any] = {"running": live, "state": rt.model_dump(mode="json")}
    if live:
        try:
            resp = httpx.get(
                f"http://{'127.0.0.1' if rt.public_host in ('0.0.0.0', '::') else rt.public_host}:{rt.public_port}/status",
                timeout=5.0,
            )
            payload["status"] = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            payload["status_error"] = str(exc)
    if json_output:
        emit_json(payload)
        return
    console = state.console
    if not live:
        console.print(
            f"[yellow]Stale runtime state:[/] ServePilot pid {rt.servepilot_pid} is no longer running."
        )
        leftovers = store.live_children(rt)
        if leftovers:
            console.print(
                f"[red]{len(leftovers)} engine process(es) from that deployment are still alive[/]; run `servepilot stop`."
            )
        else:
            console.print("Run `servepilot stop` to clear the stale state.")
        return
    lines = [
        f"model: {rt.model} (served as {rt.served_model_name})",
        f"endpoint: http://{rt.public_host}:{rt.public_port}/v1",
        f"pid: {rt.servepilot_pid} · started {rt.started_at:%Y-%m-%d %H:%M:%S} UTC",
        f"plan: {rt.plan_id} · engine {rt.engine} · TP={rt.tensor_parallel_size} · {rt.replica_count} replica(s) on GPUs {rt.gpu_groups}",
        f"backend ports: {rt.backend_ports}",
    ]
    live_status = payload.get("status")
    if isinstance(live_status, dict):
        router = live_status.get("router", {})
        lines.append(
            f"replicas healthy: {router.get('healthy_replicas')}/{router.get('total_replicas')} · in flight {router.get('inflight')} · queued {router.get('queue_depth')}"
        )
        lines.append(
            f"requests total: {router.get('total_requests')} · errors {router.get('total_errors')} · uptime {live_status.get('uptime_seconds')} s"
        )
    console.print(Panel("\n".join(lines), title="ServePilot status", expand=False))


@app.command()
@handle_errors
def stop(ctx: typer.Context, json_output: JSONOpt = False) -> None:
    """Stop the running deployment and its engine processes."""
    from servepilot.runtime.state import RuntimeStateStore

    state = get_state(ctx)
    from servepilot.optimization.controller_state import ControllerState

    controller = ControllerState(state.settings.state_dir).stop()
    if controller:
        if json_output:
            emit_json(controller)
        else:
            state.console.print(controller["message"])
        if not controller["stopped"]:
            raise typer.Exit(code=int(ExitCode.RUNTIME_FAILURE))
        return
    store = RuntimeStateStore(state.settings.state_dir)
    rt = store.read()
    if rt is None:
        if json_output:
            emit_json({"stopped": False, "reason": "no deployment recorded"})
        else:
            state.console.print("No ServePilot deployment is recorded.")
        return
    stopped, messages = store.stop(rt)
    if json_output:
        emit_json({"stopped": stopped, "messages": messages})
        return
    for m in messages:
        state.console.print(f"  {m}")
    state.console.print(
        "[green]Deployment stopped.[/]"
        if stopped
        else "[red]Some processes could not be stopped.[/]"
    )
    if not stopped:
        raise typer.Exit(code=int(ExitCode.RUNTIME_FAILURE))


# --------------------------------------------------------------------------- cache
@cache_app.command("list")
@handle_errors
def cache_list(ctx: typer.Context, json_output: JSONOpt = False) -> None:
    """List cached tuning records."""
    from servepilot.cache.store import CacheStore

    state = get_state(ctx)
    store = CacheStore(state.settings.cache_dir)
    records = store.list()
    if json_output:
        emit_json(
            [
                {
                    "key": r.key,
                    "model": r.model_profile.model_id,
                    "revision": r.model_profile.revision,
                    "created_at": r.created_at.isoformat(),
                    "status": r.status,
                    "objective": r.workload_profile.objective.value,
                    "profile": r.workload_profile.name,
                    "winner": r.winner.plan.label() if r.winner else None,
                    "gpus": [g.name for g in r.hardware_snapshot.gpus][:1],
                    "gpu_count": r.hardware_snapshot.gpu_count,
                }
                for r in records
            ]
        )
        return
    if not records:
        state.console.print(f"No tuning records in {store.dir}")
        return
    table = Table(title=f"Tuning cache ({store.dir})", expand=False)
    table.add_column("Key")
    table.add_column("Model")
    table.add_column("Hardware")
    table.add_column("Workload")
    table.add_column("Winner")
    table.add_column("Status")
    table.add_column("Created")
    for r in records:
        hw = r.hardware_snapshot
        table.add_row(
            r.key,
            r.model_profile.model_id,
            f"{hw.gpu_count} × {hw.gpus[0].name if hw.gpus else '?'}",
            f"{r.workload_profile.name}/{r.workload_profile.objective.value}",
            r.winner.plan.label() if r.winner else "-",
            r.status,
            f"{r.created_at:%Y-%m-%d %H:%M}",
        )
    state.console.print(table)


@cache_app.command("show")
@handle_errors
def cache_show(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Record key from `servepilot cache list`.")],
    json_output: JSONOpt = False,
) -> None:
    """Show one cached tuning record."""
    from servepilot.cache.store import CacheStore

    state = get_state(ctx)
    record = CacheStore(state.settings.cache_dir).load(key)
    if json_output:
        emit_json(record.model_dump(mode="json"))
        return
    console = state.console
    console.print(
        f"[bold]Tuning record {record.key}[/] · ServePilot {record.servepilot_version} · {record.status} · created {record.created_at:%Y-%m-%d %H:%M} UTC"
    )
    render_hardware(console, record.hardware_snapshot)
    console.print()
    render_model(console, record.model_profile)
    console.print()
    render_workload(console, record.workload_profile)
    console.print(f"seed {record.seed} · engine versions {record.engine_versions}")
    console.print()
    render_results_table(console, record.candidates)
    if record.winner is not None:
        console.print()
        render_selected(console, record.winner, commands=record.winner.equivalent_commands)
        if record.winner.pareto_front:
            console.print()
            render_pareto(console, record.winner.pareto_front)


@cache_app.command("clear")
@handle_errors
def cache_clear(
    ctx: typer.Context,
    key: Annotated[
        str | None, typer.Argument(help="Only delete this record (default: all).")
    ] = None,
    json_output: JSONOpt = False,
) -> None:
    """Delete cached tuning records."""
    from servepilot.cache.store import CacheStore

    state = get_state(ctx)
    store = CacheStore(state.settings.cache_dir)
    if key:
        removed = 1 if store.delete(key) else 0
    else:
        removed = store.clear()
    if json_output:
        emit_json({"removed": removed})
    else:
        state.console.print(f"Removed {removed} tuning record(s).")


register_optimization(app, handle_errors)


def main() -> None:
    try:
        app()
    except (
        EngineUnavailableError
    ) as exc:  # pragma: no cover - defensive, commands handle their own errors
        Console(stderr=True).print(exc.render())
        sys.exit(int(exc.exit_code))


if __name__ == "__main__":  # pragma: no cover
    main()
