"""Time-budgeted optimization and reproducible deployment commands."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from servepilot.cli.common import get_state, run_async
from servepilot.exceptions import ConfigurationError
from servepilot.optimization.workflow import OptimizeOptions, deploy_recipe, optimize
from servepilot.schemas.workload import Objective


def milliseconds(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s)?\s*", value)
    if not match:
        raise ConfigurationError(f"invalid duration {value!r}; use 300ms, 0.3s, or milliseconds")
    result = float(match[1]) * (1000 if match[2] == "s" else 1)
    if not math.isfinite(result) or result <= 0:
        raise ConfigurationError("latency limits must be finite and positive")
    return result


def register(app: typer.Typer, handle_errors: Callable[..., Any]) -> None:
    @app.command("optimize")
    @handle_errors
    def optimize_command(
        ctx: typer.Context,
        model: Annotated[
            str | None, typer.Option("--model", help="Hugging Face model or local model directory.")
        ] = None,
        nodes: Annotated[
            Path | None,
            typer.Option(
                "--nodes", help="Existing GPU machines and SSH access; omit for local GPUs."
            ),
        ] = None,
        minutes: Annotated[
            float | None,
            typer.Option(
                "--minutes",
                min=0.001,
                help="Total search budget including setup (default 60 minutes).",
            ),
        ] = None,
        hours: Annotated[float | None, typer.Option("--hours", min=0.001)] = None,
        output: Annotated[Path, typer.Option("--output")] = Path("servepilot-output"),
        resume: Annotated[
            Path | None,
            typer.Option(
                "--resume",
                help="Append experiments to an existing run without changing its contract.",
            ),
        ] = None,
        objective: Annotated[Objective | None, typer.Option("--objective")] = None,
        workload: Annotated[
            Path | None,
            typer.Option(
                "--workload", help="Traffic distribution YAML or representative requests JSONL."
            ),
        ] = None,
        correctness: Annotated[
            Path | None,
            typer.Option("--correctness", help="Fixed correctness cases and tolerances in YAML."),
        ] = None,
        max_p95_ttft: Annotated[str | None, typer.Option("--max-p95-ttft")] = None,
        max_p95_tpot: Annotated[str | None, typer.Option("--max-p95-tpot")] = None,
        max_p95_latency: Annotated[str | None, typer.Option("--max-p95-latency")] = None,
        min_throughput: Annotated[
            float | None,
            typer.Option("--min-throughput", min=0.001, help="Minimum output tokens per second."),
        ] = None,
        agent_model: Annotated[
            str | None, typer.Option("--agent-model", help="Model alias exposed by LiteLLM.")
        ] = None,
        litellm_url: Annotated[
            str | None,
            typer.Option(
                "--litellm-url",
                help="OpenAI-compatible reasoning endpoint; defaults to LITELLM_BASE_URL.",
            ),
        ] = None,
        agent_config: Annotated[
            Path | None,
            typer.Option(
                "--agent-config",
                help="Pi provider configuration YAML (environment variable names, no keys).",
            ),
        ] = None,
        engine: Annotated[str, typer.Option("--engine", help="auto | vllm | sglang.")] = "auto",
        runtime_config: Annotated[
            Path | None,
            typer.Option(
                "--runtime-config", help="Container image and shared-memory settings YAML."
            ),
        ] = None,
        no_bootstrap: Annotated[
            bool,
            typer.Option(
                "--no-bootstrap", help="Use existing Docker/NVIDIA Container Toolkit installations."
            ),
        ] = False,
        revision: Annotated[str | None, typer.Option("--revision")] = None,
        trust_remote_code: Annotated[bool, typer.Option("--trust-remote-code")] = False,
        concurrency: Annotated[
            str | None, typer.Option("--concurrency", help="Fixed load levels, e.g. 8,16,32,64.")
        ] = None,
        requests: Annotated[int, typer.Option("--requests-per-trial", min=2)] = 64,
        repetitions: Annotated[int, typer.Option("--repetitions", min=2)] = 2,
        plateau_minutes: Annotated[
            float | None, typer.Option("--plateau-minutes", min=0.001)
        ] = None,
        no_deploy: Annotated[
            bool,
            typer.Option("--no-deploy", help="Save verified artifacts and exit without serving."),
        ] = False,
        host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
        port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
    ) -> None:
        """Let Pi improve inference on your GPUs; verify each experiment and serve the best result."""
        if minutes is not None and hours is not None:
            raise ConfigurationError("choose --minutes or --hours")
        if not model and not resume:
            raise ConfigurationError("--model is required unless --resume is provided")
        if engine not in ("auto", "vllm", "sglang", "fake"):
            raise ConfigurationError("--engine must be auto, vllm, or sglang")
        try:
            levels = [int(value) for value in concurrency.split(",")] if concurrency else None
        except ValueError as exc:
            raise ConfigurationError("--concurrency must be comma-separated integers") from exc
        options = OptimizeOptions(
            model=model,
            revision=revision,
            nodes=nodes,
            output=output,
            resume=resume,
            seconds=hours * 3600
            if hours is not None
            else (minutes if minutes is not None else 60) * 60,
            objective=objective,
            workload=workload,
            correctness=correctness,
            agent_config=agent_config,
            agent_model=agent_model,
            litellm_url=litellm_url,
            engine=engine,
            runtime_config=runtime_config,
            bootstrap=not no_bootstrap,
            max_p95_ttft_ms=milliseconds(max_p95_ttft),
            max_p95_tpot_ms=milliseconds(max_p95_tpot),
            max_p95_latency_ms=milliseconds(max_p95_latency),
            minimum_throughput=min_throughput,
            requests=requests,
            repetitions=repetitions,
            concurrency=levels,
            plateau_seconds=plateau_minutes * 60 if plateau_minutes else None,
            trust_remote_code=trust_remote_code,
            deploy=not no_deploy,
            host=host,
            port=port,
        )
        state = get_state(ctx)
        outcome = run_async(optimize(options, state.settings, state.err_console.print))
        if outcome.best is None:
            raise typer.Exit(code=1)

    @app.command("deploy")
    @handle_errors
    def deploy_command(
        ctx: typer.Context,
        recipe: Annotated[
            Path, typer.Argument(help="Verified recipe.yaml from an optimization run.")
        ],
        nodes: Annotated[
            Path | None,
            typer.Option(
                "--nodes", help="Equivalent existing GPU machines (default: saved inventory)."
            ),
        ] = None,
        host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
        port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
        no_bootstrap: Annotated[bool, typer.Option("--no-bootstrap")] = False,
    ) -> None:
        """Deploy exact saved engine images and model revision, without repeating optimization."""
        state = get_state(ctx)
        run_async(
            deploy_recipe(
                recipe.resolve(),
                nodes=nodes,
                settings=state.settings,
                host=host,
                port=port,
                bootstrap=not no_bootstrap,
                progress=state.err_console.print,
            )
        )
