"""CLI-independent orchestration from SSH inspection through a verified deployment."""

from __future__ import annotations

import asyncio
import html
import json
import os
import signal
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from servepilot.cluster.container_launcher import ContainerLauncher
from servepilot.cluster.containers import ContainerConfig, ContainerManager
from servepilot.cluster.inventory import NodeInventory
from servepilot.cluster.model_cache import ModelCache, import_image, local_manifest
from servepilot.cluster.network import check_nccl, check_network
from servepilot.cluster.ssh_provider import SSHHardwareProvider
from servepilot.cluster.transport import SSHTransport
from servepilot.engines.process import LocalLauncher
from servepilot.engines.registry import build_registry
from servepilot.exceptions import ConfigurationError
from servepilot.hardware.base import HardwareProvider, get_hardware_provider
from servepilot.logging import redact_secrets
from servepilot.models.auth import hf_token
from servepilot.models.inspector import ModelInspector
from servepilot.models.tokenizer import load_tokenizer
from servepilot.optimization.agent import PiAgent
from servepilot.optimization.backend import FakeLaunchBackend, LaunchBackend
from servepilot.optimization.budget import BudgetExpired, TimeBudget
from servepilot.optimization.container_backend import ContainerBackend, container_registry
from servepilot.optimization.controller_state import ControllerState
from servepilot.optimization.correctness import CorrectnessVerifier, default_suite
from servepilot.optimization.loop import OptimizationOutcome, Optimizer
from servepilot.optimization.report import file_sha256, load_recipe, write_recipe, write_report
from servepilot.optimization.schemas import (
    AgentConfig,
    Contract,
    CorrectnessSuite,
    EvaluationPolicy,
    ExperimentProposal,
    ExperimentResult,
    RunDefinition,
    canonical_json,
    fingerprint,
)
from servepilot.optimization.store import ExperimentStore, atomic_write
from servepilot.optimization.workload import ReplayEntry, load_traffic
from servepilot.planner.planner import Planner
from servepilot.runtime.ports import PortAllocator
from servepilot.runtime.state import RuntimeStateStore
from servepilot.runtime.supervisor import Deployment
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import PlanConstraints, SelectedPlan
from servepilot.schemas.workload import LatencyConstraints, Objective, WorkloadProfile
from servepilot.settings import ServePilotSettings


class OptimizeOptions(Contract):
    model: str | None = None
    revision: str | None = None
    nodes: Path | None = None
    output: Path = Path("servepilot-output")
    resume: Path | None = None
    seconds: float = Field(default=3600, gt=0)
    objective: Objective | None = None
    workload: Path | None = None
    correctness: Path | None = None
    agent_config: Path | None = None
    agent_model: str | None = None
    litellm_url: str | None = None
    engine: str = "auto"
    runtime_config: Path | None = None
    bootstrap: bool = True
    max_p95_ttft_ms: float | None = Field(default=None, gt=0)
    max_p95_tpot_ms: float | None = Field(default=None, gt=0)
    max_p95_latency_ms: float | None = Field(default=None, gt=0)
    minimum_throughput: float | None = Field(default=None, gt=0)
    requests: int = Field(default=64, ge=2)
    repetitions: int = Field(default=2, ge=2)
    concurrency: list[int] | None = None
    plateau_seconds: float | None = Field(default=None, gt=0)
    trust_remote_code: bool = False
    deploy: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


def compatibility(snapshot: HardwareSnapshot, *, identities: bool) -> dict[str, Any]:
    return {
        "gpus": [
            {
                "index": g.index,
                "node": g.node_id
                if identities
                else next((i for i, n in enumerate(snapshot.nodes) if n.node_id == g.node_id), 0),
                "local": g.device_index_on_node,
                "name": g.name,
                "memory": g.total_memory_bytes,
                "cc": [g.compute_capability_major, g.compute_capability_minor],
                **({"uuid": g.uuid, "address": g.node_ip} if identities else {}),
            }
            for g in snapshot.gpus
        ],
        "topology": snapshot.topology.model_dump(mode="json"),
        "driver": snapshot.driver_version,
        "cuda": snapshot.cuda_version,
    }


def require_available(settings: ServePilotSettings) -> None:
    state = RuntimeStateStore(settings.state_dir)
    saved = state.read()
    if saved and state.is_live(saved):
        raise ConfigurationError("a deployment is already running; use servepilot stop first")
    controller = ControllerState(settings.state_dir).read()
    if controller and controller["running"]:
        raise ConfigurationError(
            "an optimization controller is already running; inspect servepilot status"
        )


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc


def agent_settings(options: OptimizeOptions) -> AgentConfig:
    data = dict(load_yaml(options.agent_config) or {}) if options.agent_config else {}
    data.setdefault("model", os.environ.get("OPENBASETEN_AGENT_MODEL", "gemini-3.8-flash"))
    data.setdefault("base_url", os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000/v1"))
    if options.agent_model:
        data["model"] = options.agent_model
    if options.litellm_url:
        data["base_url"] = options.litellm_url
    return AgentConfig.model_validate(data)


async def export_winner(
    manager: ContainerManager, store: ExperimentStore, best: ExperimentResult
) -> ExperimentResult:
    # Export a copy: original experiment evidence is immutable.
    result = best.model_copy(deep=True)
    archives, saved = {}, {}
    for node in manager.inventory.nodes:
        image = result.runtime["images"][node.node_id]
        inspected = await manager.command(node, ["image", "inspect", image, "--format", "{{.Id}}"])
        image_id = inspected.stdout.strip()
        result.runtime["images"][node.node_id] = image_id
        if image_id not in saved:
            path = f"images/{image_id.replace(':', '-')}.tar"
            digest = await manager.export_image(node, image_id, store.directory / path)
            saved[image_id] = path
            result.artifacts[path] = digest
        archives[node.node_id] = saved[image_id]
    result.runtime["image_archives"] = archives
    return result


async def serve_verified(
    *,
    definition: RunDefinition,
    best: ExperimentResult,
    backend: LaunchBackend,
    settings: ServePilotSettings,
    host: str,
    port: int,
    progress: Callable[[str], None],
    on_ready: Callable[[str], None],
) -> None:
    plan = best.proposal.plan.with_updates(max_concurrency=best.decision.recommended_concurrency)
    if isinstance(backend, ContainerBackend):
        backend.container_launcher.configure(plan, best.runtime)
        backend.registry = container_registry({plan.engine.value: best.runtime["engine_version"]})
    selected = SelectedPlan(
        plan=plan,
        objective=definition.workload.objective,
        benchmarked=True,
        slo_satisfied=True,
        final_result=best.benchmarks[0],
        engine_version=best.runtime.get("engine_version"),
        equivalent_commands=best.runtime.get("commands", []),
        rationale=["Best independently verified experiment within the allocated budget."],
    )
    deployment = Deployment(
        selected=selected,
        model=backend.runtime_model,
        engine=backend.registry.require(plan.engine),
        launcher=backend.launcher,
        ports=backend.ports,
        hardware=definition.hardware,
        hardware_provider=backend.hardware,
        state_store=RuntimeStateStore(settings.state_dir),
        host=host,
        port=port,
        served_model_name=definition.model.model_id,
        trust_remote_code=definition.trust_remote_code,
        startup_timeout=1200,
        stagger_seconds=0,
    )
    deployment.install_signal_handlers()
    try:
        progress("Starting a clean deployment of the verified winner")
        await deployment.start()
        assert best.correctness is not None
        reference = {
            o.name: o.actual
            for o in best.correctness.observations
            if not o.streaming and o.actual is not None
        }
        verified = await CorrectnessVerifier(definition.correctness).run(
            deployment.base_url, definition.model.model_id, reference=reference
        )
        if not verified.passed:
            raise ConfigurationError(
                "clean deployment failed correctness verification; no endpoint accepted"
            )
        endpoint = deployment.base_url + "/v1"
        on_ready(endpoint)
        progress(f"Ready: {endpoint}")
        await deployment._stop_event.wait()
    finally:
        await deployment.shutdown()


async def optimize(
    options: OptimizeOptions, settings: ServePilotSettings, progress: Callable[[str], None]
) -> OptimizationOutcome:
    require_available(settings)
    budget = TimeBudget(options.seconds)
    output = (options.resume or options.output).resolve()
    controller = ControllerState(settings.state_dir)
    manager: ContainerManager | None = None
    backend: LaunchBackend | None = None
    stopped = asyncio.Event()
    optimizer: Optimizer | None = None
    task = asyncio.current_task()
    preparation: dict[str, Any] = {}
    owns_output = False

    def stop() -> None:
        stopped.set()
        if optimizer is not None:
            optimizer.request_stop()
        elif task:
            task.cancel()

    def report(message: str) -> None:
        controller.write(phase="optimizing", output=output, message=message)
        progress(message)

    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop)
    try:
        with ExperimentStore(output) as store:
            report("Inspecting model and existing machines")
            existing = (
                store.definition() if options.resume and (output / "run.json").is_file() else None
            )
            if options.resume and (output / ".setup.json").is_file():
                preparation = json.loads((output / ".setup.json").read_text())
                if not existing:
                    requested = options
                    options = OptimizeOptions.model_validate(preparation["options"]).model_copy(
                        update={
                            "resume": requested.resume,
                            "seconds": requested.seconds,
                            "deploy": requested.deploy,
                            "host": requested.host,
                            "port": requested.port,
                            "bootstrap": requested.bootstrap,
                        }
                    )
                    for source, digest in preparation["source_files"].items():
                        if file_sha256(Path(source)) != digest:
                            raise ConfigurationError(
                                f"setup input changed since this run: {source}"
                            )
            elif options.resume and not existing:
                raise ConfigurationError(
                    "resume directory has neither a run definition nor a setup checkpoint"
                )
            if (
                not existing
                and not preparation
                and any(path.name != ".controller.lock" for path in output.iterdir())
            ):
                raise ConfigurationError(
                    "output directory is not empty; use --resume or another --output"
                )
            if not existing and not preparation:
                options = options.model_copy(deep=True)
                sources = {}
                for name in ("nodes", "workload", "correctness", "agent_config", "runtime_config"):
                    source = getattr(options, name)
                    if source is not None:
                        source = source.resolve()
                        setattr(options, name, source)
                        sources[str(source)] = file_sha256(source)
                if options.model:
                    model_source = Path(options.model)
                    if await asyncio.to_thread(model_source.is_dir):
                        options.model = str(await asyncio.to_thread(model_source.resolve))
                preparation = {
                    "options": options.model_dump(mode="json"),
                    "source_files": sources,
                    "namespace": "ob-" + uuid.uuid4().hex[:24],
                    "agent": agent_settings(options).model_dump(mode="json"),
                }
                atomic_write(output / ".setup.json", canonical_json(preparation).encode())
            owns_output = True
            if existing and any(
                (
                    options.model,
                    options.workload,
                    options.correctness,
                    options.objective,
                    options.revision,
                    options.max_p95_ttft_ms,
                    options.max_p95_tpot_ms,
                    options.max_p95_latency_ms,
                    options.minimum_throughput,
                    options.concurrency,
                )
            ):
                raise ConfigurationError(
                    "resume preserves the original model, objective, workload, and verification policy"
                )
            agent = PiAgent(
                existing.agent if existing else AgentConfig.model_validate(preparation["agent"])
            )
            agent.preflight()
            fake = "fake" in existing.engine_versions if existing else options.engine == "fake"
            if fake and not settings.enable_fake_engine:
                raise ConfigurationError(
                    "fake inference requires SERVEPILOT_ENABLE_FAKE_ENGINE=1 and explicit fake hardware"
                )
            inventory = (
                NodeInventory.load(options.nodes)
                if options.nodes or not existing
                else NodeInventory.model_validate(existing.nodes)
            )
            provider: HardwareProvider = (
                get_hardware_provider(settings) if fake else SSHHardwareProvider(inventory)
            )
            if not fake and options.bootstrap:
                report("Bootstrapping supplied GPU workers")
                transport = SSHTransport(inventory)
                script = (Path(__file__).parents[1] / "cluster" / "bootstrap.sh").read_bytes()

                async def bootstrap_workers() -> None:
                    await asyncio.gather(
                        *(
                            transport.run(node, ["sh", "-s"], input_data=script, timeout=1200)
                            for node in inventory.nodes
                        )
                    )

                await budget.run(bootstrap_workers)
            hardware = await budget.run(lambda: asyncio.to_thread(provider.snapshot))
            if fake and hardware.provider != "fake":
                raise ConfigurationError("fake optimization requires SERVEPILOT_FAKE_HARDWARE")
            if existing and compatibility(hardware, identities=True) != compatibility(
                existing.hardware, identities=True
            ):
                raise ConfigurationError(
                    "cluster hardware, topology, or drivers changed since this run; start a new run"
                )
            model = (
                existing.model
                if existing
                else ModelProfile.model_validate(preparation["model"])
                if preparation.get("model")
                else await budget.run(
                    lambda: asyncio.to_thread(
                        ModelInspector(token=hf_token()).inspect,
                        options.model or "",
                        options.revision,
                    )
                )
            )
            if not options.model and not existing:
                raise ConfigurationError("--model is required unless using --resume")
            manifest = None
            if model.local_path and not fake:
                model_directory = Path(model.local_path)
                manifest = await budget.run(
                    lambda: asyncio.to_thread(local_manifest, model_directory)
                )
                revision = fingerprint(manifest)
                if (existing or preparation.get("model")) and model.revision not in (
                    None,
                    revision,
                ):
                    raise ConfigurationError("local model contents changed since the original run")
                model = model.model_copy(update={"revision": revision})
            if not existing:
                preparation["model"] = model.model_dump(mode="json")
                atomic_write(output / ".setup.json", canonical_json(preparation).encode())
            tokenizer = await budget.run(
                lambda: asyncio.to_thread(
                    load_tokenizer,
                    model,
                    token=hf_token(),
                    trust_remote_code=existing.trust_remote_code
                    if existing
                    else options.trust_remote_code,
                )
            )
            if not fake and not tokenizer.exact:
                raise ConfigurationError(
                    "an exact controller-side tokenizer is required for verified token throughput; install the hf extra or supply tokenizer.json"
                )
            model_cache = None
            network: dict[str, Any] = {}
            if fake:
                registry = build_registry(settings)
                versions = {"fake": registry.require("fake").version() or "testing"}
            else:
                runtime = (
                    ContainerConfig.model_validate(existing.runtime_config)
                    if existing
                    else (
                        ContainerConfig.model_validate(load_yaml(options.runtime_config) or {})
                        if options.runtime_config
                        else ContainerConfig()
                    )
                )
                runtime.bootstrap = False  # Already bootstrapped before Python/GPU discovery.
                if existing:
                    runtime.images = existing.engine_images
                else:
                    runtime.images.update(preparation.get("engine_images", {}))
                engines = (
                    list(existing.engine_versions)
                    if existing
                    else (["vllm", "sglang"] if options.engine == "auto" else [options.engine])
                )
                setups = [
                    event["payload"]
                    for event in store.events()
                    if event["kind"] == "setup_complete"
                ]
                namespace = (
                    setups[-1]["namespace"] if existing and setups else preparation["namespace"]
                )
                manager = ContainerManager(inventory, runtime, namespace=namespace, budget=budget)
                report("Preparing pinned engine images and checking CUDA compatibility")
                await budget.run(lambda: manager.prepare(engines))
                await manager.cleanup_previous_run()
                report("Checking controller connectivity and inter-node bandwidth")
                network = await budget.run(lambda: check_network(manager, hardware))
                model_cache = ModelCache(manager, model)
                network["nccl"] = await budget.run(
                    lambda: check_nccl(manager, hardware, model_cache.volume)
                )
                manager.network = network
                report("Staging the pinned model snapshot on GPU workers")
                await budget.run(lambda: model_cache.prepare(token=hf_token(), manifest=manifest))
                versions = dict(manager.versions)
                if existing and versions != existing.engine_versions:
                    raise ConfigurationError(
                        "prepared engine versions differ from the original run"
                    )
                registry = container_registry(versions)
            if existing:
                definition = existing
            else:
                slo = LatencyConstraints(
                    max_p95_ttft_ms=options.max_p95_ttft_ms,
                    max_p95_tpot_ms=options.max_p95_tpot_ms,
                    max_p95_latency_ms=options.max_p95_latency_ms,
                )
                workload = WorkloadProfile(
                    objective=options.objective or Objective.THROUGHPUT, latency_constraints=slo
                )
                levels = [8, 16, 32, 64]
                entries: list[ReplayEntry] = []
                if options.workload:
                    workload, levels, entries = load_traffic(options.workload, workload, tokenizer)
                suite = (
                    CorrectnessSuite.model_validate(load_yaml(options.correctness))
                    if options.correctness
                    else default_suite()
                )
                definition = RunDefinition(
                    model=model,
                    hardware=hardware,
                    workload=workload,
                    policy=EvaluationPolicy(
                        requests_per_trial=options.requests,
                        repetitions=options.repetitions,
                        concurrency_levels=options.concurrency or levels,
                        minimum_output_tokens_per_second=options.minimum_throughput,
                    ),
                    correctness=suite,
                    agent=agent.config,
                    engine_images=manager.images if manager else {},
                    engine_versions=versions,
                    runtime_config=manager.config.model_dump(mode="json") if manager else {},
                    nodes=inventory.model_dump(mode="json"),
                    requests=[entry.model_dump(mode="json") for entry in entries],
                    trust_remote_code=options.trust_remote_code,
                )
                store.create(definition)
            store.append(
                "setup_complete",
                {
                    "seconds": budget.elapsed,
                    "network": network,
                    "namespace": manager.namespace if manager else None,
                },
            )
            if manifest:
                path = output / "model-manifest.json"
                if not path.exists():
                    store.artifact("model-manifest.json", canonical_json(manifest).encode())
            launcher = (
                ContainerLauncher(manager, hardware, model_volume=model_cache.volume)
                if manager and model_cache
                else LocalLauncher()
            )
            kwargs: dict[str, Any] = {
                "definition": definition,
                "store": store,
                "registry": registry,
                "launcher": launcher,
                "hardware": provider,
                "tokenizer": tokenizer,
                "ports": PortAllocator(),
                "startup_timeout": 1200,
            }
            backend = (
                ContainerBackend(manager=manager, model_cache=model_cache, **kwargs)
                if manager and model_cache
                else FakeLaunchBackend(**kwargs)
            )
            planning = Planner(
                engines=[registry.require(engine) for engine in versions],
                constraints=PlanConstraints(trust_remote_code=definition.trust_remote_code),
            ).plan(hardware, model, definition.workload)
            baselines = [
                ExperimentProposal(
                    hypothesis="Establish a verified baseline: " + plan.label(), plan=plan
                )
                for plan in planning.viable
            ]
            if not baselines:
                raise ConfigurationError(
                    "no compatible baseline plans fit the supplied GPUs",
                    hints=[item.reason for item in planning.excluded][:10],
                )
            optimizer = Optimizer(
                definition=definition,
                store=store,
                budget=budget,
                backend=backend,
                agent=agent,
                baselines=baselines,
                plateau_seconds=options.plateau_seconds,
                progress=report,
            )
            try:
                outcome = await optimizer.run()
            except BaseException:
                results = store.results()
                accepted = [r for r in results if r.decision.accepted]
                write_report(
                    store,
                    definition,
                    OptimizationOutcome(
                        accepted[-1] if accepted else None,
                        results[0] if results else None,
                        "controller_error",
                        budget.elapsed,
                        len(results),
                    ),
                )
                raise
            write_report(store, definition, outcome)
            if outcome.best is None:
                progress(
                    f"No configuration passed all verification gates. Evidence: {output / 'report.html'}"
                )
                return outcome
            if manager:
                manager.budget = None
            export_start = time.monotonic()
            report("Saving the exact verified runtime and deployment recipe")
            best = await export_winner(manager, store, outcome.best) if manager else outcome.best
            recipe = write_recipe(store, definition, best)
            store.append(
                "artifacts_saved",
                {"seconds": time.monotonic() - export_start, "recipe": str(recipe)},
            )
            progress(f"Verified recipe: {recipe}")
            if options.deploy and not stopped.is_set():
                deployment_start = time.monotonic()

                def ready(endpoint: str) -> None:
                    store.append(
                        "deployment_ready",
                        {"endpoint": endpoint, "seconds": time.monotonic() - deployment_start},
                    )
                    write_report(store, definition, outcome, endpoint=endpoint)
                    controller.clear()

                try:
                    await serve_verified(
                        definition=definition,
                        best=best,
                        backend=backend,
                        settings=settings,
                        host=options.host,
                        port=options.port,
                        progress=progress,
                        on_ready=ready,
                    )
                except Exception as exc:
                    store.append(
                        "deployment_failed",
                        {"error": str(exc), "seconds": time.monotonic() - deployment_start},
                    )
                    write_report(store, definition, outcome)
                    raise
                finally:
                    store.append(
                        "deployment_stopped",
                        {"seconds_since_start": time.monotonic() - deployment_start},
                    )
                    write_report(store, definition, outcome)
            return outcome
    except BaseException as exc:
        if owns_output and not (output / "run.json").is_file():
            if manager:
                preparation["engine_images"] = manager.images
                atomic_write(output / ".setup.json", canonical_json(preparation).encode())
            reason = (
                "setup_budget_expired"
                if isinstance(exc, BudgetExpired)
                else "manual_stop"
                if stopped.is_set()
                else "setup_failed"
            )
            detail = redact_secrets(str(exc))
            atomic_write(
                output / "setup-error.json",
                canonical_json(
                    {"reason": reason, "error": detail, "elapsed_seconds": budget.elapsed}
                ).encode(),
            )
            atomic_write(
                output / "report.html",
                (
                    "<!doctype html><meta charset='utf-8'><title>Open BaseTen setup</title><h1>No verified configuration</h1><p>"
                    + html.escape(reason + ": " + detail)
                    + "</p><p>Setup progress is saved. Continue with <code>servepilot optimize --resume "
                    + html.escape(str(output))
                    + " --minutes 60</code>.</p>"
                ).encode(),
            )
            progress(f"Setup stopped; progress and report saved in {output}")
            if isinstance(exc, BudgetExpired) or stopped.is_set():
                return OptimizationOutcome(None, None, reason, budget.elapsed, 0)
        raise
    finally:
        if backend:
            await backend.close()
        elif manager:
            await manager.cleanup()
        controller.clear()
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_running_loop().remove_signal_handler(sig)


async def deploy_recipe(
    path: Path,
    *,
    nodes: Path | None,
    settings: ServePilotSettings,
    host: str,
    port: int,
    bootstrap: bool,
    progress: Callable[[str], None],
) -> None:
    require_available(settings)
    recipe = await asyncio.to_thread(load_recipe, path)
    definition = recipe.definition.model_copy(deep=True)
    best = recipe.result.model_copy(deep=True)
    fake = "fake" in definition.engine_versions
    if fake and not settings.enable_fake_engine:
        raise ConfigurationError(
            "this is a CPU simulation recipe; explicitly enable the fake testing engine"
        )
    inventory = (
        NodeInventory.load(nodes) if nodes else NodeInventory.model_validate(definition.nodes)
    )
    provider: HardwareProvider = (
        get_hardware_provider(settings) if fake else SSHHardwareProvider(inventory)
    )
    hardware = await asyncio.to_thread(provider.snapshot)
    if compatibility(hardware, identities=False) != compatibility(
        definition.hardware, identities=False
    ):
        raise ConfigurationError(
            "recipe requires equivalent GPU layout, topology, memory, architecture, CUDA, and drivers"
        )
    manager = None
    backend: LaunchBackend | None = None
    try:
        tokenizer = await asyncio.to_thread(
            load_tokenizer,
            definition.model,
            token=hf_token(),
            trust_remote_code=definition.trust_remote_code,
        )
        if fake:
            backend = FakeLaunchBackend(
                definition=definition,
                store=ExperimentStore(path.parent),
                registry=build_registry(settings),
                launcher=LocalLauncher(),
                hardware=provider,
                tokenizer=tokenizer,
                ports=PortAllocator(),
            )
        else:
            deployment_config = ContainerConfig.model_validate(definition.runtime_config)
            deployment_config.images = definition.engine_images
            deployment_config.bootstrap = bootstrap
            manager = ContainerManager(
                inventory,
                deployment_config,
                namespace="ob-deploy-" + uuid.uuid4().hex[:20],
            )
            await manager.prepare([best.proposal.plan.engine.value])
            old_nodes = NodeInventory.model_validate(definition.nodes).nodes
            mapping = {
                old.node_id: new.node_id
                for old, new in zip(old_nodes, inventory.nodes, strict=True)
            }
            images = {}
            for old, new in zip(old_nodes, inventory.nodes, strict=True):
                archive = best.runtime.get("image_archives", {}).get(old.node_id)
                if not archive or archive not in recipe.files:
                    raise ConfigurationError(
                        "recipe is missing a checksummed exact runtime archive"
                    )
                expected = best.runtime["images"][old.node_id]
                await import_image(manager, new, path.parent / archive, expected)
                images[new.node_id] = expected
            best.runtime["images"] = images
            if best.proposal.plan.replica_nodes:
                best.proposal.plan.replica_nodes = [
                    [mapping[node] for node in group] for group in best.proposal.plan.replica_nodes
                ]
            definition.hardware = hardware
            model_cache = ModelCache(manager, definition.model)
            manifest = (
                await asyncio.to_thread(local_manifest, Path(definition.model.local_path))
                if definition.model.local_path
                else None
            )
            await model_cache.prepare(token=hf_token(), manifest=manifest)
            await check_network(manager, hardware)
            await check_nccl(manager, hardware, model_cache.volume)
            backend = ContainerBackend(
                manager=manager,
                model_cache=model_cache,
                definition=definition,
                store=ExperimentStore(path.parent),
                registry=container_registry(
                    {best.proposal.plan.engine.value: best.runtime["engine_version"]}
                ),
                launcher=ContainerLauncher(manager, hardware, model_volume=model_cache.volume),
                hardware=provider,
                tokenizer=tokenizer,
                ports=PortAllocator(),
            )
        await serve_verified(
            definition=definition,
            best=best,
            backend=backend,
            settings=settings,
            host=host,
            port=port,
            progress=progress,
            on_ready=lambda endpoint: None,
        )
    finally:
        if backend:
            await backend.close()
        elif manager:
            await manager.cleanup()
