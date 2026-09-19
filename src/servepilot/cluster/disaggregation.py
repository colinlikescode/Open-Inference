"""Managed SGLang prefill/decode workers and a per-deployment model gateway."""

from __future__ import annotations

import asyncio
import shlex

from servepilot.cluster.container_launcher import ContainerLauncher, ProcessGroup
from servepilot.engines.base import LaunchSpec
from servepilot.engines.interpreter import EngineRuntime
from servepilot.engines.sglang import SGLangEngine
from servepilot.exceptions import ConfigurationError, LaunchError
from servepilot.schemas.plan import CandidatePlan, EngineName


def replace_flags(arguments: list[str], values: dict[str, str | None]) -> list[str]:
    result = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in values:
            index += 2
        else:
            result.append(argument)
            index += 1
    for flag, value in values.items():
        if value is not None:
            result.extend([flag, value])
    return result


async def launch_disaggregated(launcher: ContainerLauncher, spec: LaunchSpec) -> ProcessGroup:
    plan = launcher.plan
    assert plan and plan.prefill
    prefill = plan.prefill
    if plan.engine != EngineName.SGLANG or plan.replica_count != 1:
        raise ConfigurationError("prefill/decode launch requires SGLang and one decode group")
    stages: list[tuple[str, CandidatePlan]] = []
    for index, ids in enumerate(prefill.gpu_groups):
        stages.append(
            (
                "prefill",
                plan.with_updates(
                    id=f"{plan.id}-prefill-{index}",
                    prefill=None,
                    gpu_groups=[ids],
                    replica_count=1,
                    tensor_parallel_size=prefill.tensor_parallel_size,
                    pipeline_parallel_size=prefill.pipeline_parallel_size,
                    memory_fraction=prefill.memory_fraction,
                ),
            )
        )
    stages.append(("decode", plan.with_updates(prefill=None)))
    children: list[ProcessGroup] = []
    entries: list[tuple[str, str, int]] = []
    runtime = launcher.runtime
    engine = SGLangEngine(runtime=EngineRuntime("python3", runtime["engine_version"]), probe=False)
    router_nodes = []
    try:
        for role, stage in stages:
            ids = stage.gpu_groups[0]
            node = launcher.manager.inventory.node(launcher.hardware.gpu(ids[0]).node_id)
            router_nodes.append(node)
            port = await launcher._port(node)
            bootstrap = await launcher._port(node)
            stage.replica_nodes = [
                [launcher.hardware.gpu(index).node_id or "local" for index in ids]
            ]
            stage.distributed_backend = "native" if stage.spans_nodes else None
            arguments = replace_flags(
                spec.args,
                {
                    "--port": str(port),
                    "--host": "0.0.0.0",
                    "--tp-size": str(stage.tensor_parallel_size),
                    "--pp-size": str(stage.pipeline_parallel_size),
                    "--mem-fraction-static": str(stage.memory_fraction),
                    "--disaggregation-mode": role,
                    "--disaggregation-transfer-backend": prefill.transfer_backend,
                    "--disaggregation-bootstrap-port": str(bootstrap),
                },
            )
            stage_spec = spec.model_copy(
                update={
                    "args": arguments,
                    "gpu_ids": ids,
                    "host": "0.0.0.0",
                    "port": port,
                    "node_id": node.node_id,
                    "node_ip": node.network_address,
                    "replica_id": stage.id,
                    "redacted_display_command": shlex.join([spec.executable, *arguments]),
                }
            )
            child_launcher = ContainerLauncher(
                launcher.manager, launcher.hardware, model_volume=launcher.model_volume
            )
            child_launcher.configure(stage, runtime)
            child = await child_launcher.launch(stage_spec)
            assert isinstance(child, ProcessGroup)
            children.append(child)
            for node_id, names in child_launcher.container_names.items():
                launcher.container_names.setdefault(node_id, []).extend(names)
            entries.append((role, stage_spec.base_url, bootstrap))
        # Every required stage must be ready before a gateway is advertised to the verifier.
        readiness = await asyncio.gather(
            *(engine.wait_until_ready(child.spec, child, 1200) for child in children)
        )
        for (role, url, _), result in zip(entries, readiness, strict=True):
            if not result.ready:
                raise LaunchError(
                    f"{role} worker {url} failed readiness",
                    hints=[
                        result.failure.message if result.failure else "worker did not become ready"
                    ],
                )
        args = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--host",
            "0.0.0.0",
            "--port",
            str(spec.port),
        ]
        for role, url, bootstrap in entries:
            args += ["--" + role, url]
            if role == "prefill":
                args.append(str(bootstrap))
        router = await launcher._start(spec, router_nodes[-1], [], args, purpose="router")
        processes = [router, *(process for child in children for process in child.processes)]
        group = ProcessGroup(spec, processes)
        launcher._groups.append(group)
        runtime["disaggregation_endpoints"] = [
            {"role": role, "url": url, "bootstrap_port": bootstrap}
            for role, url, bootstrap in entries
        ]
        return group
    except BaseException:
        await launcher.manager.cleanup()
        raise
