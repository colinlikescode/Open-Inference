"""One launch interface for local replicas, native multi-node inference, and optional Ray."""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import Sequence
from typing import Any

from servepilot.cluster.containers import ContainerManager
from servepilot.cluster.inventory import SSHNode
from servepilot.engines.base import LaunchSpec
from servepilot.engines.process import ProcessHandle
from servepilot.exceptions import ConfigurationError, LaunchError
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.plan import CandidatePlan, EngineName


class ProcessGroup:
    def __init__(self, spec: LaunchSpec, processes: Sequence[ProcessHandle]) -> None:
        self.spec = spec
        self.processes = list(processes)

    @property
    def pid(self) -> int | None:
        return self.processes[0].pid if self.processes else None

    @property
    def create_time(self) -> float | None:
        return self.processes[0].create_time if self.processes else None

    @property
    def returncode(self) -> int | None:
        return next((p.returncode for p in self.processes if p.returncode is not None), None)

    def is_running(self) -> bool:
        return bool(self.processes) and all(p.is_running() for p in self.processes)

    def stdout_tail(self) -> str:
        return "\n".join(f"[{p.spec.node_id}] {p.stdout_tail()}" for p in self.processes)[-256000:]

    def stderr_tail(self) -> str:
        return "\n".join(f"[{p.spec.node_id}] {p.stderr_tail()}" for p in self.processes)[-256000:]

    async def wait(self, timeout: float | None = None) -> int | None:
        try:
            async with asyncio.timeout(timeout):
                await asyncio.gather(*(process.wait() for process in self.processes))
            return self.returncode
        except TimeoutError:
            return None

    async def terminate(self, grace_seconds: float = 10) -> None:
        results = await asyncio.gather(
            *(p.terminate(grace_seconds) for p in self.processes), return_exceptions=True
        )
        if any(isinstance(result, BaseException) for result in results):
            raise LaunchError("a distributed runtime could not be stopped")


def native_command(
    spec: LaunchSpec, plan: CandidatePlan, *, rank: int, nodes: int, master: str, port: int
) -> list[str]:
    args = list(spec.args)
    if plan.engine == EngineName.VLLM:
        # `vllm serve` dispatches nonzero ranks to the headless executor. api_server alone does not.
        if args[:2] != ["-m", "vllm.entrypoints.openai.api_server"]:
            raise ConfigurationError("unexpected vLLM launch entrypoint")
        args = ["-m", "vllm.entrypoints.cli.main", "serve", *args[2:]]
        if plan.data_parallel_size > 1:
            per_rank = (
                1 if plan.dp_attention_enabled else plan.tensor_parallel_size
            ) * plan.pipeline_parallel_size
            if plan.data_parallel_size % nodes or len(spec.gpu_ids) // nodes % per_rank:
                raise ConfigurationError(
                    "native vLLM DP requires complete TP/PP groups per node and equal local DP counts; select Ray for a different placement"
                )
            local_dp = plan.data_parallel_size // nodes
            args += [
                "--data-parallel-size-local",
                str(local_dp),
                "--data-parallel-address",
                master,
                "--data-parallel-rpc-port",
                str(port),
            ]
            if rank > 0:
                args += ["--data-parallel-start-rank", str(rank * local_dp)]
        else:
            args += [
                "--nnodes",
                str(nodes),
                "--node-rank",
                str(rank),
                "--master-addr",
                master,
                "--master-port",
                str(port),
            ]
        if rank > 0:
            args.append("--headless")
    elif plan.engine == EngineName.SGLANG:
        args += [
            "--nnodes",
            str(nodes),
            "--node-rank",
            str(rank),
            "--dist-init-addr",
            f"{master}:{port}",
        ]
    else:
        raise ConfigurationError(f"native multi-node execution is unavailable for {plan.engine}")
    return ["python3", *args]


class ContainerLauncher:
    def __init__(
        self, manager: ContainerManager, hardware: HardwareSnapshot, *, model_volume: str
    ) -> None:
        self.manager = manager
        self.hardware = hardware
        self.model_volume = model_volume
        self.plan: CandidatePlan | None = None
        self.runtime: dict[str, Any] = {}
        self._groups: list[ProcessGroup] = []
        self.container_names: dict[str, list[str]] = {}

    def configure(self, plan: CandidatePlan, runtime: dict[str, Any]) -> None:
        if self.tracked():
            raise LaunchError("cannot replace the runtime while inference is still running")
        self.plan = plan
        self.runtime = runtime
        self.container_names = {}

    async def _port(self, node: SSHNode) -> int:
        result = await self.manager.transport.run(
            node,
            [
                node.python,
                "-c",
                "import socket; s=socket.socket(); s.bind(('0.0.0.0',0)); print(s.getsockname()[1]); s.close()",
            ],
        )
        return int(result.stdout.strip())

    def docker_run(
        self,
        spec: LaunchSpec,
        node: SSHNode,
        gpu_ids: list[int],
        name: str,
        command: list[str],
        *,
        extra_env: dict[str, str] | None = None,
    ) -> LaunchSpec:
        image = self.runtime["images"][node.node_id]
        uuids = [self.hardware.gpu(index).uuid for index in gpu_ids]
        env = {
            key: value
            for key, value in spec.env.items()
            if key.startswith(("NCCL_", "GLOO_", "VLLM_", "SGLANG_"))
            and not any(word in key for word in ("TOKEN", "SECRET", "API_KEY"))
        }
        env.update(self.runtime.get("environment", {}))
        env.update(
            CUDA_VISIBLE_DEVICES=",".join(uuids),
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            HF_HOME="/model-cache",
            HF_MODULES_CACHE="/tmp/openbaseten-hf-modules",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            VLLM_HOST_IP=node.network_address,
            PYTHONUNBUFFERED="1",
        )
        env.update(extra_env or {})
        args = [
            "run",
            "--rm",
            "--name",
            name,
            "--label",
            f"openbaseten.run={self.manager.namespace}",
            "--network",
            "host",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--shm-size",
            self.manager.config.shm_size,
            "--ulimit",
            "memlock=-1",
            "--mount",
            f"type=volume,source={self.model_volume},target=/model-cache,readonly",
        ]
        if uuids:
            args += ["--gpus", '"device=' + ",".join(uuids) + '"']
            if self.manager.details.get("nodes", {}).get(node.node_id, {}).get("infiniband"):
                args += ["--device", "/dev/infiniband"]
        for key, value in sorted(env.items()):
            args += ["--env", f"{key}={value}"]
        args += ["--entrypoint", command[0], image, *command[1:]]
        argv = self.manager.docker(node, *args)
        return spec.model_copy(
            update={
                "executable": argv[0],
                "args": argv[1:],
                "env": {},
                "node_id": node.node_id,
                "node_ip": node.network_address,
                "gpu_ids": gpu_ids,
                "redacted_display_command": shlex.join(argv),
            }
        )

    async def _start(
        self,
        spec: LaunchSpec,
        node: SSHNode,
        gpu_ids: list[int],
        command: list[str],
        *,
        purpose: str = "engine",
        env: dict[str, str] | None = None,
    ) -> ProcessHandle:
        name = self.manager.container_name(purpose)
        if (
            purpose == "engine"
            and self.runtime.get("profile_enabled")
            and self.plan
            and self.plan.engine == EngineName.VLLM
        ):
            command = [
                *command,
                "--profiler-config",
                json.dumps(
                    {
                        "profiler": "torch",
                        "torch_profiler_dir": "/tmp/openbaseten-profile",
                        "torch_profiler_with_stack": False,
                    }
                ),
            ]
        container = self.docker_run(spec, node, gpu_ids, name, command, extra_env=env)
        self.container_names.setdefault(node.node_id, []).append(name)
        return await self.manager.start_guarded(container, node, name)

    async def _ray(
        self,
        spec: LaunchSpec,
        nodes: list[SSHNode],
        by_node: dict[str, list[int]],
    ) -> tuple[list[ProcessHandle], str]:
        master = nodes[0]
        port = await self._port(master)
        address = f"{master.network_address}:{port}"
        processes = []
        for rank, node in enumerate(nodes):
            args = [
                "ray",
                "start",
                "--node-ip-address",
                node.network_address,
                "--num-gpus",
                str(len(by_node[node.node_id])),
                "--disable-usage-stats",
                "--block",
            ]
            if rank == 0:
                args += ["--head", "--port", str(port), "--include-dashboard=false"]
            else:
                args += ["--address", address]
            process = await self._start(spec, node, by_node[node.node_id], args, purpose="ray")
            processes.append(process)
            # Verify GCS is listening before asking a worker to join it.
            if rank == 0:
                for _ in range(120):
                    if not process.is_running():
                        raise LaunchError(f"Ray head exited: {process.stderr_tail()}")
                    result = await self.manager.transport.run(
                        master,
                        [
                            master.python,
                            "-c",
                            "import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=1); s.close()",
                            master.network_address,
                            str(port),
                        ],
                        timeout=3,
                        check=False,
                    )
                    if result.returncode == 0:
                        break
                    await asyncio.sleep(0.5)
                else:
                    raise LaunchError("managed Ray head did not start")
        # Exact world size: do not launch inference while a worker is still joining.
        expected = sum(len(ids) for ids in by_node.values())
        head_name = self.container_names[master.node_id][-1]
        code = "import ray,json,sys; ray.init(address=sys.argv[1],logging_level='ERROR'); print(json.dumps({'gpus':int(ray.cluster_resources().get('GPU',0))}))"
        for _ in range(120):
            result = await self.manager.command(
                master, ["exec", head_name, "python3", "-c", code, address], timeout=15, check=False
            )
            try:
                if (
                    result.returncode == 0
                    and json.loads(result.stdout.strip().splitlines()[-1])["gpus"] == expected
                ):
                    return processes, address
            except (ValueError, KeyError, IndexError):
                pass
            if any(not process.is_running() for process in processes):
                raise LaunchError("a managed Ray worker exited before joining the cluster")
            await asyncio.sleep(0.5)
        raise LaunchError("managed Ray cluster did not register all expected GPUs")

    async def launch(self, spec: LaunchSpec) -> ProcessHandle:
        if self.plan is None:
            raise LaunchError("container launcher has no experiment configuration")
        if self.plan.prefill is not None:
            from servepilot.cluster.disaggregation import launch_disaggregated

            return await launch_disaggregated(self, spec)
        by_node: dict[str, list[int]] = {}
        for index in spec.gpu_ids:
            gpu = self.hardware.gpu(index)
            if gpu.node_id is None:
                raise LaunchError("container GPU inventory is missing node identity")
            by_node.setdefault(gpu.node_id, []).append(index)
        nodes = [self.manager.inventory.node(node_id) for node_id in by_node]
        processes = []
        try:
            if len(nodes) == 1:
                processes.append(
                    await self._start(spec, nodes[0], spec.gpu_ids, ["python3", *spec.args])
                )
            elif self.plan.distributed_backend == "ray":
                workers, address = await self._ray(spec, nodes, by_node)
                processes.extend(workers)
                driver = await self._start(
                    spec,
                    nodes[0],
                    by_node[nodes[0].node_id],
                    [
                        "python3",
                        *spec.args,
                        *(
                            ["--data-parallel-backend", "ray"]
                            if self.plan.data_parallel_size > 1
                            and self.plan.engine == EngineName.VLLM
                            else []
                        ),
                    ],
                    env={"RAY_ADDRESS": address},
                )
                processes.insert(0, driver)
            else:
                if self.plan.distributed_backend not in ("mp", "native"):
                    raise ConfigurationError("multi-node inference requires mp, native, or ray")
                if len({len(ids) for ids in by_node.values()}) != 1:
                    raise ConfigurationError(
                        "native distributed replicas require equal GPU counts per participating node"
                    )
                port = await self._port(nodes[0])
                # Start every node before checking HTTP readiness on rank zero.
                for rank, node in enumerate(nodes):
                    command = native_command(
                        spec,
                        self.plan,
                        rank=rank,
                        nodes=len(nodes),
                        master=nodes[0].network_address,
                        port=port,
                    )
                    processes.append(await self._start(spec, node, by_node[node.node_id], command))
            group = ProcessGroup(spec, processes)
            self._groups.append(group)
            return group
        except BaseException:
            await self.manager.cleanup()
            raise

    async def shutdown_all(self, grace_seconds: float = 10) -> None:
        await self.manager.cleanup()
        self._groups = []

    def tracked(self) -> list[ProcessHandle]:
        return [group for group in self._groups if any(p.is_running() for p in group.processes)]

    def supports_node(self, node_id: str | None) -> bool:
        try:
            self.manager.inventory.node(node_id)
            return True
        except ConfigurationError:
            return False
