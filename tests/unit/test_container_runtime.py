"""Exercise real launch adapters and TCP control paths without Docker or GPU hardware."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from servepilot.cluster.container_launcher import ContainerLauncher, native_command
from servepilot.cluster.containers import ContainerConfig, ContainerManager, runtime_dockerfile
from servepilot.cluster.inventory import NodeInventory, SSHNode
from servepilot.cluster.network import check_network
from servepilot.engines.base import LaunchSpec, ReadinessResult
from servepilot.engines.interpreter import EngineRuntime
from servepilot.engines.sglang import SGLangEngine
from servepilot.engines.vllm import VLLMEngine
from servepilot.exceptions import ConfigurationError, LaunchError
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan, EngineName, PrefillConfig
from servepilot.testing.fake_hardware import cluster


class RecordedProcess:
    def __init__(self, spec: LaunchSpec) -> None:
        self.spec = spec
        self.pid = self.create_time = None
        self.returncode: int | None = None

    def is_running(self) -> bool:
        return self.returncode is None

    def stdout_tail(self) -> str:
        return ""

    def stderr_tail(self) -> str:
        return ""

    async def terminate(self, grace_seconds: float = 10) -> None:
        self.returncode = 0

    async def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


@pytest.mark.parametrize("engine", [EngineName.VLLM, EngineName.SGLANG])
async def test_native_two_node_launch_uses_correct_ranks_gpu_uuids_and_isolation(
    engine: EngineName, dense_8b: ModelProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    hardware = cluster(2, 4)
    inventory = NodeInventory(
        nodes=[
            SSHNode(host=f"node-{rank}", address=f"10.0.0.{rank + 1}", local=False)
            for rank in range(2)
        ]
    )
    manager = ContainerManager(inventory, ContainerConfig(bootstrap=False), namespace="test-native")
    launcher = ContainerLauncher(manager, hardware, model_volume="model-volume")
    plan = CandidatePlan(
        id="native",
        engine=engine,
        gpu_groups=[list(range(8))],
        tensor_parallel_size=8,
        replica_count=1,
        context_length=8192,
        distributed_backend="mp" if engine == EngineName.VLLM else "native",
        replica_nodes=[[g.node_id for g in hardware.gpus]],
    )
    launcher.configure(plan, {"images": {"node-0": "sha256:one", "node-1": "sha256:two"}})
    adapter = (
        VLLMEngine(runtime=EngineRuntime("python3", "0.28.0"))
        if engine == EngineName.VLLM
        else SGLangEngine(runtime=EngineRuntime("python3", "0.5.19"))
    )
    spec = adapter.build_launch_spec(
        dense_8b, plan, replica_index=0, host="0.0.0.0", port=8123, node=("node-0", "10.0.0.1")
    )
    spec.env.update(
        HF_TOKEN="private-model-key",
        OPENAI_API_KEY="reasoning-key",
        PATH="/host/bin",
        NCCL_SOCKET_IFNAME="eth0",
    )
    processes = []

    async def start(container: LaunchSpec, *_args: object) -> RecordedProcess:
        process = RecordedProcess(container)
        processes.append(process)
        return process

    monkeypatch.setattr(manager, "start_guarded", start)
    monkeypatch.setattr(launcher, "_port", AsyncMock(return_value=23456))
    group = await launcher.launch(spec)
    assert len(processes) == 2 and group.is_running()
    for rank, process in enumerate(processes):
        command = process.spec.command
        assert command[:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
        assert command[command.index("--node-rank") + 1] == str(rank)
        assert command[command.index("--nnodes") + 1] == "2"
        assert (
            command[command.index("--gpus") + 1]
            == '"device=' + ",".join(g.uuid for g in hardware.gpus[rank * 4 : rank * 4 + 4]) + '"'
        )
        assert "type=volume,source=model-volume,target=/model-cache,readonly" in command
        assert "NCCL_SOCKET_IFNAME=eth0" in command
        assert not any(
            "reasoning-key" in arg or "private-model-key" in arg or "/host/bin" in arg
            for arg in command
        )
        # The controller connects to the worker's daemon, but containers cannot access it.
        assert not any("docker.sock" in arg for arg in command[command.index("run") + 1 :])
        if engine == EngineName.VLLM:
            assert "vllm.entrypoints.cli.main" in command
            assert ("--headless" in command) == (rank > 0)
        else:
            assert command[command.index("--dist-init-addr") + 1] == "10.0.0.1:23456"
    processes[1].returncode = 1
    assert not group.is_running()
    assert launcher.tracked(), "surviving ranks must remain tracked after another rank dies"
    await group.terminate()
    assert not launcher.tracked()


async def test_failed_remote_launch_cleans_previous_rank(
    dense_8b: ModelProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    hardware = cluster(2, 1)
    manager = ContainerManager(
        NodeInventory(nodes=[SSHNode(host="node-0"), SSHNode(host="node-1")]),
        ContainerConfig(),
        namespace="test-failure",
    )
    launcher = ContainerLauncher(manager, hardware, model_volume="models")
    plan = CandidatePlan(
        id="test",
        engine=EngineName.VLLM,
        gpu_groups=[[0, 1]],
        tensor_parallel_size=2,
        replica_count=1,
        context_length=8192,
        distributed_backend="mp",
    )
    launcher.configure(plan, {"images": {"node-0": "image", "node-1": "image"}})
    spec = VLLMEngine(runtime=EngineRuntime("python3", "0.28.0")).build_launch_spec(
        dense_8b, plan, replica_index=0, host="0.0.0.0", port=8123
    )
    first = RecordedProcess(spec)
    start = AsyncMock(side_effect=[first, LaunchError("SSH worker disconnected")])
    monkeypatch.setattr(manager, "start_guarded", start)
    monkeypatch.setattr(manager, "cleanup", first.terminate)
    monkeypatch.setattr(launcher, "_port", AsyncMock(return_value=23456))
    with pytest.raises(LaunchError, match="disconnected"):
        await launcher.launch(spec)
    assert not first.is_running()


async def test_real_pairwise_tcp_preflight_from_cpu_controller() -> None:
    inventory = NodeInventory(
        nodes=[
            SSHNode(host="node-0", address="127.0.0.1", local=True),
            SSHNode(host="node-1", address="127.0.0.1", local=True),
        ]
    )
    manager = ContainerManager(
        inventory, ContainerConfig(bootstrap=False), namespace="test-network"
    )
    report = await check_network(manager, cluster(2, 1))
    assert report["controller_connectivity"] == "passed"
    assert {(pair["source"], pair["target"]) for pair in report["tcp"]} == {
        ("node-0", "node-1"),
        ("node-1", "node-0"),
    }
    assert all(
        pair["bytes"] == 32 * 1024 * 1024 and pair["gigabits_per_second"] > 0
        for pair in report["tcp"]
    )


def test_recipe_dockerfile_preserves_literal_shell_and_rejects_from_injection() -> None:
    command = "printf '%s' '$(touch /tmp/not-executed-by-controller)'"
    dockerfile = runtime_dockerfile("vllm/vllm-openai@sha256:123", [command])
    assert "RUN [" in dockerfile and "$(touch" in dockerfile
    with pytest.raises(ConfigurationError):
        runtime_dockerfile("image\nRUN unwanted", [])


async def test_prefill_decode_workers_use_separate_gpus_and_a_cpu_gateway(
    dense_8b: ModelProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    hardware = cluster(2, 1)
    manager = ContainerManager(
        NodeInventory(
            nodes=[
                SSHNode(host="node-0", address="10.0.0.1"),
                SSHNode(host="node-1", address="10.0.0.2"),
            ]
        ),
        ContainerConfig(),
        namespace="test-pd",
    )
    launcher = ContainerLauncher(manager, hardware, model_volume="models")
    plan = CandidatePlan(
        id="pd",
        engine=EngineName.SGLANG,
        gpu_groups=[[1]],
        tensor_parallel_size=1,
        replica_count=1,
        context_length=8192,
        prefill=PrefillConfig(gpu_groups=[[0]], tensor_parallel_size=1),
    )
    runtime = {
        "images": {"node-0": "prefill-image", "node-1": "decode-image"},
        "engine_version": "0.5.19",
    }
    launcher.configure(plan, runtime)
    adapter = SGLangEngine(runtime=EngineRuntime("python3", "0.5.19"))
    spec = adapter.build_launch_spec(
        dense_8b, plan, replica_index=0, host="0.0.0.0", port=8123, node=("node-1", "10.0.0.2")
    )
    processes = []

    async def start(container: LaunchSpec, *_args: object) -> RecordedProcess:
        process = RecordedProcess(container)
        processes.append(process)
        return process

    monkeypatch.setattr(manager, "start_guarded", start)
    monkeypatch.setattr(
        ContainerLauncher, "_port", AsyncMock(side_effect=[31001, 32001, 31002, 32002])
    )
    monkeypatch.setattr(
        SGLangEngine,
        "wait_until_ready",
        AsyncMock(return_value=ReadinessResult(ready=True, elapsed_seconds=0)),
    )
    group = await launcher.launch(spec)
    assert group.is_running() and len(processes) == 3
    prefill, decode, router = [process.spec.command for process in processes]
    assert prefill[prefill.index("--disaggregation-mode") + 1] == "prefill"
    assert decode[decode.index("--disaggregation-mode") + 1] == "decode"
    assert '"device=GPU-fake-node-0-00"' in prefill
    assert '"device=GPU-fake-node-1-00"' in decode
    assert "--gpus" not in router
    assert router[router.index("--prefill") + 1 : router.index("--prefill") + 3] == [
        "http://10.0.0.1:31001",
        "32001",
    ]
    assert router[router.index("--decode") + 1] == "http://10.0.0.2:31002"
    assert router[router.index("--port") + 1] == "8123"
    await group.terminate()
    assert all(not process.is_running() for process in processes)


def test_native_vllm_data_parallel_assigns_local_ranks(deepseek_v3: ModelProfile) -> None:
    plan = CandidatePlan(
        id="dp",
        engine=EngineName.VLLM,
        gpu_groups=[list(range(8))],
        tensor_parallel_size=8,
        data_parallel_size=8,
        dp_attention_enabled=True,
        expert_parallel_enabled=True,
        replica_count=1,
        context_length=8192,
        distributed_backend="mp",
    )
    spec = VLLMEngine(runtime=EngineRuntime("python3", "0.28.0")).build_launch_spec(
        deepseek_v3, plan, replica_index=0, host="0.0.0.0", port=8123
    )
    worker = native_command(spec, plan, rank=1, nodes=2, master="10.0.0.1", port=23456)
    assert worker[worker.index("--tensor-parallel-size") + 1] == "1"
    assert worker[worker.index("--data-parallel-size-local") + 1] == "4"
    assert worker[worker.index("--data-parallel-start-rank") + 1] == "4"
    assert "--headless" in worker and "--nnodes" not in worker
