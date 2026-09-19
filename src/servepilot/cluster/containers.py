"""Isolated engine images on existing SSH nodes, with no controller filesystem mounts.

Experimental commands run inside containers without the Docker socket, host PID namespace,
host credentials, or writable model cache. Successful changes become new image layers; failed
commands discard their container. Only controller code invokes Docker on the host.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import shlex
import uuid
from pathlib import Path
from typing import Any

from pydantic import Field

from servepilot.cluster.inventory import NodeInventory, SSHNode
from servepilot.cluster.ssh_launcher import SSHProcessHandle
from servepilot.cluster.transport import CommandResult, SSHTransport
from servepilot.engines.base import LaunchSpec
from servepilot.exceptions import ConfigurationError, LaunchError, ServePilotError
from servepilot.optimization.budget import TimeBudget
from servepilot.optimization.schemas import Contract, RuntimeChanges, canonical_json
from servepilot.optimization.store import ExperimentStore


class ContainerConfig(Contract):
    images: dict[str, str] = Field(
        default_factory=lambda: {
            "vllm": "vllm/vllm-openai:v0.28.0",
            "sglang": "lmsysorg/sglang:v0.5.19",
        }
    )
    bootstrap: bool = True
    shm_size: str = "16g"


def change_commands(changes: RuntimeChanges) -> list[str]:
    """File edits become literal, reproducible container commands, never host shell strings."""
    commands = []
    for file in changes.files:
        encoded = base64.b64encode(file.content.encode()).decode()
        path = "/opt/openbaseten/changes/" + file.path
        program = (
            "import base64,pathlib; p=pathlib.Path("
            + repr(path)
            + "); p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(base64.b64decode("
            + repr(encoded)
            + "))"
        )
        commands.append(shlex.join(["python3", "-c", program]))
    return [*commands, *changes.commands]


def runtime_dockerfile(base_image: str, commands: list[str]) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]*", base_image):
        raise ConfigurationError("invalid base image reference")
    lines = [f"FROM {base_image}", "WORKDIR /opt/openbaseten/changes"]
    lines.extend("RUN " + json.dumps(["/bin/sh", "-ec", command]) for command in commands)
    return "\n".join(lines) + "\n"


class ContainerManager:
    def __init__(
        self,
        inventory: NodeInventory,
        config: ContainerConfig,
        *,
        namespace: str,
        budget: TimeBudget | None = None,
        transport: SSHTransport | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9-]{1,48}", namespace):
            raise ValueError("invalid container namespace")
        self.inventory = inventory
        self.config = config
        self.namespace = namespace
        self.budget = budget
        self.transport = transport or SSHTransport(inventory)
        self.prefixes: dict[str, list[str]] = {}
        self.images: dict[str, str] = {}
        self.versions: dict[str, str] = {}
        self.details: dict[str, Any] = {}
        self.network: dict[str, Any] = {}
        self._handles: list[SSHProcessHandle] = []

    def docker(self, node: SSHNode, *arguments: str) -> list[str]:
        # A user's Docker context must not redirect work to a machine outside this inventory.
        return [
            *self.prefixes.get(node.node_id, ["docker"]),
            "--host",
            "unix:///var/run/docker.sock",
            *arguments,
        ]

    async def command(
        self,
        node: SSHNode,
        arguments: list[str],
        *,
        timeout: float = 1200,
        input_data: bytes | None = None,
        check: bool = True,
    ) -> CommandResult:
        async def run() -> CommandResult:
            return await self.transport.run(
                node,
                self.docker(node, *arguments),
                timeout=timeout,
                input_data=input_data,
                check=check,
            )

        return await self.budget.run(run) if self.budget is not None else await run()

    async def prepare(self, engines: list[str]) -> None:
        async def prepare_node(node: SSHNode) -> None:
            if self.config.bootstrap:
                script = Path(__file__).with_name("bootstrap.sh").read_bytes()

                async def operation() -> CommandResult:
                    return await self.transport.run(
                        node, ["sh", "-s"], input_data=script, timeout=1200
                    )

                if self.budget is not None:
                    await self.budget.run(operation)
                else:
                    await operation()
            probe = await self.transport.run(
                node,
                [
                    "docker",
                    "--host",
                    "unix:///var/run/docker.sock",
                    "info",
                    "--format",
                    "{{json .ServerVersion}}",
                ],
                check=False,
            )
            if probe.returncode:
                probe = await self.transport.run(
                    node,
                    [
                        "sudo",
                        "-n",
                        "docker",
                        "--host",
                        "unix:///var/run/docker.sock",
                        "info",
                        "--format",
                        "{{json .ServerVersion}}",
                    ],
                    check=False,
                )
                if probe.returncode:
                    raise ConfigurationError(
                        f"{node.host}: Docker is unavailable; bootstrap needs root or passwordless sudo",
                        hints=[probe.stderr[-2000:]],
                    )
                self.prefixes[node.node_id] = ["sudo", "-n", "docker"]
            else:
                self.prefixes[node.node_id] = ["docker"]
            host_probe = await self.transport.run(
                node,
                [
                    node.python,
                    "-c",
                    "import pathlib,json; print(json.dumps({'infiniband':pathlib.Path('/dev/infiniband').is_dir(),'rdma_devices':[p.name for p in pathlib.Path('/sys/class/infiniband').glob('*')],'interfaces':[p.name for p in pathlib.Path('/sys/class/net').glob('*')]}))",
                ],
            )
            self.details.setdefault("nodes", {})[node.node_id] = json.loads(host_probe.stdout)

        await asyncio.gather(*(prepare_node(node) for node in self.inventory.nodes))
        first = self.inventory.nodes[0]
        for engine in engines:
            image = self.config.images.get(engine)
            if image is None:
                raise ConfigurationError(f"no container image configured for {engine}")
            await self.command(first, ["pull", image])
            result = await self.command(
                first, ["image", "inspect", image, "--format", "{{json .RepoDigests}}"]
            )
            digests = json.loads(result.stdout)
            if not digests:
                raise ConfigurationError(f"cannot resolve immutable digest of {image}")
            pinned = str(digests[0])
            self.images[engine] = pinned
            for node in self.inventory.nodes:
                if node != first:
                    await self.command(node, ["pull", pinned])
                program = (
                    "import importlib.metadata as m,json,torch,platform; "
                    f"print(json.dumps({{'engine':m.version({engine!r}),'torch':torch.__version__,'cuda':torch.version.cuda,'architecture':platform.machine()}})); "
                    "assert torch.cuda.is_available(), 'NVIDIA driver/CUDA/container compatibility check failed'; "
                    "x=torch.ones(1,device='cuda'); assert x.item()==1"
                )
                probe = await self.probe(node, pinned, program, gpus=True)
                payload = json.loads(probe.stdout.strip().splitlines()[-1])
                self.details.setdefault(engine, {})[node.node_id] = payload
                if engine in self.versions and self.versions[engine] != payload["engine"]:
                    raise ConfigurationError(f"{engine} version differs across nodes")
                self.versions[engine] = payload["engine"]

    def container_name(self, purpose: str) -> str:
        return f"{self.namespace}-{purpose}-{uuid.uuid4().hex[:10]}"

    async def probe(
        self,
        node: SSHNode,
        image: str,
        program: str,
        *,
        arguments: list[str] | None = None,
        gpus: bool = False,
    ) -> CommandResult:
        name = self.container_name("probe")
        command = self.docker(
            node,
            "run",
            "--rm",
            "--name",
            name,
            "--label",
            f"openbaseten.run={self.namespace}",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            *(["--gpus", "all"] if gpus else []),
            "--entrypoint",
            "python3",
            image,
            "-c",
            program,
            *(arguments or []),
        )
        spec = LaunchSpec(
            executable=command[0],
            args=command[1:],
            env={},
            host=node.network_address,
            port=0,
            gpu_ids=[],
            node_id=node.node_id,
            redacted_display_command=shlex.join(command),
        )
        handle = await self.start_guarded(spec, node, name)
        try:
            if self.budget:
                await self.budget.run(lambda: handle.wait(), limit=120)
            elif await handle.wait(timeout=120) is None:
                raise LaunchError(f"runtime probe timed out on {node.host}")
            result = CommandResult(
                handle.returncode or 0, handle.stdout_tail(), handle.stderr_tail()
            )
            if result.returncode:
                raise LaunchError(
                    f"runtime probe failed on {node.host}", hints=[result.stderr[-4000:]]
                )
            return result
        finally:
            await handle.terminate()
            await self.remove(node, name)

    async def remove(self, node: SSHNode, name: str) -> None:
        # Cleanup deliberately bypasses the expired optimization budget.
        result = await self.transport.run(
            node, self.docker(node, "rm", "--force", name), timeout=45, check=False
        )
        if result.returncode and "No such container" not in result.stderr:
            raise LaunchError(
                f"{node.host}: could not remove owned container {name}",
                hints=[result.stderr[-2000:]],
            )

    async def run_step(self, node: SSHNode, image: str, command: str) -> tuple[str, str]:
        name = self.container_name("edit")
        script = (
            "set -eu\nmkdir -p /opt/openbaseten/changes\ncd /opt/openbaseten/changes\n" + command
        )
        try:
            await self.command(
                node,
                [
                    "create",
                    "--name",
                    name,
                    "--label",
                    f"openbaseten.run={self.namespace}",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges",
                    "--network",
                    "bridge",
                    "--entrypoint",
                    "/bin/sh",
                    image,
                    "-ec",
                    script,
                ],
            )
            # A bounded attached process executes the edit. Discard the container on failure.
            start = self.docker(node, "start", "--attach", name)
            spec = LaunchSpec(
                executable=start[0],
                args=start[1:],
                env={},
                host=node.network_address,
                port=0,
                gpu_ids=[],
                node_id=node.node_id,
                redacted_display_command=shlex.join(start),
            )
            guardian = SSHProcessHandle(spec, self.transport)
            self._handles.append(guardian)
            await guardian.start(
                cleanup_command=self.docker(node, "rm", "--force", name), preserve_on_success=True
            )
            if self.budget is not None:
                await self.budget.run(lambda: guardian.wait())
            else:
                await guardian.wait()
            result = CommandResult(
                guardian.returncode or 0, guardian.stdout_tail(), guardian.stderr_tail()
            )
            inspected = await self.command(
                node, ["inspect", name, "--format", "{{.State.ExitCode}}"]
            )
            if int(inspected.stdout.strip()) != 0:
                raise LaunchError(
                    f"isolated runtime command failed on {node.host}",
                    hints=[result.stdout[-4000:], result.stderr[-4000:]],
                )
            committed = await self.command(node, ["commit", name])
            image_id = committed.stdout.strip()
            if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
                raise LaunchError("Docker returned an invalid committed image id")
            return image_id, result.stdout[-32000:] + result.stderr[-32000:]
        finally:
            if "guardian" in locals():
                await guardian.terminate()
            await self.remove(node, name)

    async def build(
        self,
        engine: str,
        changes: RuntimeChanges,
        *,
        store: ExperimentStore,
        number: int,
        parent: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = self.images[engine]
        inherited = list((parent or {}).get("commands_applied", []))
        commands = change_commands(changes)
        images: dict[str, str] = {}
        output: dict[str, list[str]] = {}
        for node in self.inventory.nodes:
            if parent is not None and node.node_id not in parent.get("images", {}):
                raise ConfigurationError(f"parent runtime is missing node {node.node_id}")
            image = parent["images"][node.node_id] if parent is not None else base
            # Parent snapshots must exist on each node; do not silently omit accepted changes.
            await self.command(node, ["image", "inspect", image])
            logs = []
            for step, command in enumerate(commands):
                before = image
                relative = f"experiments/experiment_{number:04d}/build-node-{self.inventory.nodes.index(node)}-step-{step}.json"
                try:
                    image, log = await self.run_step(node, image, command)
                except BaseException as exc:
                    detail = exc.render() if isinstance(exc, ServePilotError) else str(exc)
                    store.artifact(
                        relative,
                        canonical_json(
                            {
                                "command": command,
                                "image_before": before,
                                "status": "failed",
                                "error": detail,
                            }
                        ).encode(),
                    )
                    raise
                store.artifact(
                    relative,
                    canonical_json(
                        {
                            "command": command,
                            "image_before": before,
                            "image_after": image,
                            "status": "completed",
                            "output": log,
                        }
                    ).encode(),
                )
                logs.append(log)
            images[node.node_id] = image
            output[node.node_id] = logs
        paths = []
        for file in {file.path: file for file in [*changes.files, *changes.artifacts]}.values():
            folder = "kernels" if file.path.endswith((".cu", ".cuh", ".py")) else "patches"
            relative = f"{folder}/experiment_{number:04d}/{file.path}"
            store.artifact(relative, file.content.encode())
            paths.append(relative)
        relative = f"experiments/experiment_{number:04d}/runtime-build.json"
        store.artifact(relative, canonical_json({"commands": commands, "logs": output}).encode())
        paths.append(relative)
        dockerfile = runtime_dockerfile(base, [*inherited, *commands])
        environment = {**(parent or {}).get("environment", {}), **changes.environment}
        for key, value in sorted(environment.items()):
            dockerfile += "ENV " + key + "=" + json.dumps(value).replace("$", "\\$") + "\n"
        versions = {}
        for node in self.inventory.nodes:
            version = await self.probe(
                node,
                images[node.node_id],
                f"import importlib.metadata; print(importlib.metadata.version({engine!r}))",
            )
            versions[node.node_id] = version.stdout.strip()
        if len(set(versions.values())) != 1:
            raise ConfigurationError("edited engine versions differ across nodes")
        return {
            "runtime": "docker",
            "base_image": base,
            "images": images,
            "commands_applied": [*inherited, *commands],
            "environment": environment,
            "dockerfile": dockerfile,
            "artifact_paths": [*(parent or {}).get("artifact_paths", []), *paths],
            "engine_version": next(iter(versions.values())),
        }

    async def read_file(self, node: SSHNode, image: str, path: str) -> str:
        program = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.open().read(100000))"
        result = await self.probe(node, image, program, arguments=[path])
        return result.stdout

    async def start_guarded(self, spec: LaunchSpec, node: SSHNode, name: str) -> SSHProcessHandle:
        handle = SSHProcessHandle(spec, self.transport)
        self._handles.append(handle)
        try:
            await handle.start(cleanup_command=self.docker(node, "rm", "--force", name))
        except BaseException:
            await handle.terminate()
            await self.remove(node, name)
            raise
        return handle

    async def cleanup(self) -> None:
        handles, self._handles = self._handles, []
        outcomes = await asyncio.gather(
            *(handle.terminate() for handle in handles), return_exceptions=True
        )
        if any(isinstance(outcome, BaseException) for outcome in outcomes) or any(
            handle.is_running() for handle in handles
        ):
            raise LaunchError("could not verify cleanup of all managed containers")

    async def cleanup_previous_run(self) -> None:
        """Only objects with this exact persisted run label; no global Docker prune or stop."""
        for node in self.inventory.nodes:
            result = await self.transport.run(
                node,
                self.docker(
                    node, "ps", "-aq", "--filter", f"label=openbaseten.run={self.namespace}"
                ),
            )
            for container in result.stdout.split():
                if not re.fullmatch(r"[a-f0-9]{12,64}", container):
                    raise LaunchError("invalid container id returned by Docker")
                await self.remove(node, container)

    async def export_image(self, node: SSHNode, image: str, path: Path) -> str:
        """Stream an exact accepted image to disk; never buffer a multi-GB archive in memory."""
        return await self._export(node, ["save", image], path)

    async def export_profiles(self, node: SSHNode, container: str, path: Path) -> str:
        return await self._export(
            node,
            ["cp", f"{container}:/tmp/openbaseten-profile", "-"],
            path,
            max_bytes=512 * 1024 * 1024,
        )

    async def _export(
        self, node: SSHNode, arguments: list[str], path: Path, *, max_bytes: int | None = None
    ) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        proc = await self.transport.start(node, self.docker(node, *arguments))
        assert proc.stdout is not None and proc.stderr is not None and proc.stdin is not None
        proc.stdin.close()
        stderr_task = asyncio.create_task(proc.stderr.read())
        digest = hashlib.sha256()
        temporary = path.with_suffix(path.suffix + ".partial")
        size = 0
        try:
            with temporary.open("wb") as stream:
                temporary.chmod(0o600)
                while chunk := await proc.stdout.read(1024 * 1024):
                    size += len(chunk)
                    if max_bytes is not None and size > max_bytes:
                        raise LaunchError("profiler archive exceeds the 512 MiB capture limit")
                    await asyncio.to_thread(stream.write, chunk)
                    digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if await proc.wait() != 0:
                raise LaunchError(
                    f"could not export runtime artifact on {node.host}",
                    hints=[(await stderr_task).decode(errors="replace")[-2000:]],
                )
            os.replace(temporary, path)
            return digest.hexdigest()
        finally:
            await SSHTransport.stop_process(proc)
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
            temporary.unlink(missing_ok=True)
