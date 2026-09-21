"""Stage immutable model weights on supplied nodes before experimental code runs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from servepilot.cluster.containers import ContainerManager
from servepilot.cluster.inventory import SSHNode
from servepilot.cluster.ssh_launcher import SSHProcessHandle
from servepilot.cluster.transport import SSHTransport
from servepilot.engines.base import LaunchSpec
from servepilot.exceptions import ConfigurationError, LaunchError
from servepilot.optimization.schemas import fingerprint
from servepilot.schemas.model import ModelProfile


def local_manifest(directory: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                manifest[path.relative_to(directory).as_posix()] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    if not manifest or "config.json" not in manifest:
        raise ConfigurationError("local model must include config.json and model files")
    return manifest


class ModelCache:
    def __init__(self, manager: ContainerManager, model: ModelProfile) -> None:
        self.manager = manager
        self.model = model
        self.volume = (
            "opensandbox-model-"
            + fingerprint({"id": model.model_id, "revision": model.revision})[:24]
        )

    async def _run(self, node: SSHNode, image: str, program: str, payload: dict[str, Any]) -> None:
        name = self.manager.container_name("download")
        argv = self.manager.docker(
            node,
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--label",
            f"opensandbox.run={self.manager.namespace}",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--mount",
            f"type=volume,source={self.volume},target=/model-cache",
            "--env",
            "HF_HOME=/model-cache",
            "--entrypoint",
            "python3",
            image,
            "-c",
            program,
        )
        spec = LaunchSpec(
            executable=argv[0],
            args=argv[1:],
            env={},
            host=node.network_address,
            port=0,
            gpu_ids=[],
            node_id=node.node_id,
            redacted_display_command=shlex.join(argv),
        )
        guardian = SSHProcessHandle(spec, self.manager.transport)
        try:
            # The credential travels only in SSH stdin to a trusted base image. It is never
            # written into an image, environment, command line, recipe, or experimental image.
            await guardian.start(
                cleanup_command=self.manager.docker(node, "rm", "--force", name),
                input_text=json.dumps(payload),
            )
            if self.manager.budget:
                await self.manager.budget.run(lambda: guardian.wait())
            else:
                await guardian.wait()
            if guardian.returncode != 0:
                detail = guardian.stderr_tail()
                if payload.get("token"):
                    detail = detail.replace(payload["token"], "[REDACTED]")
                raise LaunchError(f"model staging failed on {node.host}", hints=[detail[-4000:]])
        finally:
            await guardian.terminate()
            await self.manager.remove(node, name)

    async def _upload(self, node: SSHNode, image: str, directory: Path) -> None:
        name = self.manager.container_name("upload")
        argv = self.manager.docker(
            node,
            "run",
            "--rm",
            "-i",
            "--name",
            name,
            "--label",
            f"opensandbox.run={self.manager.namespace}",
            "--network",
            "none",
            "--mount",
            f"type=volume,source={self.volume},target=/model-cache",
            "--entrypoint",
            "/bin/sh",
            image,
            "-ec",
            "mkdir -p /model-cache/local; tar -xf - -C /model-cache/local",
        )
        remote = await self.manager.transport.start(node, argv)
        local = await asyncio.create_subprocess_exec(
            "tar",
            "-chf",
            "-",
            "-C",
            str(directory),
            ".",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert remote.stdin and remote.stdout and remote.stderr and local.stdout and local.stderr
        drains = [
            asyncio.create_task(remote.stdout.read()),
            asyncio.create_task(remote.stderr.read()),
            asyncio.create_task(local.stderr.read()),
        ]
        try:
            while chunk := await local.stdout.read(1024 * 1024):
                remote.stdin.write(chunk)
                await remote.stdin.drain()
            remote.stdin.close()
            if await local.wait() or await remote.wait():
                raise LaunchError(f"local model upload failed on {node.host}")
        finally:
            await SSHTransport.stop_process(local)
            await SSHTransport.stop_process(remote)
            for task in drains:
                task.cancel()
            await asyncio.gather(*drains, return_exceptions=True)
            await self.manager.remove(node, name)

    async def prepare(
        self, *, token: str | None = None, manifest: dict[str, str] | None = None
    ) -> None:
        image = next(iter(self.manager.images.values()))
        for node in self.manager.inventory.nodes:
            await self.manager.command(node, ["volume", "create", self.volume])
            if self.model.local_path:
                if not manifest or fingerprint(manifest) != self.model.revision:
                    raise ConfigurationError(
                        "local model content no longer matches the pinned revision"
                    )
                directory = Path(self.model.local_path)

                async def operation(target: SSHNode = node, source: Path = directory) -> None:
                    await self._upload(target, image, source)

                if self.manager.budget:
                    await self.manager.budget.run(operation)
                else:
                    await operation()
                # Detect a changing source directory, truncated copies, and corrupted weights.
                await self._run(
                    node,
                    image,
                    "import hashlib,json,pathlib,sys; manifest=json.load(sys.stdin); root=pathlib.Path('/model-cache/local'); actual={p.relative_to(root).as_posix():hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in root.rglob('*') if p.is_file()}; assert actual==manifest, 'model checksum mismatch'",
                    manifest,
                )
            else:
                if not self.model.revision:
                    raise ConfigurationError("a resolved model revision is required")
                await self._run(
                    node,
                    image,
                    "import json,sys; from huggingface_hub import snapshot_download; p=json.load(sys.stdin); snapshot_download(repo_id=p['model'], revision=p['revision'], token=p.get('token'), cache_dir='/model-cache/hub'); print('Model snapshot ready')",
                    {"model": self.model.model_id, "revision": self.model.revision, "token": token},
                )

    def runtime_model(self) -> ModelProfile:
        if self.model.local_path:
            return self.model.model_copy(
                update={"local_path": "/model-cache/local", "revision": None}
            )
        return self.model


async def import_image(
    manager: ContainerManager, node: SSHNode, archive: Path, expected: str
) -> None:
    proc = await manager.transport.start(node, manager.docker(node, "load"))
    assert proc.stdin and proc.stdout and proc.stderr
    readers = [asyncio.create_task(proc.stdout.read()), asyncio.create_task(proc.stderr.read())]
    try:
        with archive.open("rb") as stream:
            while chunk := await asyncio.to_thread(stream.read, 1024 * 1024):
                proc.stdin.write(chunk)
                await proc.stdin.drain()
        proc.stdin.close()
        if await proc.wait():
            raise LaunchError(f"could not import saved engine image on {node.host}")
        await manager.command(node, ["image", "inspect", expected])
    finally:
        await SSHTransport.stop_process(proc)
        for task in readers:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*readers, return_exceptions=True)
