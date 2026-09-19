"""Argument-safe SSH control of existing machines; no provider APIs or machine lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
from dataclasses import dataclass

from servepilot.cluster.inventory import NodeInventory, SSHNode
from servepilot.exceptions import LaunchError
from servepilot.logging import redact_secrets


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class SSHTransport:
    def __init__(self, inventory: NodeInventory) -> None:
        self.inventory = inventory

    def argv(self, node: SSHNode, command: list[str]) -> list[str]:
        if not command:
            raise ValueError("empty remote command")
        if node.is_local:
            return list(command)
        args = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"ConnectTimeout={self.inventory.connect_timeout_seconds}",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(node.port),
        ]
        if node.identity_file:
            args.extend(["-i", node.identity_file, "-o", "IdentitiesOnly=yes"])
        if self.inventory.ssh_known_hosts:
            args.extend(["-o", f"UserKnownHostsFile={self.inventory.ssh_known_hosts}"])
        destination = f"{node.user}@{node.host}" if node.user else node.host
        # OpenSSH joins the remote command into a shell string. Quote every argv element once.
        return [*args, "--", destination, shlex.join(command)]

    async def start(self, node: SSHNode, command: list[str]) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                *self.argv(node, command),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise LaunchError(f"{node.host}: cannot start control command: {exc}") from exc

    async def run(
        self,
        node: SSHNode,
        command: list[str],
        *,
        input_data: bytes | None = None,
        timeout: float = 60,
        check: bool = True,
    ) -> CommandResult:
        proc = await self.start(node, command)
        try:
            async with asyncio.timeout(timeout):
                stdout, stderr = await proc.communicate(input_data)
        except BaseException:
            await self.stop_process(proc)
            raise
        result = CommandResult(
            proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
        )
        if check and result.returncode:
            raise LaunchError(
                f"{node.host}: command failed with exit {result.returncode}",
                hints=[redact_secrets(result.stderr[-4000:] or result.stdout[-4000:])],
            )
        return result

    @staticmethod
    async def stop_process(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            async with asyncio.timeout(5):
                await proc.wait()
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
