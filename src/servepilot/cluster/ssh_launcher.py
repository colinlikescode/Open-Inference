"""Remote process handles whose local PID always identifies the SSH guardian, not a worker PID."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import psutil

from servepilot.cluster.inventory import NodeInventory
from servepilot.cluster.transport import SSHTransport
from servepilot.engines.base import LaunchSpec
from servepilot.engines.process import ProcessHandle, RingBuffer
from servepilot.exceptions import ConfigurationError, LaunchError
from servepilot.logging import redact_secrets


class SSHProcessHandle:
    def __init__(self, spec: LaunchSpec, transport: SSHTransport) -> None:
        self.spec = spec
        self.transport = transport
        self._proc: asyncio.subprocess.Process | None = None
        self._create_time: float | None = None
        self.remote_pid: int | None = None
        self._stdout = RingBuffer()
        self._stderr = RingBuffer()
        self._tasks: list[asyncio.Task[None]] = []
        self._started = asyncio.Event()
        self._remote_code: int | None = None

    async def start(
        self,
        *,
        cleanup_command: list[str] | None = None,
        preserve_on_success: bool = False,
        input_text: str | None = None,
    ) -> None:
        node = self.transport.inventory.node(self.spec.node_id)
        guardian = Path(__file__).with_name("worker.py").read_text()
        self._proc = await self.transport.start(node, [node.python, "-u", "-c", guardian])
        self._create_time = psutil.Process(self._proc.pid).create_time()
        assert self._proc.stdin is not None
        payload = {
            "command": self.spec.command,
            "env": self.spec.env,
            "cwd": self.spec.cwd,
            "cleanup_command": cleanup_command,
            "preserve_on_success": preserve_on_success,
            "input_text": input_text,
        }
        self._proc.stdin.write(json.dumps(payload).encode() + b"\n")
        self._tasks = [
            asyncio.create_task(self._read_events()),
            asyncio.create_task(self._read_stderr()),
            asyncio.create_task(self._heartbeat()),
        ]
        try:
            await self._proc.stdin.drain()
            async with asyncio.timeout(30):
                await self._started.wait()
            if self.remote_pid is None:
                raise LaunchError(
                    f"{node.host}: remote guardian did not start: {self.stderr_tail()}"
                )
        except BaseException:
            await self.terminate()
            raise

    async def _read_events(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while line := await self._proc.stdout.readline():
                try:
                    event: dict[str, Any] = json.loads(line)
                    if event["event"] == "started":
                        self.remote_pid = int(event["pid"])
                        self._started.set()
                    elif event["event"] == "log":
                        target = self._stdout if event["stream"] == "stdout" else self._stderr
                        target.append(redact_secrets(event["text"]))
                    elif event["event"] == "exited":
                        self._remote_code = int(event["returncode"])
                except (ValueError, KeyError, TypeError):
                    self._stderr.append("invalid event from remote guardian")
        finally:
            self._started.set()

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while line := await self._proc.stderr.readline():
            self._stderr.append(redact_secrets(line.decode(errors="replace").rstrip()))

    async def _heartbeat(self) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            while self._proc.returncode is None:
                await asyncio.sleep(5)
                self._proc.stdin.write(b"heartbeat\n")
                await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionError):
            pass

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def create_time(self) -> float | None:
        return self._create_time

    @property
    def returncode(self) -> int | None:
        if self._remote_code is not None:
            return self._remote_code
        return self._proc.returncode if self._proc is not None else None

    def is_running(self) -> bool:
        return self._proc is not None and self.returncode is None

    def stdout_tail(self) -> str:
        return self._stdout.text()

    def stderr_tail(self) -> str:
        return self._stderr.text()

    async def wait(self, timeout: float | None = None) -> int | None:
        if self._proc is None:
            return None
        try:
            async with asyncio.timeout(timeout):
                await self._proc.wait()
                if self._tasks:
                    await asyncio.gather(*self._tasks[:2])
            return self.returncode
        except TimeoutError:
            return None

    async def terminate(self, grace_seconds: float = 10) -> None:
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError, ConnectionError):
                self._proc.stdin.write(b"stop\n")
                await self._proc.stdin.drain()
                self._proc.stdin.close()
        if await self.wait(timeout=grace_seconds + 35) is None:
            await self.transport.stop_process(self._proc)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []


class SSHLauncher:
    def __init__(self, inventory: NodeInventory, transport: SSHTransport | None = None) -> None:
        self.inventory = inventory
        self.transport = transport or SSHTransport(inventory)
        self._processes: list[SSHProcessHandle] = []

    async def launch(self, spec: LaunchSpec) -> ProcessHandle:
        handle = SSHProcessHandle(spec, self.transport)
        self._processes.append(handle)
        try:
            await handle.start()
        except BaseException:
            await handle.terminate()
            raise
        return handle

    async def shutdown_all(self, grace_seconds: float = 10) -> None:
        processes, self._processes = self._processes, []
        results = await asyncio.gather(
            *(p.terminate(grace_seconds) for p in processes), return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures or any(p.is_running() for p in processes):
            raise LaunchError("could not confirm cleanup of all SSH runtimes")

    def tracked(self) -> list[ProcessHandle]:
        return [p for p in self._processes if p.is_running()]

    def supports_node(self, node_id: str | None) -> bool:
        try:
            self.inventory.node(node_id)
            return True
        except ConfigurationError:
            return False
