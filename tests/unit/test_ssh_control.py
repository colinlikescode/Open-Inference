"""The same guardian wire protocol runs locally and over SSH."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import psutil
import pytest
from pydantic import ValidationError

from servepilot.cluster.inventory import NodeInventory, SSHNode
from servepilot.cluster.ssh_launcher import SSHLauncher
from servepilot.cluster.transport import SSHTransport
from servepilot.engines.base import LaunchSpec


def spec(code: str) -> LaunchSpec:
    return LaunchSpec(
        executable=sys.executable,
        args=["-u", "-c", code],
        env={},
        host="127.0.0.1",
        port=9999,
        gpu_ids=[],
        redacted_display_command="python test",
        node_id="local",
    )


def test_ssh_quotes_arguments_without_remote_shell_expansion() -> None:
    node = SSHNode(host="worker.example", user="ubuntu", local=False)
    transport = SSHTransport(NodeInventory(nodes=[node]))
    command = ["python3", "-c", "print('$(touch /tmp/unwanted)')", "spaces and `backticks`"]
    args = transport.argv(node, command)
    assert shlex.split(args[-1]) == command
    assert "StrictHostKeyChecking=yes" in args and "BatchMode=yes" in args


@pytest.mark.parametrize(
    "host", ["-oProxyCommand=bad", "host;rm", "user@host", "host space", "$(bad)"]
)
def test_host_options_cannot_be_injected(host: str) -> None:
    with pytest.raises(ValidationError):
        SSHNode(host=host)


def test_dedicated_controller_does_not_need_to_be_a_gpu_node(tmp_path: Path) -> None:
    file = tmp_path / "nodes.yaml"
    file.write_text("nodes:\n  - host: gpu1.example\n  - host: gpu2.example\nssh_user: ubuntu\n")
    inventory = NodeInventory.load(file)
    assert not any(node.is_local for node in inventory.nodes)
    assert all(node.user == "ubuntu" for node in inventory.nodes)


async def test_guardian_tracks_local_identity_and_cleans_remote_tree(tmp_path: Path) -> None:
    child_pid = tmp_path / "child.pid"
    code = f"import subprocess, time, pathlib; p=subprocess.Popen([{sys.executable!r}, '-c', 'import time; time.sleep(60)']); pathlib.Path({str(child_pid)!r}).write_text(str(p.pid)); print('engine started', flush=True); time.sleep(60)"
    launcher = SSHLauncher(NodeInventory.load(None))
    handle = await launcher.launch(spec(code))
    try:
        for _ in range(100):
            if child_pid.exists() and "engine started" in handle.stdout_tail():
                break
            await asyncio.sleep(0.02)
        assert child_pid.exists() and handle.is_running()
        assert handle.pid and handle.create_time
        pid = int(child_pid.read_text())
        assert psutil.pid_exists(pid)
    finally:
        await launcher.shutdown_all()
    assert not handle.is_running()
    for _ in range(50):
        if not psutil.pid_exists(pid):
            break
        await asyncio.sleep(0.02)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


async def test_guardian_stops_engine_when_controller_pipe_closes() -> None:
    launcher = SSHLauncher(NodeInventory.load(None))
    handle = await launcher.launch(spec("import time; time.sleep(60)"))
    assert handle.is_running()
    # Closing stdin simulates a lost SSH connection, including a killed controller.
    process = handle._proc  # type: ignore[attr-defined]
    process.stdin.close()
    assert await handle.wait(10) is not None
    await launcher.shutdown_all()
