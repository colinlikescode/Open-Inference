"""Standalone stdlib-only engine guardian, sent over SSH; no remote package install required.

The controller sends one JSON launch descriptor, then heartbeats. EOF or a lost controller
lease terminates the process group and any explicitly owned container. This file is also
executed by tests with a local Python interpreter using the same wire protocol.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from typing import Any


def main() -> None:
    payload = json.loads(sys.stdin.buffer.readline())
    stopped = threading.Event()
    lock = threading.Lock()

    def emit(event: dict[str, Any]) -> None:
        with lock:
            try:
                print(json.dumps(event), flush=True)
            except (BrokenPipeError, OSError):
                stopped.set()

    def stop(*args: object) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    proc = subprocess.Popen(
        payload["command"],
        env={**os.environ, **payload.get("env", {})},
        cwd=payload.get("cwd"),
        stdin=subprocess.PIPE if payload.get("input_text") is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    emit({"event": "started", "pid": proc.pid})

    if payload.get("input_text") is not None:
        assert proc.stdin is not None
        proc.stdin.write(payload["input_text"].encode())
        proc.stdin.close()

    def pump(stream: Any, name: str) -> None:
        while chunk := stream.readline(16384):
            emit({"event": "log", "stream": name, "text": chunk.decode(errors="replace").rstrip()})

    readers = []
    for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        reader = threading.Thread(target=pump, args=(stream, name), daemon=True)
        readers.append(reader)
        reader.start()
    heartbeat = time.monotonic()
    try:
        while proc.poll() is None and not stopped.is_set():
            ready, _, _ = select.select([sys.stdin.buffer], [], [], 0.25)
            if ready:
                line = sys.stdin.buffer.readline()
                if not line or line.strip() == b"stop":
                    stopped.set()
                    break
                heartbeat = time.monotonic()
            if time.monotonic() - heartbeat > float(payload.get("lease_seconds", 30)):
                stopped.set()
                emit(
                    {
                        "event": "log",
                        "stream": "stderr",
                        "text": "controller lease expired; stopping owned runtime",
                    }
                )
                break
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=float(payload.get("grace_seconds", 10)))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
        # Also remove descendants whose parent exited without reaping them.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        cleanup = payload.get("cleanup_command")
        if cleanup and not (
            payload.get("preserve_on_success") and proc.returncode == 0 and not stopped.is_set()
        ):
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    cleanup,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                    check=False,
                )
        for reader in readers:
            reader.join(timeout=2)
        emit({"event": "exited", "returncode": proc.returncode})


if __name__ == "__main__":
    main()
