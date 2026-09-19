"""Status and stop for a controller that is still inspecting or optimizing."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from servepilot.optimization.store import atomic_write
from servepilot.runtime.state import current_process_create_time, process_matches


class ControllerState:
    def __init__(self, directory: Path) -> None:
        self.path = directory / "optimization.json"

    def read(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        data: dict[str, Any] = json.loads(self.path.read_text())
        data["running"] = process_matches(data["pid"], data["create_time"])
        return data

    def write(self, *, phase: str, output: Path, message: str) -> None:
        atomic_write(
            self.path,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "create_time": current_process_create_time(),
                    "phase": phase,
                    "output": str(output.resolve()),
                    "message": message,
                }
            ).encode(),
        )

    def clear(self) -> None:
        data = self.read()
        if data and data["pid"] == os.getpid():
            self.path.unlink(missing_ok=True)

    def stop(self) -> dict[str, Any] | None:
        data = self.read()
        if not data or not data["running"]:
            return None
        os.kill(data["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 45
        while process_matches(data["pid"], data["create_time"]) and time.monotonic() < deadline:
            time.sleep(0.1)
        stopped = not process_matches(data["pid"], data["create_time"])
        return {
            "stopped": stopped,
            "optimization": data,
            "message": "Search stop requested; completed evidence and artifacts are preserved."
            if stopped
            else "Cleanup is still running; check status again.",
        }
