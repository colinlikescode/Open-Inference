"""Durable, exclusively locked experiment history owned by the controller.

The agent receives serialized context through tools, not a writable mount of this directory.
Hash chaining detects corruption on resume; atomic replacement avoids partial JSONL records.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import IO, Any

from servepilot.exceptions import ConfigurationError
from servepilot.optimization.schemas import (
    ExperimentProposal,
    ExperimentResult,
    RunDefinition,
    canonical_json,
    fingerprint,
    utc_now,
)


def atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class ExperimentStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self._lock: IO[str] | None = None

    def __enter__(self) -> ExperimentStore:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / ".controller.lock").open("a+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            self._lock = None
            raise ConfigurationError(f"another controller owns {self.directory}") from exc
        return self

    def __exit__(self, *args: object) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def _require_lock(self) -> None:
        if self._lock is None:
            raise RuntimeError("experiment store must be exclusively locked before writing")

    def create(self, definition: RunDefinition) -> None:
        self._require_lock()
        reserved = {".controller.lock"}
        if (self.directory / ".setup.json").is_file():
            reserved.update({".setup.json", "setup-error.json", "report.html"})
        existing = [p for p in self.directory.iterdir() if p.name not in reserved]
        if existing:
            raise ConfigurationError(
                f"output directory {self.directory} is not empty; use --resume or a new directory"
            )
        envelope = {
            "definition": definition.model_dump(mode="json"),
            "sha256": fingerprint(definition),
        }
        atomic_write(self.directory / "run.json", canonical_json(envelope).encode())
        for name in ("experiments", "benchmark-results", "profiler-results", "patches", "kernels"):
            (self.directory / name).mkdir()
        self.append("run_created", {"definition_sha256": fingerprint(definition)})

    def definition(self) -> RunDefinition:
        try:
            envelope = json.loads((self.directory / "run.json").read_text())
            definition = RunDefinition.model_validate(envelope["definition"])
            if fingerprint(definition) != envelope["sha256"]:
                raise ValueError("run definition checksum mismatch")
            return definition
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(f"cannot resume {self.directory}: {exc}") from exc

    def events(self) -> list[dict[str, Any]]:
        path = self.directory / "experiments.jsonl"
        if not path.exists():
            return []
        previous = "0" * 64
        events: list[dict[str, Any]] = []
        try:
            for sequence, line in enumerate(path.read_text().splitlines(), 1):
                event = json.loads(line)
                digest = event.pop("sha256")
                if event["sequence"] != sequence or event["previous"] != previous:
                    raise ValueError(f"broken event chain at record {sequence}")
                if fingerprint(event) != digest:
                    raise ValueError(f"event checksum mismatch at record {sequence}")
                event["sha256"] = digest
                previous = digest
                events.append(event)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(f"experiment history is corrupt: {exc}") from exc
        return events

    def append(self, kind: str, payload: dict[str, Any]) -> None:
        self._require_lock()
        events = self.events()
        event = {
            "sequence": len(events) + 1,
            "previous": events[-1]["sha256"] if events else "0" * 64,
            "at": utc_now().isoformat(),
            "kind": kind,
            "payload": payload,
        }
        event["sha256"] = fingerprint(event)
        events.append(event)
        data = "".join(canonical_json(e) + "\n" for e in events)
        atomic_write(self.directory / "experiments.jsonl", data.encode())

    def begin(self, proposal: ExperimentProposal) -> int:
        starts = [e for e in self.events() if e["kind"] == "experiment_started"]
        number = max((int(e["payload"]["id"]) for e in starts), default=0) + 1
        self.append(
            "experiment_started", {"id": number, "proposal": proposal.model_dump(mode="json")}
        )
        self.experiment_dir(number).mkdir(parents=True, exist_ok=True)
        return number

    def experiment_dir(self, number: int) -> Path:
        if number < 1:
            raise ValueError("experiment id must be positive")
        return self.directory / "experiments" / f"experiment_{number:04d}"

    def finish(self, result: ExperimentResult) -> None:
        self._require_lock()
        events = self.events()
        starts = [
            e
            for e in events
            if e["kind"] == "experiment_started" and e["payload"]["id"] == result.id
        ]
        if len(starts) != 1:
            raise ConfigurationError(f"experiment {result.id} was not started")
        if any(
            e["kind"] == "experiment_finished" and e["payload"]["id"] == result.id for e in events
        ):
            raise ConfigurationError(f"experiment {result.id} already has an immutable result")
        if fingerprint(starts[0]["payload"]["proposal"]) != fingerprint(result.proposal):
            raise ConfigurationError("experiment proposal changed after it was started")
        data = canonical_json(result).encode()
        path = self.experiment_dir(result.id) / "result.json"
        if path.exists() and path.read_bytes() != data:
            raise ConfigurationError(f"refusing to overwrite experiment {result.id}")
        atomic_write(path, data)
        self.append(
            "experiment_finished", {"id": result.id, "sha256": hashlib.sha256(data).hexdigest()}
        )

    def results(self) -> list[ExperimentResult]:
        results = []
        for event in self.events():
            if event["kind"] != "experiment_finished":
                continue
            payload = event["payload"]
            try:
                data = (self.experiment_dir(payload["id"]) / "result.json").read_bytes()
                if hashlib.sha256(data).hexdigest() != payload["sha256"]:
                    raise ValueError("result checksum mismatch")
                result = ExperimentResult.model_validate_json(data)
                if result.id != payload["id"]:
                    raise ValueError("result id mismatch")
                results.append(result)
            except (OSError, ValueError, KeyError) as exc:
                raise ConfigurationError(f"invalid experiment result: {exc}") from exc
        return results

    def pending(self) -> list[dict[str, Any]]:
        events = self.events()
        completed = {e["payload"]["id"] for e in events if e["kind"] == "experiment_finished"}
        return [
            e
            for e in events
            if e["kind"] == "experiment_started" and e["payload"]["id"] not in completed
        ]

    def artifact(self, relative: str, data: bytes) -> str:
        self._require_lock()
        path = self.directory / relative
        if path.resolve() == self.directory or not path.resolve().is_relative_to(self.directory):
            raise ConfigurationError("artifact path escapes run directory")
        if path.exists() and path.read_bytes() != data:
            raise ConfigurationError(f"refusing to overwrite historical artifact {relative}")
        atomic_write(path, data)
        return hashlib.sha256(data).hexdigest()
