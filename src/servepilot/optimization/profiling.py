"""Bounded summaries of diagnostic traces. Engine output never determines acceptance."""

from __future__ import annotations

import gzip
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_archive(path: Path) -> dict[str, Any]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    files, ignored = [], []
    limit = 64 * 1024 * 1024
    # Never extract an engine-authored archive onto the controller filesystem.
    with tarfile.open(path, "r|*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith((".json", ".json.gz")):
                continue
            if member.size > limit:
                ignored.append(member.name)
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            try:
                data = (
                    gzip.GzipFile(fileobj=stream).read(limit + 1)
                    if member.name.endswith(".gz")
                    else stream.read(limit + 1)
                )
                if len(data) > limit:
                    ignored.append(member.name)
                    continue
                trace = json.loads(data)
                for event in trace.get("traceEvents", []):
                    if event.get("ph") != "X" or not any(
                        part in str(event.get("cat", "")).lower()
                        for part in ("kernel", "cuda", "gpu")
                    ):
                        continue
                    duration = float(event.get("dur", 0))
                    if math.isfinite(duration) and duration > 0:
                        name = str(event.get("name", "unknown"))[:300]
                        totals[name] += duration
                        counts[name] += 1
                files.append(member.name)
            except (ValueError, OSError, TypeError, AttributeError):
                ignored.append(member.name)
    return {
        "diagnostic_only": True,
        "trace_files": files,
        "unparsed_files": ignored,
        "top_gpu_events": [
            {"name": name, "total_microseconds": total, "calls": counts[name]}
            for name, total in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:30]
        ],
    }
