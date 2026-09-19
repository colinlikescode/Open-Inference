"""Engine-authored trace archives are read as data, never extracted or trusted as scores."""

import io
import json
import tarfile
from pathlib import Path

from servepilot.optimization.profiling import summarize_archive


def test_trace_summary_does_not_extract_paths_and_ignores_non_gpu_events(tmp_path: Path) -> None:
    archive = tmp_path / "traces.tar"
    trace = {
        "traceEvents": [
            {"name": "attention", "cat": "kernel", "ph": "X", "dur": 200},
            {"name": "attention", "cat": "kernel", "ph": "X", "dur": 100},
            {"name": "python", "cat": "cpu_op", "ph": "X", "dur": 99999},
        ]
    }
    data = json.dumps(trace).encode()
    with tarfile.open(archive, "w") as stream:
        member = tarfile.TarInfo("../../escaped.json")
        member.size = len(data)
        stream.addfile(member, io.BytesIO(data))
    summary = summarize_archive(archive)
    assert summary["diagnostic_only"]
    assert summary["top_gpu_events"] == [
        {"name": "attention", "total_microseconds": 300, "calls": 2}
    ]
    assert list(tmp_path.iterdir()) == [archive]
