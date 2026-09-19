"""Standalone, stdlib-only NVIDIA inspection executed on existing nodes over SSH."""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import re
import socket
import subprocess
import time
from typing import Any
from xml.etree import ElementTree


def numeric(value: str | None, default: float | None = None) -> float | None:
    match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", value or "")
    return float(match[0]) if match else default


def topology(text: str, indices: list[int]) -> dict[str, Any]:
    lines = [line.split() for line in text.splitlines()]
    header = next(
        (
            line
            for line in lines
            if line and all(re.fullmatch(r"GPU\d+", x) for x in line[: len(indices)])
        ),
        [],
    )
    labels = [int(label[3:]) for label in header if re.fullmatch(r"GPU\d+", label)]
    edges = []
    if len(indices) == 1:
        return {"available": True, "edges": []}
    if not labels:
        return {"available": False, "edges": [], "error": "nvidia-smi topology unavailable"}
    for row in lines:
        if not row or not re.fullmatch(r"GPU\d+", row[0]):
            continue
        source = int(row[0][3:])
        for target, relation in zip(labels, row[1:], strict=False):
            if source not in indices or target not in indices or source >= target:
                continue
            nvlink = bool(re.fullmatch(r"NV\d+", relation))
            edges.append(
                {
                    "gpu_a": source,
                    "gpu_b": target,
                    "relationship": "NV" if nvlink else relation,
                    "nvlink_detected": nvlink,
                    "nvlink_link_count": int(relation[2:]) if nvlink else None,
                }
            )
    return {"available": bool(edges), "edges": edges}


def parse(xml: str, query: str, topology_text: str) -> dict[str, Any]:
    root = ElementTree.fromstring(xml)
    identities = list(csv.reader(io.StringIO(query)))
    by_uuid = {row[1].strip(): row for row in identities if len(row) >= 3}
    devices: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for fallback_index, gpu in enumerate(root.findall("gpu")):
        uuid = gpu.findtext("uuid") or ""
        row = by_uuid.get(uuid)
        index = int(row[0]) if row else fallback_index
        capability = row[2].strip().split(".") if row else []
        total = int((numeric(gpu.findtext("fb_memory_usage/total"), 0) or 0) * 1024**2)
        free = int((numeric(gpu.findtext("fb_memory_usage/free"), 0) or 0) * 1024**2)
        used = int((numeric(gpu.findtext("fb_memory_usage/used"), 0) or 0) * 1024**2)
        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": gpu.findtext("product_name") or "unknown NVIDIA GPU",
                "total_memory_bytes": total,
                "free_memory_bytes": free,
                "used_memory_bytes": used,
                "pci_bus_id": gpu.findtext("pci/pci_bus_id"),
                "mig_mode": gpu.findtext("mig_mode/current_mig"),
                "compute_capability_major": int(capability[0])
                if capability and capability[0].isdigit()
                else None,
                "compute_capability_minor": int(capability[1])
                if len(capability) > 1 and capability[1].isdigit()
                else None,
            }
        )
        samples.append(
            {
                "index": index,
                "timestamp": time.time(),
                "utilization_percent": numeric(gpu.findtext("utilization/gpu_util")),
                "memory_used_bytes": used,
                "memory_total_bytes": total,
                "temperature_c": numeric(gpu.findtext("temperature/gpu_temp")),
                "power_watts": numeric(
                    gpu.findtext("gpu_power_readings/power_draw")
                    or gpu.findtext("power_readings/power_draw")
                ),
            }
        )
    return {
        "snapshot": {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "gpu_count": len(devices),
            "gpus": devices,
            "topology": topology(topology_text, [g["index"] for g in devices]),
            "driver_version": root.findtext("driver_version"),
            "cuda_version": root.findtext("cuda_version"),
            "provider": "ssh",
        },
        "samples": samples,
        "cpu_count": os.cpu_count(),
    }


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("GPU workers must run Linux")
    xml = subprocess.check_output(["nvidia-smi", "-q", "-x"], text=True, timeout=20)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,compute_cap", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    topo = subprocess.run(
        ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=20, check=False
    )
    print(json.dumps(parse(xml, query.stdout, topo.stdout)))


if __name__ == "__main__":
    main()
