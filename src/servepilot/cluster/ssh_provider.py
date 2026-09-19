"""Cluster inventory through ordinary SSH; the controller itself can have zero GPUs."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from servepilot.cluster.inventory import NodeInventory, SSHNode
from servepilot.cluster.ray_provider import merge_node_snapshots
from servepilot.cluster.transport import SSHTransport
from servepilot.exceptions import HardwareError
from servepilot.logging import redact_secrets
from servepilot.schemas.hardware import GPUSample, HardwareSnapshot, NodeInfo


class SSHHardwareProvider:
    name = "ssh"

    def __init__(self, inventory: NodeInventory) -> None:
        self.inventory = inventory
        self.transport = SSHTransport(inventory)
        self._last: HardwareSnapshot | None = None

    def _probe(self, node: SSHNode) -> dict[str, Any]:
        command = [node.python, "-c", Path(__file__).with_name("probe.py").read_text()]
        try:
            result = subprocess.run(
                self.transport.argv(node, command),
                capture_output=True,
                text=True,
                timeout=65,
                check=False,
            )
            if result.returncode:
                raise HardwareError(
                    f"{node.host}: GPU/driver inspection failed",
                    hints=[redact_secrets(result.stderr[-2000:])],
                )
            payload: dict[str, Any] = json.loads(result.stdout)
            return payload
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise HardwareError(f"{node.host}: cannot inspect GPU hardware: {exc}") from exc

    def snapshot(self) -> HardwareSnapshot:
        with ThreadPoolExecutor(max_workers=min(len(self.inventory.nodes), 16)) as pool:
            payloads = list(pool.map(self._probe, self.inventory.nodes))
        pairs = []
        for index, (node, payload) in enumerate(zip(self.inventory.nodes, payloads, strict=True)):
            snapshot = HardwareSnapshot.model_validate(payload["snapshot"])
            if snapshot.gpu_count == 0 or any(g.total_memory_bytes <= 0 for g in snapshot.gpus):
                raise HardwareError(f"{node.host}: no usable NVIDIA GPUs found")
            if snapshot.platform != "Linux":
                raise HardwareError(f"{node.host}: GPU workers must run Linux")
            info = NodeInfo(
                node_id=node.node_id,
                node_ip=node.network_address,
                hostname=snapshot.hostname,
                cpu_count=payload.get("cpu_count"),
                is_head=index == 0,
            )
            pairs.append((info, snapshot))
        snapshot = merge_node_snapshots(pairs, address=None)
        snapshot.provider = "ssh"
        self._last = snapshot
        return snapshot

    def sample(self, gpu_indices: Sequence[int]) -> list[GPUSample]:
        if self._last is None:
            self.snapshot()
        assert self._last is not None
        selected = [self._last.gpu(index) for index in gpu_indices]
        nodes = [
            node
            for node in self.inventory.nodes
            if any(g.node_id == node.node_id for g in selected)
        ]
        with ThreadPoolExecutor(max_workers=max(1, min(len(nodes), 16))) as pool:
            payloads = list(pool.map(self._probe, nodes))
        samples = []
        for node, payload in zip(nodes, payloads, strict=True):
            mapping = {
                g.device_index_on_node: g.index for g in selected if g.node_id == node.node_id
            }
            for raw in payload["samples"]:
                sample = GPUSample.model_validate(raw)
                if sample.index in mapping:
                    sample.index = mapping[sample.index]
                    samples.append(sample)
        return samples
