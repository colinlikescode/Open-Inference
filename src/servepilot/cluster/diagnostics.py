"""Read-only machine diagnostics for a CPU controller and SSH GPU workers."""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any

from servepilot.cluster.containers import ContainerConfig, ContainerManager
from servepilot.cluster.inventory import NodeInventory
from servepilot.cluster.network import check_network
from servepilot.cluster.ssh_provider import SSHHardwareProvider
from servepilot.cluster.transport import SSHTransport


async def diagnose(inventory: NodeInventory) -> dict[str, Any]:
    hardware = await asyncio.to_thread(SSHHardwareProvider(inventory).snapshot)
    manager = ContainerManager(inventory, ContainerConfig(bootstrap=False), namespace="ob-doctor")
    network = await check_network(manager, hardware)
    checks = []
    transport = SSHTransport(inventory)
    for node in inventory.nodes:
        docker = await transport.run(
            node,
            [
                "docker",
                "--host",
                "unix:///var/run/docker.sock",
                "info",
                "--format",
                "{{json .Runtimes}}",
            ],
            check=False,
        )
        if docker.returncode:
            docker = await transport.run(
                node,
                [
                    "sudo",
                    "-n",
                    "docker",
                    "--host",
                    "unix:///var/run/docker.sock",
                    "info",
                    "--format",
                    "{{json .Runtimes}}",
                ],
                check=False,
            )
        ready = docker.returncode == 0 and '"nvidia"' in docker.stdout
        checks.append(
            {
                "node": node.node_id,
                "check": "Docker and NVIDIA runtime",
                "status": "ok" if ready else "warn",
                "detail": "ready"
                if ready
                else "optimize will attempt bootstrap; requires root or passwordless sudo",
            }
        )
    checks.append(
        {
            "check": "Pi",
            "status": "ok" if shutil.which("pi") else "warn",
            "detail": "installed"
            if shutil.which("pi")
            else "install Node.js 22.19+ and @earendil-works/pi-coding-agent@0.83.0 on the controller",
        }
    )
    checks.append(
        {
            "check": "LiteLLM credentials",
            "status": "ok" if os.environ.get("LITELLM_API_KEY") else "warn",
            "detail": "LITELLM_API_KEY is set"
            if os.environ.get("LITELLM_API_KEY")
            else "set LITELLM_API_KEY or select a different api_key_env in --agent-config",
        }
    )
    return {
        "status": "ok" if all(c["status"] == "ok" for c in checks) else "warn",
        "checks": checks,
        "hardware": hardware.model_dump(mode="json"),
        "network": network,
    }
