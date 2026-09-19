"""Bounded TCP and NCCL preflight on user-supplied nodes."""

from __future__ import annotations

import asyncio
import json
import shlex
from typing import Any

from servepilot.cluster.container_launcher import ContainerLauncher
from servepilot.cluster.containers import ContainerManager
from servepilot.cluster.ssh_launcher import SSHProcessHandle
from servepilot.engines.base import LaunchSpec
from servepilot.exceptions import LaunchError
from servepilot.schemas.hardware import HardwareSnapshot

SERVER = """import socket,json
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('0.0.0.0',0)); s.listen(32)
print(json.dumps({'port':s.getsockname()[1]}),flush=True)
while True:
 c,a=s.accept(); c.settimeout(30)
 with c:
  count=0
  while True:
   data=c.recv(1048576)
   if not data: break
   count+=len(data)
  c.sendall(str(count).encode())
"""
CLIENT = """import socket,time,json,sys
t=time.perf_counter(); s=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=15); latency=(time.perf_counter()-t)*1000
data=b'x'*1048576; t=time.perf_counter()
for _ in range(32): s.sendall(data)
s.shutdown(socket.SHUT_WR); count=int(s.recv(100)); elapsed=time.perf_counter()-t; s.close()
assert count==32*1048576
print(json.dumps({'connect_ms':latency,'bytes':count,'seconds':elapsed,'gigabits_per_second':count*8/elapsed/1e9}))
"""
NCCL = """import torch,torch.distributed as d,datetime,time,json,sys
rank=int(sys.argv[1]); world=int(sys.argv[2]); torch.cuda.set_device(0)
d.init_process_group('nccl',init_method=sys.argv[3],rank=rank,world_size=world,timeout=datetime.timedelta(seconds=90))
x=torch.ones(1048576,device='cuda'); d.all_reduce(x); torch.cuda.synchronize(); assert x[0].item()==world
d.barrier(); t=time.perf_counter()
for _ in range(10): x.fill_(1); d.all_reduce(x)
torch.cuda.synchronize(); elapsed=time.perf_counter()-t; assert x[0].item()==world
print(json.dumps({'rank':rank,'world':world,'all_reduce_seconds':elapsed/10,'payload_bytes':x.numel()*x.element_size(),'nccl':torch.cuda.nccl.version()}),flush=True)
d.destroy_process_group()
"""


async def check_network(manager: ContainerManager, hardware: HardwareSnapshot) -> dict[str, Any]:
    listeners: list[SSHProcessHandle] = []
    pairs = []
    try:
        for target in manager.inventory.nodes:
            spec = LaunchSpec(
                executable=target.python,
                args=["-u", "-c", SERVER],
                env={},
                host=target.network_address,
                port=0,
                gpu_ids=[],
                node_id=target.node_id,
                redacted_display_command="TCP network preflight listener",
            )
            listener = SSHProcessHandle(spec, manager.transport)
            listeners.append(listener)
            await listener.start()
            async with asyncio.timeout(15):
                while not listener.stdout_tail().strip():
                    if not listener.is_running():
                        raise LaunchError(f"network listener failed on {target.host}")
                    await asyncio.sleep(0.05)
            port = json.loads(listener.stdout_tail().splitlines()[0])["port"]
            # Confirm that the CPU controller can reach each worker's inference network.
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(target.network_address, port), timeout=15
            )
            writer.write_eof()
            await asyncio.wait_for(reader.read(100), timeout=15)
            writer.close()
            await writer.wait_closed()
            for source in manager.inventory.nodes:
                if source == target:
                    continue
                result = await manager.transport.run(
                    source,
                    [source.python, "-c", CLIENT, target.network_address, str(port)],
                    timeout=45,
                )
                pairs.append(
                    {
                        "source": source.node_id,
                        "target": target.node_id,
                        **json.loads(result.stdout),
                    }
                )
            await listener.terminate()
        return {
            "tcp": pairs,
            "controller_connectivity": "passed",
            "nodes": len(manager.inventory.nodes),
        }
    except Exception as exc:
        raise LaunchError(f"cluster network preflight failed: {exc}") from exc
    finally:
        await asyncio.gather(*(listener.terminate() for listener in listeners))


async def check_nccl(
    manager: ContainerManager, hardware: HardwareSnapshot, model_volume: str
) -> list[dict[str, Any]]:
    if len(manager.inventory.nodes) < 2:
        return []
    launcher = ContainerLauncher(manager, hardware, model_volume=model_volume)
    launcher.runtime = {
        "images": {
            node.node_id: next(iter(manager.images.values())) for node in manager.inventory.nodes
        }
    }
    head = manager.inventory.nodes[0]
    port = await launcher._port(head)
    handles = []
    try:
        for rank, node in enumerate(manager.inventory.nodes):
            gpu = next(g for g in hardware.gpus if g.node_id == node.node_id)
            command = [
                "python3",
                "-c",
                NCCL,
                str(rank),
                str(len(manager.inventory.nodes)),
                f"tcp://{head.network_address}:{port}",
            ]
            name = manager.container_name("nccl")
            spec = LaunchSpec(
                executable="python3",
                args=[],
                env={},
                host=node.network_address,
                port=0,
                gpu_ids=[gpu.index],
                node_id=node.node_id,
                redacted_display_command=shlex.join(command),
            )
            handle = await manager.start_guarded(
                launcher.docker_run(spec, node, [gpu.index], name, command), node, name
            )
            handles.append(handle)
        async with asyncio.timeout(120):
            await asyncio.gather(*(handle.wait() for handle in handles))
        results = []
        for node, handle in zip(manager.inventory.nodes, handles, strict=True):
            if handle.returncode:
                raise LaunchError(
                    f"NCCL failed on {node.host}", hints=[handle.stderr_tail()[-4000:]]
                )
            lines = [
                line for line in handle.stdout_tail().splitlines() if line.startswith('{"rank"')
            ]
            if not lines:
                raise LaunchError(f"NCCL produced no completed collective on {node.host}")
            results.append({"node": node.node_id, **json.loads(lines[-1])})
        return results
    finally:
        await manager.cleanup()
