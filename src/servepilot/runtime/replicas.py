"""Launch and track every replica of one plan.

Shared by tuning sessions (temporary) and production deployments. Startup is controlled: the
first replica starts alone so the model is downloaded/cached once, remaining replicas start
staggered. Any startup failure tears the whole set down and surfaces a classified failure.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from servepilot.constants import (
    DEFAULT_GRACEFUL_SHUTDOWN_SECONDS,
    DEFAULT_STARTUP_STAGGER_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
)
from servepilot.engines.base import InferenceEngine, LaunchSpec
from servepilot.engines.process import Launcher, ProcessHandle, verify_all_exited
from servepilot.exceptions import LaunchError
from servepilot.logging import get_logger
from servepilot.runtime.ports import PortAllocator
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.benchmark import CandidateFailure, FailureType
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import CandidatePlan
from servepilot.schemas.runtime import ReplicaStatus

log = get_logger(__name__)


class ReplicaLaunchError(LaunchError):
    def __init__(self, failure: CandidateFailure, replica_id: str) -> None:
        super().__init__(f"{replica_id}: {failure.message}")
        self.failure = failure
        self.replica_id = replica_id


@dataclass
class Replica:
    index: int
    spec: LaunchSpec
    process: ProcessHandle | None = None
    ready: bool = False
    launch_seconds: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.spec.replica_id

    @property
    def base_url(self) -> str:
        return self.spec.base_url


class ReplicaSet:
    def __init__(
        self,
        *,
        plan: CandidatePlan,
        model: ModelProfile,
        engine: InferenceEngine,
        launcher: Launcher,
        ports: PortAllocator,
        hardware: HardwareSnapshot,
        host: str = "127.0.0.1",
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        stagger_seconds: float = DEFAULT_STARTUP_STAGGER_SECONDS,
        router: ReplicaRouter | None = None,
    ) -> None:
        self.plan = plan
        self.model = model
        self.engine = engine
        self.launcher = launcher
        self.ports = ports
        self.hardware = hardware
        self.host = host
        self.served_model_name = served_model_name
        self.trust_remote_code = trust_remote_code
        self.startup_timeout = startup_timeout
        self.stagger_seconds = stagger_seconds
        self.router = router
        self.replicas: list[Replica] = []
        self._allocated_ports: list[int] = []

    # ------------------------------------------------------------------ specs
    def _node_for(self, gpu_ids: list[int]) -> tuple[tuple[str, str] | None, list[int]]:
        gpus = [self.hardware.gpu(i) for i in gpu_ids]
        node: tuple[str, str] | None = None
        first = gpus[0]
        if first.node_id is not None and first.node_ip is not None:
            node = (first.node_id, first.node_ip)
        local_ids = [g.device_index_on_node for g in gpus]
        return node, local_ids

    def build_spec(self, replica_index: int, port: int) -> LaunchSpec:
        node, local_ids = self._node_for(self.plan.gpu_groups[replica_index])
        bind_host = self.host
        if node is not None and node[1] not in ("127.0.0.1", "localhost"):
            # Remote replicas must listen on an address reachable from the router.
            bind_host = "0.0.0.0"
        return self.engine.build_launch_spec(
            self.model,
            self.plan,
            replica_index=replica_index,
            host=bind_host,
            port=port,
            served_model_name=self.served_model_name,
            trust_remote_code=self.trust_remote_code,
            node=node,
            local_gpu_ids=local_ids,
        )

    def specs(self) -> list[LaunchSpec]:
        """Launch specs without reserving ports (for ``--dry-run``)."""
        base = self.ports.allocate()
        self.ports.release(base)
        return [self.build_spec(i, base + i) for i in range(self.plan.replica_count)]

    # ------------------------------------------------------------------ lifecycle
    async def _launch_one(self, index: int) -> Replica:
        port = self.ports.allocate()
        self._allocated_ports.append(port)
        spec = self.build_spec(index, port)
        replica = Replica(index=index, spec=spec)
        start = time.monotonic()
        replica.process = await self.launcher.launch(spec)
        if self.router is not None:
            state = self.router.add_replica(
                replica.id, replica.base_url, gpu_ids=spec.gpu_ids, pid=replica.process.pid
            )
            state.status = ReplicaStatus.STARTING
        try:
            readiness = await self.engine.wait_until_ready(
                spec, replica.process, self.startup_timeout
            )
        except BaseException:
            await replica.process.terminate()
            raise
        if not readiness.ready:
            failure = readiness.failure or CandidateFailure(
                type=FailureType.UNKNOWN, message="not ready"
            )
            if replica.process is not None:
                await replica.process.terminate()
            if self.router is not None:
                self.router.set_status(replica.id, ReplicaStatus.STOPPED, failure.message)
            raise ReplicaLaunchError(failure, replica.id)
        replica.ready = True
        replica.launch_seconds = time.monotonic() - start
        replica.metadata = dict(readiness.metadata)
        if self.router is not None:
            self.router.set_status(replica.id, ReplicaStatus.HEALTHY)
        log.info("%s ready in %.1fs at %s", replica.id, replica.launch_seconds, replica.base_url)
        return replica

    async def start(self) -> None:
        """Start all replicas; on any failure stop everything and raise :class:`ReplicaLaunchError`."""
        try:
            first = await self._launch_one(0)
            self.replicas.append(first)
            if self.plan.replica_count > 1:
                tasks: list[asyncio.Task[Replica]] = []
                for index in range(1, self.plan.replica_count):
                    if self.stagger_seconds > 0 and index > 1:
                        await asyncio.sleep(self.stagger_seconds)
                    tasks.append(asyncio.create_task(self._launch_one(index)))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                errors = [r for r in results if isinstance(r, BaseException)]
                self.replicas.extend(r for r in results if isinstance(r, Replica))
                if errors:
                    raise errors[0]
        except BaseException:
            await self.stop()
            raise

    async def stop(self, grace_seconds: float = DEFAULT_GRACEFUL_SHUTDOWN_SECONDS) -> bool:
        """Terminate every replica; returns True when all processes are confirmed gone."""
        handles = [r.process for r in self.replicas if r.process is not None]
        await asyncio.gather(*(h.terminate(grace_seconds) for h in handles), return_exceptions=True)
        still_alive = await verify_all_exited(handles, timeout=15.0)
        for r in self.replicas:
            if self.router is not None:
                self.router.set_status(r.id, ReplicaStatus.STOPPED)
                self.router.remove_replica(r.id)
        for port in self._allocated_ports:
            self.ports.release(port)
        self._allocated_ports = []
        if still_alive:
            log.error("%d replica process(es) are still alive after termination", len(still_alive))
            return False
        return True

    async def restart_replica(self, replica: Replica) -> Replica:
        """Replace a dead replica in place (same index/port).

        When the restart fails, ``replica`` stays in :attr:`replicas` (its process already
        terminated) with an ``UNHEALTHY`` router entry, so the health checker sees it as dead
        again and may retry while its restart budget lasts; only the checker moves a replica to
        ``STOPPED``.
        """
        if replica.process is not None:
            await replica.process.terminate()
        restarts = 1
        if self.router is not None:
            previous = self.router.replica(replica.id)
            if previous is not None:
                restarts = previous.restarts + 1
            self.router.remove_replica(replica.id)
        spec = self.build_spec(replica.index, replica.spec.port)
        fresh = Replica(index=replica.index, spec=spec)
        start = time.monotonic()
        failure: CandidateFailure | None = None
        try:
            fresh.process = await self.launcher.launch(spec)
        except LaunchError as exc:
            failure = CandidateFailure(
                type=FailureType.ENGINE_CRASH, message=exc.message, stage="restart"
            )
        if self.router is not None:
            state = self.router.add_replica(
                fresh.id,
                fresh.base_url,
                gpu_ids=spec.gpu_ids,
                pid=fresh.process.pid if fresh.process is not None else None,
            )
            state.status = ReplicaStatus.STARTING
            state.restarts = restarts
        if fresh.process is not None:
            readiness = await self.engine.wait_until_ready(
                spec, fresh.process, self.startup_timeout
            )
            if readiness.ready:
                fresh.metadata = dict(readiness.metadata)
            else:
                failure = readiness.failure or CandidateFailure(message="not ready")
                await fresh.process.terminate()
        if failure is not None:
            if self.router is not None:
                self.router.set_status(fresh.id, ReplicaStatus.UNHEALTHY, failure.message)
            raise ReplicaLaunchError(failure, fresh.id)
        fresh.ready = True
        fresh.launch_seconds = time.monotonic() - start
        if self.router is not None:
            self.router.set_status(fresh.id, ReplicaStatus.HEALTHY)
        self.replicas[self.replicas.index(replica)] = fresh
        return fresh

    @property
    def processes(self) -> list[ProcessHandle]:
        return [r.process for r in self.replicas if r.process is not None]

    @property
    def ports_in_use(self) -> list[int]:
        return [r.spec.port for r in self.replicas]

    def engine_version(self) -> str | None:
        return self.engine.version()
