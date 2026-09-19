"""Periodic replica health checking with bounded restarts."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Awaitable, Callable

import httpx

from servepilot.constants import (
    DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS,
    DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS,
    DEFAULT_REPLICA_MAX_RESTARTS,
    DEFAULT_REPLICA_RESTART_BACKOFF_SECONDS,
    DEFAULT_REPLICA_RESTART_WINDOW_SECONDS,
    DEFAULT_UNHEALTHY_AFTER_FAILURES,
)
from servepilot.logging import get_logger
from servepilot.runtime.replicas import Replica, ReplicaSet
from servepilot.runtime.router import ReplicaRouter
from servepilot.schemas.runtime import ReplicaStatus

log = get_logger(__name__)

RestartHook = Callable[[Replica], Awaitable[Replica]]


class HealthChecker:
    """Marks replicas unhealthy after consecutive probe failures; restarts dead ones with backoff.

    Restart attempts are bounded per replica within a rolling window so a crash-looping engine
    cannot destabilise the router; other replicas keep serving throughout.
    """

    def __init__(
        self,
        router: ReplicaRouter,
        replica_set: ReplicaSet,
        *,
        interval: float = DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS,
        timeout: float = DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS,
        unhealthy_after: int = DEFAULT_UNHEALTHY_AFTER_FAILURES,
        max_restarts: int = DEFAULT_REPLICA_MAX_RESTARTS,
        restart_window: float = DEFAULT_REPLICA_RESTART_WINDOW_SECONDS,
        restart_backoff: float = DEFAULT_REPLICA_RESTART_BACKOFF_SECONDS,
        restart_enabled: bool = True,
        on_restart: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._router = router
        self._set = replica_set
        self._interval = interval
        self._timeout = timeout
        self._unhealthy_after = unhealthy_after
        self._max_restarts = max_restarts
        self._window = restart_window
        self._backoff = restart_backoff
        self._restart_enabled = restart_enabled
        self._on_restart = on_restart
        self._task: asyncio.Task[None] | None = None
        self._restart_times: dict[int, deque[float]] = {}
        self._restarting: set[int] = set()
        self._restart_tasks: set[asyncio.Task[None]] = set()
        self.events: list[str] = []

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Stop probing and abandon pending restarts so none launches an engine mid-shutdown."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        pending = list(self._restart_tasks)
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._restarting.clear()  # a task cancelled before it ever ran skips its finally

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while True:
                await self.check_once(client)
                await asyncio.sleep(self._interval)

    async def check_once(self, client: httpx.AsyncClient) -> None:
        live: list[Replica] = []
        for replica in list(self._set.replicas):
            if replica.index in self._restarting:
                continue
            state = self._router.replica(replica.id)
            if state is None or state.status == ReplicaStatus.STOPPED:
                continue
            if replica.process is not None and not replica.process.is_running():
                await self._handle_dead(
                    replica, f"process exited with code {replica.process.returncode}"
                )
                continue
            live.append(replica)
        # Probes are independent network calls; one hung replica must not delay the others.
        results = await asyncio.gather(*(self._probe(client, r) for r in live))
        for replica, ok in zip(live, results, strict=True):
            state = self._router.replica(replica.id)
            if state is None:
                continue
            if ok:
                if state.status == ReplicaStatus.UNHEALTHY:
                    self._record(f"{replica.id} recovered")
                if state.status != ReplicaStatus.HEALTHY:
                    self._router.set_status(replica.id, ReplicaStatus.HEALTHY)
                state.consecutive_health_failures = 0
            else:
                state.consecutive_health_failures += 1
                if (
                    state.consecutive_health_failures >= self._unhealthy_after
                    and state.status == ReplicaStatus.HEALTHY
                ):
                    self._router.set_status(
                        replica.id, ReplicaStatus.UNHEALTHY, "health probe failing"
                    )
                    self._record(
                        f"{replica.id} marked unhealthy after {state.consecutive_health_failures} failed probes"
                    )

    async def _probe(self, client: httpx.AsyncClient, replica: Replica) -> bool:
        for path in (replica.spec.health_path, replica.spec.readiness_path):
            try:
                resp = await client.get(replica.base_url + path)
            except (httpx.HTTPError, OSError):
                continue
            if 200 <= resp.status_code < 300:
                return True
        return False

    async def _handle_dead(self, replica: Replica, reason: str) -> None:
        self._router.set_status(replica.id, ReplicaStatus.UNHEALTHY, reason)
        self._record(f"{replica.id} died: {reason}")
        log.error("%s died: %s", replica.id, reason)
        if not self._restart_enabled:
            self._router.set_status(replica.id, ReplicaStatus.STOPPED, reason)
            return
        times = self._restart_times.setdefault(replica.index, deque())
        now = time.monotonic()
        while times and now - times[0] > self._window:
            times.popleft()
        if len(times) >= self._max_restarts:
            self._router.set_status(replica.id, ReplicaStatus.STOPPED, "restart budget exhausted")
            self._record(
                f"{replica.id} exceeded {self._max_restarts} restarts in {self._window:.0f}s; giving up"
            )
            log.error("%s exceeded restart budget; leaving it stopped", replica.id)
            return
        times.append(now)
        self._restarting.add(replica.index)
        task = asyncio.create_task(self._restart(replica, attempt=len(times)))
        self._restart_tasks.add(task)
        task.add_done_callback(self._restart_tasks.discard)

    async def _restart(self, replica: Replica, attempt: int) -> None:
        delay = self._backoff * (2 ** (attempt - 1))
        self._record(
            f"restarting {replica.id} in {delay:.0f}s (attempt {attempt}/{self._max_restarts})"
        )
        try:
            await asyncio.sleep(delay)
            fresh = await self._set.restart_replica(replica)
            if self._on_restart is not None:
                await self._on_restart()
            self._record(f"{fresh.id} restarted successfully")
        except Exception as exc:
            self._record(f"{replica.id} restart failed: {exc}")
            log.error("restart of %s failed: %s", replica.id, exc)
        finally:
            self._restarting.discard(replica.index)

    def _record(self, event: str) -> None:
        self.events.append(f"{time.strftime('%H:%M:%S')} {event}")
        del self.events[:-200]
        log.info(event)
