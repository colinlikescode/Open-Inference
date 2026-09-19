"""Production deployment: replicas + health checking + public HTTP endpoint + runtime state."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from servepilot import __version__
from servepilot.api.app import ServingContext, create_app
from servepilot.constants import (
    DEFAULT_MAX_QUEUE_DEPTH,
    DEFAULT_STARTUP_STAGGER_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
)
from servepilot.engines.base import InferenceEngine
from servepilot.engines.process import Launcher
from servepilot.exceptions import LaunchError
from servepilot.hardware.base import HardwareProvider
from servepilot.logging import get_logger
from servepilot.runtime.health import HealthChecker
from servepilot.runtime.ports import PortAllocator
from servepilot.runtime.replicas import ReplicaLaunchError, ReplicaSet
from servepilot.runtime.router import ReplicaRouter
from servepilot.runtime.server import HTTPServer
from servepilot.runtime.state import RuntimeStateStore, current_process_create_time
from servepilot.schemas.hardware import HardwareSnapshot
from servepilot.schemas.model import ModelProfile
from servepilot.schemas.plan import SelectedPlan
from servepilot.schemas.runtime import ChildProcessRecord, RuntimeState

log = get_logger(__name__)


class Deployment:
    """Runs the selected plan until asked to stop (SIGINT/SIGTERM or :meth:`shutdown`)."""

    def __init__(
        self,
        *,
        selected: SelectedPlan,
        model: ModelProfile,
        engine: InferenceEngine,
        launcher: Launcher,
        ports: PortAllocator,
        hardware: HardwareSnapshot,
        hardware_provider: HardwareProvider | None,
        state_store: RuntimeStateStore,
        host: str,
        port: int,
        served_model_name: str | None = None,
        trust_remote_code: bool = False,
        max_queue_depth: int = DEFAULT_MAX_QUEUE_DEPTH,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        stagger_seconds: float = DEFAULT_STARTUP_STAGGER_SECONDS,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.selected = selected
        self.plan = selected.plan
        self.model = model
        self.engine = engine
        self.launcher = launcher
        self.ports = ports
        self.hardware = hardware
        self.hardware_provider = hardware_provider
        self.state_store = state_store
        self.host = host
        self.port = port
        self.served_model_name = served_model_name or model.model_id
        self.trust_remote_code = trust_remote_code
        self.max_queue_depth = max_queue_depth
        self.startup_timeout = startup_timeout
        self.stagger_seconds = stagger_seconds
        self._on_event = on_event or (lambda msg: None)

        self.router = ReplicaRouter(
            max_concurrency=self.plan.max_concurrency, max_queue_depth=max_queue_depth
        )
        self.replica_set = ReplicaSet(
            plan=self.plan,
            model=model,
            engine=engine,
            launcher=launcher,
            ports=ports,
            hardware=hardware,
            served_model_name=self.served_model_name,
            trust_remote_code=trust_remote_code,
            startup_timeout=startup_timeout,
            stagger_seconds=stagger_seconds,
            router=self.router,
        )
        self.health = HealthChecker(self.router, self.replica_set, on_restart=self._refresh_state)
        self.ctx = ServingContext(
            router=self.router,
            model_id=model.model_id,
            served_model_name=self.served_model_name,
            backend_model_name=self.served_model_name,
            selected=selected,
            hardware=hardware_provider,
            gpu_indices=self.plan.gpu_ids,
            extra_status=self._extra_status,
        )
        self.server = HTTPServer(create_app(self.ctx), host, port)
        self._stop_event = asyncio.Event()
        self.ready = asyncio.Event()

    # ------------------------------------------------------------------ status
    def _extra_status(self) -> dict[str, Any]:
        return {
            "servepilot_pid": os.getpid(),
            "public_endpoint": f"http://{self.host}:{self.port}/v1",
            "health_events": self.health.events[-20:],
            "backend_ports": self.replica_set.ports_in_use,
        }

    @property
    def base_url(self) -> str:
        return self.server.base_url

    # ------------------------------------------------------------------ lifecycle
    async def _refresh_state(self) -> None:
        await asyncio.to_thread(self._write_state)

    def _write_state(self) -> None:
        children = [
            ChildProcessRecord(
                pid=r.process.pid or 0,
                create_time=r.process.create_time or 0.0,
                replica_id=r.id,
                port=r.spec.port,
                gpu_ids=r.spec.gpu_ids,
            )
            for r in self.replica_set.replicas
            if r.process is not None and r.process.pid
        ]
        state = RuntimeState(
            servepilot_pid=os.getpid(),
            servepilot_create_time=current_process_create_time(),
            servepilot_version=__version__,
            model=self.model.model_id,
            served_model_name=self.served_model_name,
            public_host=self.host,
            public_port=self.port,
            backend_ports=self.replica_set.ports_in_use,
            children=children,
            started_at=datetime.now(tz=UTC),
            plan_id=self.plan.id,
            engine=self.plan.engine.value,
            tensor_parallel_size=self.plan.tensor_parallel_size,
            replica_count=self.plan.replica_count,
            gpu_groups=self.plan.gpu_groups,
        )
        self.state_store.write(state)

    def install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, self.request_shutdown, sig.name)

    def request_shutdown(self, reason: str = "requested") -> None:
        if not self._stop_event.is_set():
            self._on_event(f"shutdown requested ({reason})")
            self._stop_event.set()

    async def start(self) -> None:
        self._on_event(f"starting {self.plan.replica_count} replica(s) of {self.plan.label()}")
        try:
            await self.replica_set.start()
        except ReplicaLaunchError as exc:
            tail_lines = [line for line in exc.failure.stderr_tail.splitlines() if line.strip()][
                -12:
            ]
            detail = (
                ("\n\nLast engine log lines:\n  " + "\n  ".join(tail_lines)) if tail_lines else ""
            )
            raise LaunchError(
                f"replica {exc.replica_id} failed to start: {exc.failure.message}{detail}",
                hints=[
                    "Run with -vv to stream the full engine logs.",
                    "Use `servepilot serve MODEL --retune` if the hardware changed since tuning.",
                ],
            ) from exc
        await self.server.start()
        await self.health.start()
        await asyncio.to_thread(self._write_state)  # fsync'd write; keep it off the loop
        self.ready.set()
        self._on_event(f"serving {self.served_model_name} at {self.base_url}/v1")

    async def shutdown(self) -> None:
        self._on_event("stopping HTTP server")
        await self.health.stop()
        with contextlib.suppress(Exception):
            await self.server.stop()
        self._on_event("terminating engine replicas")
        await self.replica_set.stop()
        await self.launcher.shutdown_all()
        with contextlib.suppress(Exception):
            await self.ctx.aclose()
        await asyncio.to_thread(self.state_store.clear)
        self._on_event("shutdown complete")

    async def run(self) -> None:
        """Start, serve until a shutdown is requested, then clean up (also on errors)."""
        self.install_signal_handlers()
        try:
            await self.start()
            await self._stop_event.wait()
        finally:
            await self.shutdown()
