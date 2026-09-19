"""One monotonic budget shared by setup, the agent, and every experiment."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class BudgetExpired(TimeoutError):
    pass


class TimeBudget:
    def __init__(self, seconds: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("time budget must be finite and positive")
        self.seconds = seconds
        self._clock = clock
        self._start = clock()
        self._deadline = self._start + seconds

    @property
    def elapsed(self) -> float:
        return max(0.0, self._clock() - self._start)

    @property
    def remaining(self) -> float:
        return max(0.0, self._deadline - self._clock())

    @property
    def expired(self) -> bool:
        return self.remaining <= 0

    async def run(self, operation: Callable[[], Awaitable[T]], *, limit: float | None = None) -> T:
        remaining = self.remaining
        if remaining <= 0:
            raise BudgetExpired("optimization time budget expired")
        seconds = min(remaining, limit) if limit is not None else remaining
        timer = asyncio.timeout(seconds)
        try:
            async with timer:
                return await operation()
        except TimeoutError as exc:
            if timer.expired() and self.expired:
                raise BudgetExpired("optimization time budget expired") from exc
            raise
