"""Bounded async fan-out, rate limiters and retry helpers used by source
adapters and agent dispatch."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from ..errors import MRAError

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class RateLimiter:
    """Token-bucket limiter shared by a source adapter across its calls.

    Public APIs (arXiv, Semantic Scholar) rate-limit aggressively; hitting them
    politely is cheaper than retrying through a 429 storm.
    """

    rate_per_second: float
    burst: int = 1
    _tokens: float = field(default=0.0, init=False)
    _updated: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)

    async def acquire(self) -> None:
        if self.rate_per_second <= 0:
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    float(self.burst), self._tokens + (now - self._updated) * self.rate_per_second
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self.rate_per_second)


async def gather_bounded(
    tasks: Sequence[Callable[[], Awaitable[T]]],
    *,
    limit: int = 8,
    return_exceptions: bool = True,
) -> list[T | BaseException]:
    """Run awaitable factories with at most ``limit`` in flight.

    Factories (not coroutines) so nothing starts before its slot is free.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    semaphore = asyncio.Semaphore(limit)

    async def _run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    return await asyncio.gather(
        *(_run(factory) for factory in tasks), return_exceptions=return_exceptions
    )


async def map_bounded(
    items: Iterable[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    limit: int = 8,
) -> list[R | BaseException]:
    """``gather_bounded`` over a collection with a single async function."""
    materialized = list(items)
    return await gather_bounded([lambda item=item: fn(item) for item in materialized], limit=limit)  # type: ignore[misc]


def backoff_delays(
    attempts: int,
    *,
    base: float = 0.5,
    factor: float = 2.0,
    cap: float = 30.0,
    jitter: float = 0.25,
) -> list[float]:
    """Exponential backoff schedule with proportional jitter."""
    delays: list[float] = []
    for attempt in range(attempts):
        raw = min(cap, base * (factor**attempt))
        delays.append(raw * (1.0 + random.uniform(-jitter, jitter)))
    return delays


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base: float = 0.5,
    cap: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Retry an awaitable factory with backoff, honouring ``MRAError.retryable``."""
    delays = backoff_delays(attempts, base=base, cap=cap)
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except retry_on as exc:  # noqa: PERF203 - retry loop is the point
            if isinstance(exc, MRAError) and not exc.retryable:
                raise
            last = exc
            if attempt == attempts - 1:
                break
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            await asyncio.sleep(delays[attempt])
    assert last is not None
    raise last


def retry_sync(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base: float = 0.5,
    cap: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    """Blocking counterpart of :func:`retry_async`."""
    delays = backoff_delays(attempts, base=base, cap=cap)
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            if isinstance(exc, MRAError) and not exc.retryable:
                raise
            last = exc
            if attempt == attempts - 1:
                break
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            time.sleep(delays[attempt])
    assert last is not None
    raise last


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, refusing to nest inside a live loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    raise RuntimeError("run_sync called from inside a running event loop")


__all__ = [
    "RateLimiter",
    "backoff_delays",
    "gather_bounded",
    "map_bounded",
    "retry_async",
    "retry_sync",
    "run_sync",
]
