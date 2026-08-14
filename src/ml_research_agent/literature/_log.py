"""Logger adapter used across ``literature/``.

The platform's structured logger is being written concurrently, so this layer
accepts *any* logger object and degrades to a stdlib one. Every truncation and
every degraded source in this layer goes through here -- bounded fan-out is
only honest if the truncation is visible.
"""

from __future__ import annotations

import logging
from typing import Any

_FALLBACK = logging.getLogger("ml_research_agent.literature")


def emit(logger: Any | None, level: str, event: str, /, **fields: Any) -> None:
    """Log ``event`` with structured ``fields`` against whatever logger we got.

    Structured loggers take keyword fields; stdlib ones do not, so the fields
    are folded into the message on ``TypeError`` rather than losing the record.
    """
    target = logger if logger is not None else _FALLBACK
    method = getattr(target, level, None) or getattr(target, "info", None)
    if method is None:  # pragma: no cover - not a logger at all
        return
    try:
        method(event, **fields)
    except TypeError:
        detail = " ".join(f"{k}={v!r}" for k, v in sorted(fields.items()))
        method(f"{event} {detail}".strip())


def truncated(
    logger: Any | None, *, stage: str, kept: int, dropped: int, cap: int, **fields: Any
) -> None:
    """Record that a cap fired. Never call the cap without calling this."""
    if dropped <= 0:
        return
    emit(
        logger,
        "warning",
        "literature.truncated",
        stage=stage,
        kept=kept,
        dropped=dropped,
        cap=cap,
        **fields,
    )


__all__ = ["emit", "truncated"]
