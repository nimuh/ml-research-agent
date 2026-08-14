"""Span tracing over agent turns, tool calls and runs; renders a timeline of
what the team actually did."""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..types import utcnow
from .logging import StructuredLogger, active_redact_keys, redact


@dataclass
class Span:
    """One timed unit of work. Nesting is expressed by ``parent_id``."""

    id: str
    name: str
    parent_id: str | None = None
    started_at: datetime = field(default_factory=utcnow)
    duration_seconds: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    depth: int = 0

    def set(self, **attributes: Any) -> None:
        """Attach attributes discovered mid-span (token counts, hit/miss, ids)."""
        self.attributes.update({k: v for k, v in attributes.items() if v is not None})

    def safe_attributes(self) -> dict[str, Any]:
        """Attributes with secrets masked.

        Traces leave the process (exported to disk, rendered into a gate
        packet), so PLAN §6's "redacted in traces" has to be enforced at the
        egress points, not left to whoever calls ``span.set``.
        """
        masked = redact(dict(self.attributes), active_redact_keys())
        return masked if isinstance(masked, dict) else dict(self.attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "started_at": self.started_at.isoformat(),
            "duration_seconds": round(self.duration_seconds, 6),
            "status": self.status,
            "depth": self.depth,
            "attributes": self.safe_attributes(),
        }


class Tracer:
    """Collects spans in memory and can render them as a timeline.

    Deliberately not OpenTelemetry: the requirement is that a finished run can
    be explained offline from the workspace, which a self-contained span list
    plus the JSONL log satisfies without a collector.
    """

    def __init__(self, enabled: bool = True, logger: StructuredLogger | None = None) -> None:
        self.enabled = enabled
        self.logger = logger
        self.spans: list[Span] = []
        self._stack: list[Span] = []

    @property
    def current(self) -> Span | None:
        return self._stack[-1] if self._stack else None

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = self.current
        span = Span(
            id=f"span_{uuid.uuid4().hex[:12]}",
            name=name,
            parent_id=parent.id if parent else None,
            attributes={k: v for k, v in attributes.items() if v is not None},
            depth=len(self._stack),
        )
        if not self.enabled:
            # Still yield a usable span so callers never branch on tracing state.
            yield span
            return
        self.spans.append(span)
        self._stack.append(span)
        started = time.perf_counter()
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.set(error=type(exc).__name__, error_message=str(exc)[:500])
            raise
        else:
            span.status = "ok"
        finally:
            span.duration_seconds = time.perf_counter() - started
            self._stack.pop()
            if self.logger is not None:
                self.logger.debug(
                    "span",
                    span=span.name,
                    span_id=span.id,
                    parent_id=span.parent_id,
                    status=span.status,
                    duration_seconds=round(span.duration_seconds, 4),
                    **span.attributes,
                )

    def export(self) -> list[dict[str, Any]]:
        return [span.to_dict() for span in self.spans]

    def render_timeline(self) -> str:
        """Indented, chronological view -- what ran, how long, and what failed."""
        if not self.spans:
            return "(no spans recorded)"
        lines: list[str] = []
        for span in self.spans:
            marker = {"ok": "+", "error": "!", "running": "~"}.get(span.status, "?")
            detail = " ".join(f"{k}={v}" for k, v in span.safe_attributes().items())
            indent = "  " * span.depth
            lines.append(
                f"{marker} {indent}{span.name} ({span.duration_seconds:.3f}s)"
                + (f"  {detail}" if detail else "")
            )
        total = sum(s.duration_seconds for s in self.spans if s.depth == 0)
        lines.append(f"= total {total:.3f}s across {len(self.spans)} spans")
        return "\n".join(lines)

    def reset(self) -> None:
        self.spans.clear()
        self._stack.clear()


__all__ = ["Span", "Tracer"]
