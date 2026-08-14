"""Cost ledger across LLM tokens, compute hours and API calls, attributed to
phase and agent."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import utcnow
from ..utils.io import append_jsonl, read_jsonl
from .logging import StructuredLogger, get_logger

# USD per 1M tokens, (input, output). Keep in sync with configs/default.yaml tiers.
# USD per million tokens, (input, output). Sonnet 5 is listed at its standard
# rate rather than the promotional one: pricing a call slightly high makes a
# budget stop early, which is the safe direction for a hard ceiling.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

FALLBACK_MODEL = "claude-sonnet-5"

_logger: StructuredLogger = get_logger("observability.cost")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Price a call. Unknown models fall back to standard-tier pricing.

    Falling back rather than raising is deliberate: a new model id must never
    stop a run, but it must never be silently free either -- an unpriced call
    would make the budget ceiling unenforceable.
    """
    prices = PRICING.get(model)
    if prices is None:
        prices = PRICING[FALLBACK_MODEL]
        _logger.warning("unknown model pricing; using standard tier", model=model)
    per_input, per_output = prices
    return (input_tokens / 1_000_000) * per_input + (output_tokens / 1_000_000) * per_output


@dataclass
class CostEntry:
    """One priced event: an LLM call, a compute hour, an API request."""

    kind: str
    model: str | None
    input_tokens: int
    output_tokens: int
    usd: float
    phase: str | None
    agent: str | None
    at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["at"] = self.at.isoformat()
        return record


class CostLedger:
    """Append-only record of everything a project spent.

    Persisted as JSONL as it happens, so a crashed run still reports what it
    burned -- a ledger that only exists in memory is not an audit trail.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.entries: list[CostEntry] = []
        if self.path is not None and self.path.exists():
            self.entries = [_entry_from_dict(r) for r in read_jsonl(self.path)]

    def record(
        self,
        *,
        kind: str,
        usd: float,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        phase: str | None = None,
        agent: str | None = None,
    ) -> CostEntry:
        entry = CostEntry(
            kind=kind,
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            usd=float(usd),
            phase=phase,
            agent=agent,
        )
        self.entries.append(entry)
        if self.path is not None:
            append_jsonl(self.path, entry.to_dict())
        return entry

    @property
    def total_usd(self) -> float:
        return sum(e.usd for e in self.entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.input_tokens + e.output_tokens for e in self.entries)

    def by_phase(self) -> dict[str, float]:
        return self._group(lambda e: e.phase)

    def by_agent(self) -> dict[str, float]:
        return self._group(lambda e: e.agent)

    def by_kind(self) -> dict[str, float]:
        return self._group(lambda e: e.kind)

    def by_model(self) -> dict[str, float]:
        return self._group(lambda e: e.model)

    def summary(self) -> dict[str, Any]:
        return {
            "total_usd": round(self.total_usd, 6),
            "total_tokens": self.total_tokens,
            "input_tokens": sum(e.input_tokens for e in self.entries),
            "output_tokens": sum(e.output_tokens for e in self.entries),
            "calls": len(self.entries),
            "by_phase": self.by_phase(),
            "by_agent": self.by_agent(),
            "by_kind": self.by_kind(),
            "by_model": self.by_model(),
        }

    def render_table(self) -> str:
        """Plain text -- it goes into Markdown reports and human gate packets."""
        rows: list[tuple[str, str, str]] = []
        for label, grouped in (("phase", self.by_phase()), ("agent", self.by_agent())):
            for key, usd in sorted(grouped.items(), key=lambda kv: -kv[1]):
                rows.append((label, key, f"${usd:.4f}"))
        header = ("scope", "name", "usd")
        widths = [
            max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
            for i in range(3)
        ]
        lines = [
            "  ".join(header[i].ljust(widths[i]) for i in range(3)),
            "  ".join("-" * widths[i] for i in range(3)),
        ]
        lines.extend("  ".join(row[i].ljust(widths[i]) for i in range(3)) for row in rows)
        lines.append("")
        lines.append(f"total  ${self.total_usd:.4f}  over {self.total_tokens} tokens")
        return "\n".join(lines)

    def _group(self, key: Callable[[CostEntry], str | None]) -> dict[str, float]:
        totals: dict[str, float] = defaultdict(float)
        for entry in self.entries:
            totals[str(key(entry) or "unattributed")] += entry.usd
        return {k: round(v, 6) for k, v in sorted(totals.items())}


def _entry_from_dict(record: dict[str, Any]) -> CostEntry:
    at = record.get("at")
    return CostEntry(
        kind=str(record.get("kind", "unknown")),
        model=record.get("model"),
        input_tokens=int(record.get("input_tokens", 0)),
        output_tokens=int(record.get("output_tokens", 0)),
        usd=float(record.get("usd", 0.0)),
        phase=record.get("phase"),
        agent=record.get("agent"),
        at=datetime.fromisoformat(at) if isinstance(at, str) else utcnow(),
    )


__all__ = [
    "FALLBACK_MODEL",
    "PRICING",
    "CostEntry",
    "CostLedger",
    "estimate_cost",
]
