"""Token/cost budgets per project, phase and agent; raises BudgetExceeded and
lets the orchestrator degrade gracefully instead of silently truncating work."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from ..config import BudgetConfig
from ..errors import BudgetExceeded
from ..observability.cost import CostLedger


class BudgetTracker:
    """Hard ceilings on spend: USD per project, tokens per phase, wall clock.

    The contract is *check before you spend*: :meth:`check` is called with the
    projected cost of a call and raises before the money leaves. A tracker that
    only noticed overspend afterwards would make the ceiling advisory, and the
    invariant is that budgets stop work rather than degrading it.
    """

    def __init__(self, config: BudgetConfig, *, ledger: CostLedger | None = None) -> None:
        self.config = config
        self.ledger = ledger
        # A resumed project inherits what the persisted ledger already recorded,
        # otherwise a crash-and-resume would silently double the effective budget.
        self._spent_usd: float = ledger.total_usd if ledger is not None else 0.0
        self._tokens: dict[str, int] = defaultdict(int)
        if ledger is not None:
            for entry in ledger.entries:
                self._tokens[entry.phase or "unattributed"] += (
                    entry.input_tokens + entry.output_tokens
                )
        self._phase_usd: dict[str, float] = defaultdict(float)
        if ledger is not None:
            for entry in ledger.entries:
                self._phase_usd[entry.phase or "unattributed"] += entry.usd
        self.started_at = time.monotonic()

    # -- accounting ---------------------------------------------------------

    def record(self, *, usd: float, tokens: int = 0, phase: str | None = None) -> None:
        """Book spend that already happened. Never raises -- :meth:`check` does."""
        key = phase or "unattributed"
        self._spent_usd += float(usd)
        self._phase_usd[key] += float(usd)
        self._tokens[key] += int(tokens)

    def check(
        self,
        *,
        projected_usd: float = 0.0,
        projected_tokens: int = 0,
        phase: str | None = None,
    ) -> None:
        """Raise :class:`BudgetExceeded` if the projected spend breaks a ceiling."""
        self._check_wallclock()
        key = phase or "unattributed"

        projected_total = self._spent_usd + max(projected_usd, 0.0)
        if projected_total > self.config.usd_per_project:
            raise BudgetExceeded(
                "project USD ceiling reached",
                scope="project",
                limit=self.config.usd_per_project,
                spent=round(projected_total, 6),
                unit="usd",
                phase=key,
            )

        if self.config.usd_per_phase is not None:
            phase_total = self._phase_usd[key] + max(projected_usd, 0.0)
            if phase_total > self.config.usd_per_phase:
                raise BudgetExceeded(
                    "phase USD ceiling reached",
                    scope="phase",
                    limit=self.config.usd_per_phase,
                    spent=round(phase_total, 6),
                    unit="usd",
                    phase=key,
                )

        token_total = self._tokens[key] + max(projected_tokens, 0)
        if token_total > self.config.tokens_per_phase:
            raise BudgetExceeded(
                "phase token ceiling reached",
                scope="phase",
                limit=float(self.config.tokens_per_phase),
                spent=float(token_total),
                unit="tokens",
                phase=key,
            )

    def _check_wallclock(self) -> None:
        elapsed = self.elapsed_hours()
        if elapsed > self.config.max_wallclock_hours:
            raise BudgetExceeded(
                "wall-clock ceiling reached",
                scope="project",
                limit=self.config.max_wallclock_hours,
                spent=round(elapsed, 4),
                unit="hours",
            )

    # -- introspection ------------------------------------------------------

    @property
    def spent_usd(self) -> float:
        return round(self._spent_usd, 6)

    @property
    def remaining_usd(self) -> float:
        return round(max(self.config.usd_per_project - self._spent_usd, 0.0), 6)

    def tokens_for_phase(self, phase: str) -> int:
        return self._tokens[phase or "unattributed"]

    def remaining_tokens(self, phase: str) -> int:
        return max(self.config.tokens_per_phase - self.tokens_for_phase(phase), 0)

    def elapsed_hours(self) -> float:
        return (time.monotonic() - self.started_at) / 3600.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_usd,
            "usd_per_project": self.config.usd_per_project,
            "usd_per_phase": self.config.usd_per_phase,
            "tokens_per_phase": self.config.tokens_per_phase,
            "tokens_by_phase": dict(sorted(self._tokens.items())),
            "usd_by_phase": {k: round(v, 6) for k, v in sorted(self._phase_usd.items())},
            "elapsed_hours": round(self.elapsed_hours(), 4),
            "max_wallclock_hours": self.config.max_wallclock_hours,
        }


__all__ = ["BudgetTracker"]
