"""Budgets are hard stops: they must raise before the spend, never after."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_research_agent.config import BudgetConfig
from ml_research_agent.errors import BudgetExceeded
from ml_research_agent.llm.budget import BudgetTracker
from ml_research_agent.observability.cost import CostLedger


def make(**overrides: object) -> BudgetTracker:
    defaults: dict[str, object] = {
        "usd_per_project": 10.0,
        "tokens_per_phase": 1000,
        "max_wallclock_hours": 1.0,
    }
    defaults.update(overrides)
    return BudgetTracker(BudgetConfig(**defaults))  # type: ignore[arg-type]


def test_records_and_reports_spend() -> None:
    tracker = make()
    tracker.record(usd=2.5, tokens=100, phase="survey")

    assert tracker.spent_usd == 2.5
    assert tracker.remaining_usd == 7.5
    assert tracker.tokens_for_phase("survey") == 100
    assert tracker.tokens_for_phase("curate") == 0


def test_check_passes_under_the_ceiling() -> None:
    tracker = make()
    tracker.record(usd=5.0, phase="survey")

    tracker.check(projected_usd=4.0, phase="survey")


def test_project_ceiling_raises_before_the_spend() -> None:
    tracker = make()
    tracker.record(usd=9.0, phase="survey")

    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.check(projected_usd=2.0, phase="survey")

    assert excinfo.value.scope == "project"
    assert excinfo.value.unit == "usd"
    assert excinfo.value.retryable is False
    # The spend never happened, so the tracker must be unchanged.
    assert tracker.spent_usd == 9.0


def test_phase_token_ceiling_is_per_phase() -> None:
    tracker = make()
    tracker.record(usd=0.0, tokens=900, phase="survey")

    tracker.check(projected_tokens=900, phase="curate")

    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.check(projected_tokens=200, phase="survey")
    assert excinfo.value.unit == "tokens"
    assert excinfo.value.context["phase"] == "survey"


def test_optional_phase_usd_ceiling() -> None:
    tracker = make(usd_per_phase=1.0)
    tracker.record(usd=0.9, phase="design")

    tracker.check(projected_usd=0.05, phase="design")
    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.check(projected_usd=0.5, phase="design")
    assert excinfo.value.scope == "phase"


def test_wallclock_ceiling() -> None:
    tracker = make()
    tracker.started_at -= 2 * 3600  # monotonic seconds: pretend two hours passed

    with pytest.raises(BudgetExceeded) as excinfo:
        tracker.check()
    assert excinfo.value.unit == "hours"


def test_resume_inherits_spend_from_a_persisted_ledger(tmp_path: Path) -> None:
    path = tmp_path / "cost.jsonl"
    ledger = CostLedger(path)
    ledger.record(kind="llm", usd=9.5, input_tokens=600, output_tokens=200, phase="survey")

    resumed = BudgetTracker(
        BudgetConfig(usd_per_project=10.0, tokens_per_phase=1000, max_wallclock_hours=1.0),
        ledger=CostLedger(path),
    )

    assert resumed.spent_usd == 9.5
    assert resumed.tokens_for_phase("survey") == 800
    with pytest.raises(BudgetExceeded):
        resumed.check(projected_usd=1.0, phase="survey")


def test_a_refused_check_books_nothing_anywhere(tmp_path: Path) -> None:
    """Every ceiling must raise *before* the spend, not book it and complain."""
    path = tmp_path / "cost.jsonl"
    ledger = CostLedger(path)
    ledger.record(kind="llm", usd=0.9, input_tokens=500, output_tokens=400, phase="design")
    tracker = BudgetTracker(
        BudgetConfig(
            usd_per_project=10.0,
            usd_per_phase=1.0,
            tokens_per_phase=1000,
            max_wallclock_hours=1.0,
        ),
        ledger=ledger,
    )
    before = tracker.snapshot()

    refusals = [
        {"projected_usd": 0.5},  # phase USD ceiling
        {"projected_tokens": 200},  # phase token ceiling
        {"projected_usd": 100.0},  # project USD ceiling
    ]
    for projection in refusals:
        with pytest.raises(BudgetExceeded) as excinfo:
            tracker.check(phase="design", **projection)  # type: ignore[arg-type]
        # Never retryable: a ceiling is a stop, not a transient failure.
        assert excinfo.value.retryable is False

    assert tracker.spent_usd == 0.9
    assert tracker.tokens_for_phase("design") == 900
    assert {k: v for k, v in tracker.snapshot().items() if k != "elapsed_hours"} == {
        k: v for k, v in before.items() if k != "elapsed_hours"
    }
    # ...and nothing was written to the persisted ledger either.
    assert len(ledger.entries) == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_snapshot_is_serializable() -> None:
    tracker = make()
    tracker.record(usd=1.0, tokens=10, phase="frame")

    snapshot = tracker.snapshot()
    assert snapshot["spent_usd"] == 1.0
    assert snapshot["tokens_by_phase"] == {"frame": 10}
    assert snapshot["usd_by_phase"] == {"frame": 1.0}
