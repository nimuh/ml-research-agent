"""Cost accounting: pricing, attribution, and survival across a crash."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_research_agent.config import ObservabilityConfig
from ml_research_agent.observability import logging as mra_logging
from ml_research_agent.observability.cost import (
    PRICING,
    CostLedger,
    estimate_cost,
)
from ml_research_agent.observability.logging import configure_logging


def test_estimate_cost_uses_the_model_price() -> None:
    per_input, per_output = PRICING["claude-sonnet-5"]
    expected = (1_000_000 / 1_000_000) * per_input + (500_000 / 1_000_000) * per_output

    assert estimate_cost("claude-sonnet-5", 1_000_000, 500_000) == expected


def test_unknown_model_falls_back_to_standard_pricing() -> None:
    fallback = estimate_cost("claude-not-a-model", 1_000_000, 0)

    assert fallback == estimate_cost("claude-sonnet-5", 1_000_000, 0)
    # An unpriced call must never be silently free, or the ceiling is unenforceable.
    assert fallback > 0


def test_unknown_model_pricing_is_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mra_logging, "_sink", mra_logging._sink)  # restored on teardown
    configure_logging(ObservabilityConfig(console=False), log_dir=tmp_path)

    estimate_cost("claude-not-a-model", 1_000, 1_000)

    records = [
        json.loads(line)
        for line in (tmp_path / "mra.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r["level"] for r in records] == ["WARNING"]
    assert records[0]["model"] == "claude-not-a-model"


def test_zero_tokens_cost_nothing() -> None:
    assert estimate_cost("claude-opus-5", 0, 0) == 0.0


def test_totals_and_attribution() -> None:
    ledger = CostLedger()
    ledger.record(
        kind="llm",
        usd=1.0,
        model="claude-opus-5",
        input_tokens=100,
        output_tokens=50,
        phase="frame",
        agent="framer",
    )
    ledger.record(
        kind="llm",
        usd=0.25,
        model="claude-haiku-4-5-20251001",
        input_tokens=10,
        output_tokens=5,
        phase="survey",
        agent="scout",
    )
    ledger.record(kind="compute", usd=0.5, phase="run")

    assert ledger.total_usd == 1.75
    assert ledger.total_tokens == 165
    assert ledger.by_phase() == {"frame": 1.0, "run": 0.5, "survey": 0.25}
    assert ledger.by_agent() == {"framer": 1.0, "scout": 0.25, "unattributed": 0.5}
    assert ledger.summary()["calls"] == 3


def test_entries_persist_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "cost.jsonl"
    ledger = CostLedger(path)
    ledger.record(kind="llm", usd=2.0, model="claude-opus-5", input_tokens=7, output_tokens=3)

    # A fresh ledger over the same file is what a resumed run sees.
    reloaded = CostLedger(path)
    assert reloaded.total_usd == 2.0
    assert reloaded.total_tokens == 10
    assert reloaded.entries[0].model == "claude-opus-5"


def test_render_table_includes_the_total() -> None:
    ledger = CostLedger()
    ledger.record(kind="llm", usd=1.5, phase="design", agent="planner")

    table = ledger.render_table()
    assert "design" in table
    assert "planner" in table
    assert "$1.5000" in table


def test_render_table_on_an_empty_ledger() -> None:
    assert "total" in CostLedger().render_table()
