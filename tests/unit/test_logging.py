"""Logs are the audit trail -- and must never carry a secret."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from ml_research_agent.config import ObservabilityConfig
from ml_research_agent.observability import logging as mra_logging
from ml_research_agent.observability.logging import configure_logging, get_logger, redact


@pytest.fixture(autouse=True)
def restore_sink() -> Iterator[None]:
    """configure_logging installs process-wide state; put it back afterwards."""
    original = mra_logging._sink
    yield
    mra_logging._sink = original


def read_records(log_dir: Path) -> list[dict[str, object]]:
    lines = (log_dir / "mra.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_writes_jsonl_with_correlation_ids(tmp_path: Path) -> None:
    configure_logging(ObservabilityConfig(console=False), log_dir=tmp_path)
    logger = get_logger("test", run_id="run_1").bind(phase="frame", agent="framer")

    logger.info("agent finished", cost_usd=0.25)

    record = read_records(tmp_path)[0]
    assert record["event"] == "agent finished"
    assert record["run_id"] == "run_1"
    assert record["phase"] == "frame"
    assert record["agent"] == "framer"
    assert record["cost_usd"] == 0.25
    assert record["level"] == "INFO"


def test_bind_returns_a_new_logger(tmp_path: Path) -> None:
    configure_logging(ObservabilityConfig(console=False), log_dir=tmp_path)
    base = get_logger("test", run_id="run_1")
    bound = base.bind(agent="scout")

    assert bound is not base
    assert "agent" not in base.context
    assert bound.context["agent"] == "scout"


def test_secrets_are_redacted_recursively(tmp_path: Path) -> None:
    configure_logging(ObservabilityConfig(console=False), log_dir=tmp_path)

    get_logger("test").warning(
        "request",
        api_key="sk-ant-do-not-log-me",
        headers={"Authorization": "Bearer sk-ant-secret", "user-agent": "mra"},
        nested=[{"anthropic_api_key": "sk-ant-nested"}],
        harmless="visible",
    )

    raw = (tmp_path / "mra.jsonl").read_text(encoding="utf-8")
    assert "sk-ant" not in raw
    record = read_records(tmp_path)[0]
    assert record["api_key"] == "***redacted***"
    assert record["headers"] == {"Authorization": "***redacted***", "user-agent": "mra"}
    assert record["nested"] == [{"anthropic_api_key": "***redacted***"}]
    assert record["harmless"] == "visible"


def test_configured_redact_keys_are_the_ones_applied(tmp_path: Path) -> None:
    """The defaults passing is not evidence that the *configured* list is used."""
    configure_logging(
        ObservabilityConfig(console=False, redact_keys=["clearance", "api_key"]),
        log_dir=tmp_path,
    )

    get_logger("test").info("request", clearance="top-secret-value", harmless="visible")

    record = read_records(tmp_path)[0]
    assert record["clearance"] == "***redacted***"
    assert record["harmless"] == "visible"
    assert "top-secret-value" not in (tmp_path / "mra.jsonl").read_text(encoding="utf-8")
    assert mra_logging.active_redact_keys() == ("clearance", "api_key")


def test_level_filtering(tmp_path: Path) -> None:
    configure_logging(ObservabilityConfig(log_level="WARNING", console=False), log_dir=tmp_path)
    logger = get_logger("test")

    logger.debug("quiet")
    logger.info("also quiet")
    logger.error("loud")

    assert [r["event"] for r in read_records(tmp_path)] == ["loud"]


def test_redact_helper_is_usable_standalone() -> None:
    assert redact({"token": "abc", "keep": 1}, ["token"]) == {"token": "***redacted***", "keep": 1}


def test_logging_without_configure_does_not_write_anywhere(tmp_path: Path) -> None:
    mra_logging._sink = mra_logging._Sink()

    get_logger("test").info("no sink configured")

    assert not list(tmp_path.iterdir())
