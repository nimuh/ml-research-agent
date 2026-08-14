"""Spans must nest, time, and record failure -- a run you cannot explain is broken."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_research_agent.config import ObservabilityConfig
from ml_research_agent.observability import logging as mra_logging
from ml_research_agent.observability.logging import configure_logging
from ml_research_agent.observability.tracing import Tracer


def test_span_records_name_status_and_duration() -> None:
    tracer = Tracer()

    with tracer.span("phase.frame", phase="frame") as span:
        span.set(agent="framer")

    exported = tracer.export()
    assert len(exported) == 1
    assert exported[0]["name"] == "phase.frame"
    assert exported[0]["status"] == "ok"
    assert exported[0]["attributes"] == {"phase": "frame", "agent": "framer"}
    assert exported[0]["duration_seconds"] >= 0.0


def test_spans_nest_via_parent_id_and_depth() -> None:
    tracer = Tracer()

    with tracer.span("outer"):
        with tracer.span("inner"):
            pass
        with tracer.span("sibling"):
            pass

    outer, inner, sibling = tracer.export()
    assert inner["parent_id"] == outer["id"]
    assert sibling["parent_id"] == outer["id"]
    assert (outer["depth"], inner["depth"], sibling["depth"]) == (0, 1, 1)
    assert tracer.current is None


def test_an_exception_marks_the_span_and_still_propagates() -> None:
    tracer = Tracer()

    with pytest.raises(ValueError, match="nope"), tracer.span("failing"):
        raise ValueError("nope")

    span = tracer.export()[0]
    assert span["status"] == "error"
    assert span["attributes"]["error"] == "ValueError"
    assert span["attributes"]["error_message"] == "nope"
    assert tracer.current is None  # the stack unwound


def test_none_attributes_are_dropped() -> None:
    tracer = Tracer()

    with tracer.span("call", phase=None, agent="scout") as span:
        span.set(cost_usd=None)

    assert tracer.export()[0]["attributes"] == {"agent": "scout"}


def test_disabled_tracer_still_yields_a_usable_span() -> None:
    tracer = Tracer(enabled=False)

    with tracer.span("ignored") as span:
        span.set(anything="fine")

    assert tracer.export() == []
    assert tracer.render_timeline() == "(no spans recorded)"


def test_render_timeline_is_indented_and_totalled() -> None:
    tracer = Tracer()
    with tracer.span("outer"), tracer.span("inner", tool="read_file"):
        pass

    timeline = tracer.render_timeline()
    lines = timeline.splitlines()
    assert lines[0].startswith("+ outer")
    assert lines[1].startswith("+   inner")
    assert "tool=read_file" in lines[1]
    assert lines[-1].startswith("= total")


def test_secrets_are_redacted_from_exports_and_timelines() -> None:
    """PLAN §6: secrets are redacted in traces, not only in logs.

    Spans leave the process -- exported to disk, rendered into gate packets --
    so masking has to happen at the egress point rather than relying on every
    caller of ``span.set`` to be careful.
    """
    tracer = Tracer()

    with tracer.span("llm.complete", model="claude-opus-5") as span:
        span.set(api_key="sk-ant-must-not-appear", headers={"authorization": "Bearer sk-ant-x"})

    exported = tracer.export()[0]
    assert exported["attributes"]["api_key"] == "***redacted***"
    assert exported["attributes"]["headers"] == {"authorization": "***redacted***"}
    assert exported["attributes"]["model"] == "claude-opus-5"
    assert "sk-ant" not in str(exported)
    assert "sk-ant" not in tracer.render_timeline()


def test_traces_mask_the_configured_keys_not_a_hardcoded_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One configured redaction list, not a second copy in the tracer to drift."""
    monkeypatch.setattr(mra_logging, "_sink", mra_logging._sink)  # restored on teardown
    configure_logging(
        ObservabilityConfig(console=False, redact_keys=["clearance"]), log_dir=tmp_path
    )
    tracer = Tracer()

    with tracer.span("gate") as span:
        span.set(clearance="top-secret-value", phase="design")

    exported = tracer.export()[0]
    assert exported["attributes"]["clearance"] == "***redacted***"
    assert exported["attributes"]["phase"] == "design"
    assert "top-secret-value" not in tracer.render_timeline()


def test_reset_clears_state() -> None:
    tracer = Tracer()
    with tracer.span("x"):
        pass
    tracer.reset()

    assert tracer.export() == []
