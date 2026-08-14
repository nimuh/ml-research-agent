"""The registry enforces per-agent allow-lists and never raises past the loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from ml_research_agent.config import Config, ObservabilityConfig
from ml_research_agent.errors import SandboxViolation, ToolError
from ml_research_agent.observability import logging as mra_logging
from ml_research_agent.observability.logging import configure_logging
from ml_research_agent.observability.tracing import Tracer
from ml_research_agent.tools.registry import (
    ToolContext,
    ToolRegistry,
    ToolResult,
    default_registry,
)


class EchoArgs(BaseModel):
    text: str
    times: int = 1


class EchoTool:
    """Repeat the text."""

    name = "echo"
    description = "Echo the given text."
    parameters = EchoArgs

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, EchoArgs)
        return ToolResult(ok=True, output=args.text * args.times, metadata={"agent": ctx.agent})


class BoomArgs(BaseModel):
    kind: str = "crash"


class BoomTool:
    """Fails in every way a tool can fail."""

    name = "boom"
    description = "Raise something."
    parameters = BoomArgs

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        assert isinstance(args, BoomArgs)
        if args.kind == "sandbox":
            raise SandboxViolation("tried to escape", path="/etc/passwd")
        if args.kind == "tool":
            raise ToolError("upstream said no")
        raise RuntimeError("unhandled explosion")


@pytest.fixture
def registry(config: Config) -> ToolRegistry:
    reg = ToolRegistry(config, tracer=Tracer(enabled=True))
    reg.register(EchoTool())
    reg.register(BoomTool())
    return reg


def test_call_runs_an_allowed_tool(registry: ToolRegistry) -> None:
    result = registry.call("echo", {"text": "hi", "times": 2}, agent="framer", allow=["echo"])

    assert result.ok
    assert result.output == "hihi"
    assert result.metadata["agent"] == "framer"


def test_none_allow_list_means_no_restriction(registry: ToolRegistry) -> None:
    assert registry.call("echo", {"text": "hi"}, agent="framer").ok


def test_a_tool_outside_the_allow_list_is_refused_not_raised(registry: ToolRegistry) -> None:
    result = registry.call("boom", {"kind": "crash"}, agent="framer", allow=["echo"])

    assert result.ok is False
    assert result.metadata["reason"] == "not_allowed"
    assert "allow-list" in (result.error or "")


def test_unknown_and_disabled_tools_are_refused(registry: ToolRegistry, config: Config) -> None:
    unknown = registry.call("nope", {}, agent="framer")
    assert unknown.ok is False
    assert unknown.metadata["reason"] == "unknown"

    registry.register(EchoTool(), enabled=False)
    disabled = registry.call("echo", {"text": "x"}, agent="framer")
    assert disabled.ok is False
    assert disabled.metadata["reason"] == "disabled"


def test_bad_arguments_come_back_as_a_failed_result(registry: ToolRegistry) -> None:
    result = registry.call("echo", {"times": "many"}, agent="framer")

    assert result.ok is False
    assert result.metadata["reason"] == "invalid_arguments"


@pytest.mark.parametrize(
    ("kind", "reason"),
    [("sandbox", "sandbox_violation"), ("tool", "tool_error"), ("crash", "exception")],
)
def test_every_failure_mode_becomes_a_tool_result(
    registry: ToolRegistry, kind: str, reason: str
) -> None:
    result = registry.call("boom", {"kind": kind}, agent="framer")

    assert result.ok is False
    assert result.metadata["reason"] == reason
    assert result.error


def test_schemas_only_covers_available_tools(registry: ToolRegistry) -> None:
    schemas = registry.schemas(["echo", "nope"])

    assert [s["name"] for s in schemas] == ["echo"]
    assert schemas[0]["input_schema"]["properties"]["times"]["default"] == 1


def test_names_excludes_disabled_tools(registry: ToolRegistry) -> None:
    registry.register(BoomTool(), enabled=False)

    assert registry.names() == ["echo"]


def test_get_raises_for_an_unknown_tool(registry: ToolRegistry) -> None:
    with pytest.raises(KeyError):
        registry.get("nope")


def test_calls_are_traced(registry: ToolRegistry) -> None:
    registry.call("echo", {"text": "hi"}, agent="framer")

    spans = registry.tracer.export()
    assert [s["name"] for s in spans] == ["tool.call"]
    assert spans[0]["attributes"]["tool"] == "echo"


def test_refusals_and_violations_are_logged(
    registry: ToolRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unlogged refusal is invisible in an audit, so the denials must land."""
    monkeypatch.setattr(mra_logging, "_sink", mra_logging._sink)  # restored on teardown
    configure_logging(ObservabilityConfig(console=False), log_dir=tmp_path)

    registry.call("boom", {"kind": "crash"}, agent="framer", allow=["echo"])
    registry.call("boom", {"kind": "sandbox"}, agent="framer")

    records = [
        json.loads(line)
        for line in (tmp_path / "mra.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    denied = next(r for r in records if r["event"] == "tool call denied by allow-list")
    assert (denied["level"], denied["tool"], denied["agent"]) == ("WARNING", "boom", "framer")

    violation = next(r for r in records if r["event"] == "sandbox violation")
    assert (violation["level"], violation["tool"]) == ("ERROR", "boom")


def test_default_registry_wires_the_standard_tools(config: Config) -> None:
    registry = default_registry(config)

    assert registry.names() == [
        "http_get",
        "list_dir",
        "patch_file",
        "python_exec",
        "read_file",
        "shell",
        "write_file",
    ]
    schemas: list[dict[str, Any]] = registry.schemas(registry.names())
    assert all(s["description"] for s in schemas)
