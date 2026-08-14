"""The shell tool is a security boundary: deny-list, allow-list, and a real timeout."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ml_research_agent.config import Config
from ml_research_agent.errors import SandboxViolation, ToolError
from ml_research_agent.tools.registry import ToolContext
from ml_research_agent.tools.shell import ShellArgs, ShellTool


@pytest.fixture
def ctx(config: Config, workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, config=config, agent="tester")


def test_allow_listed_command_runs(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.txt").write_text("hello", encoding="utf-8")

    result = ShellTool().run(ShellArgs(command="ls"), ctx)

    assert result.ok
    assert "a.txt" in result.output
    assert result.metadata["exit_code"] == 0


def test_nonzero_exit_is_reported_not_raised(ctx: ToolContext) -> None:
    result = ShellTool().run(ShellArgs(command="ls definitely_not_here"), ctx)

    assert result.ok is False
    assert result.metadata["exit_code"] != 0
    assert result.error is not None


@pytest.mark.parametrize("command", ["shutdown", "shutdown -h now", "mkfs.ext4 /dev/sda"])
def test_denied_commands_raise(ctx: ToolContext, command: str) -> None:
    with pytest.raises(SandboxViolation):
        ShellTool().run(ShellArgs(command=command), ctx)


def test_the_deny_list_is_enforced_for_an_allow_listed_binary(
    config: Config, workspace: Path
) -> None:
    """Every default-denied string is also off the allow-list, so those cases
    cannot tell the two checks apart. This one can: ``ls`` is allow-listed."""
    denied_ctx = ToolContext(
        workspace=workspace,
        config=config.with_overrides(**{"sandbox.denied_commands": ["ls -la"]}),
        agent="tester",
    )

    with pytest.raises(SandboxViolation) as excinfo:
        ShellTool().run(ShellArgs(command="ls -la"), denied_ctx)

    assert excinfo.value.context["matched"] == "ls -la"
    assert excinfo.value.retryable is False
    # ...and the same binary without the denied string still runs.
    assert ShellTool().run(ShellArgs(command="ls"), denied_ctx).ok


@pytest.mark.parametrize(
    "command",
    ["ls | grep a", "ls && rm -rf .", "ls; echo hi", "cat a.txt > b.txt", "echo `whoami`"],
)
def test_shell_operators_are_refused(ctx: ToolContext, command: str) -> None:
    with pytest.raises(SandboxViolation):
        ShellTool().run(ShellArgs(command=command), ctx)


@pytest.mark.parametrize("command", ["python -c 'print(1)'", "curl https://example.com", "bash"])
def test_binaries_off_the_allow_list_are_refused(ctx: ToolContext, command: str) -> None:
    with pytest.raises(SandboxViolation):
        ShellTool().run(ShellArgs(command=command), ctx)


def test_working_dir_cannot_escape_the_workspace(ctx: ToolContext) -> None:
    with pytest.raises(SandboxViolation):
        ShellTool().run(ShellArgs(command="ls", working_dir="../.."), ctx)


@pytest.mark.skipif(shutil.which("sleep") is None, reason="no sleep binary")
def test_timeout_kills_the_command(ctx: ToolContext) -> None:
    tool = ShellTool(allowed=frozenset({"sleep"}))

    with pytest.raises(ToolError) as excinfo:
        tool.run(ShellArgs(command="sleep 10", timeout_seconds=1), ctx)

    assert "timed out" in excinfo.value.message
    assert excinfo.value.context["timeout_seconds"] == 1


def test_output_is_truncated(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "big.txt").write_text("y" * 50_000, encoding="utf-8")

    result = ShellTool().run(ShellArgs(command="cat big.txt"), ctx)

    assert result.metadata["truncated"] is True
    assert "truncated" in result.output
    assert len(result.output) < 50_000
