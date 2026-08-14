"""Filesystem tools: traversal must be impossible, not merely discouraged."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_research_agent.config import Config
from ml_research_agent.errors import SandboxViolation, ToolError
from ml_research_agent.tools.fs import (
    ListDirArgs,
    ListDirTool,
    PatchFileArgs,
    PatchFileTool,
    ReadFileArgs,
    ReadFileTool,
    WriteFileArgs,
    WriteFileTool,
)
from ml_research_agent.tools.registry import ToolContext


@pytest.fixture
def ctx(config: Config, workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, config=config, agent="tester")


def test_write_then_read(ctx: ToolContext) -> None:
    written = WriteFileTool().run(WriteFileArgs(path="notes/a.md", content="hello"), ctx)
    assert written.ok

    read = ReadFileTool().run(ReadFileArgs(path="notes/a.md"), ctx)
    assert read.ok
    assert read.output == "hello"
    assert read.metadata["truncated"] is False


def test_read_truncates_with_an_explicit_marker(ctx: ToolContext) -> None:
    WriteFileTool().run(WriteFileArgs(path="big.txt", content="x" * 5000), ctx)

    result = ReadFileTool().run(ReadFileArgs(path="big.txt", max_bytes=100), ctx)

    assert result.metadata["truncated"] is True
    assert "truncated" in result.output
    assert result.output.startswith("x" * 100)


def test_read_missing_file_is_a_tool_error(ctx: ToolContext) -> None:
    with pytest.raises(ToolError):
        ReadFileTool().run(ReadFileArgs(path="nope.txt"), ctx)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "../../etc/passwd",
        "notes/../../escape.txt",
        "/etc/passwd",
        "/tmp/escape.txt",
    ],
)
def test_traversal_attempts_raise_sandbox_violation(ctx: ToolContext, path: str) -> None:
    with pytest.raises(SandboxViolation):
        ReadFileTool().run(ReadFileArgs(path=path), ctx)
    with pytest.raises(SandboxViolation):
        WriteFileTool().run(WriteFileArgs(path=path, content="pwned"), ctx)
    with pytest.raises(SandboxViolation):
        ListDirTool().run(ListDirArgs(path=path), ctx)


def test_a_symlink_out_of_the_workspace_is_refused(ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (ctx.workspace / "link.txt").symlink_to(outside)

    with pytest.raises(SandboxViolation):
        ReadFileTool().run(ReadFileArgs(path="link.txt"), ctx)


def test_a_symlinked_directory_out_of_the_workspace_is_refused(
    ctx: ToolContext, tmp_path: Path
) -> None:
    # Reading through a symlink is only half the hole: writing, listing and
    # patching through one must be refused too.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (ctx.workspace / "esc").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SandboxViolation):
        WriteFileTool().run(WriteFileArgs(path="esc/planted.txt", content="pwned"), ctx)
    with pytest.raises(SandboxViolation):
        ListDirTool().run(ListDirArgs(path="esc"), ctx)
    with pytest.raises(SandboxViolation):
        PatchFileTool().run(
            PatchFileArgs(path="esc/secret.txt", old_string="secret", new_string="pwned"), ctx
        )
    assert not (outside / "planted.txt").exists()
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret"


def test_a_prefix_sibling_of_the_workspace_root_is_refused(config: Config, tmp_path: Path) -> None:
    """``/w/ws-evil`` starts with ``/w/ws`` but is not inside it.

    A containment check written as a string prefix would let this through,
    which is why the invariant is stated as "impossible by construction".
    """
    root = tmp_path / "ws"
    root.mkdir()
    sibling = tmp_path / "ws-evil"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("secret", encoding="utf-8")
    ctx = ToolContext(workspace=root, config=config, agent="tester")

    for path in ("../ws-evil/secret.txt", str(sibling / "secret.txt")):
        with pytest.raises(SandboxViolation):
            ReadFileTool().run(ReadFileArgs(path=path), ctx)
        with pytest.raises(SandboxViolation):
            WriteFileTool().run(WriteFileArgs(path=path, content="pwned"), ctx)

    assert (sibling / "secret.txt").read_text(encoding="utf-8") == "secret"


def test_write_stays_inside_and_creates_parents(ctx: ToolContext, workspace: Path) -> None:
    WriteFileTool().run(WriteFileArgs(path="deep/nested/x.txt", content="ok"), ctx)

    assert (workspace / "deep" / "nested" / "x.txt").read_text(encoding="utf-8") == "ok"


def test_list_dir_lists_and_truncates(ctx: ToolContext) -> None:
    for i in range(5):
        WriteFileTool().run(WriteFileArgs(path=f"d/f{i}.txt", content="x"), ctx)

    listing = ListDirTool().run(ListDirArgs(path="d"), ctx)
    assert listing.metadata["count"] == 5
    assert "f0.txt" in listing.output

    truncated = ListDirTool().run(ListDirArgs(path="d", max_entries=2), ctx)
    assert truncated.metadata["truncated"] is True
    assert "3 more entries" in truncated.output


def test_list_dir_on_a_file_is_a_tool_error(ctx: ToolContext) -> None:
    WriteFileTool().run(WriteFileArgs(path="a.txt", content="x"), ctx)

    with pytest.raises(ToolError):
        ListDirTool().run(ListDirArgs(path="a.txt"), ctx)


def test_patch_replaces_a_unique_string(ctx: ToolContext, workspace: Path) -> None:
    WriteFileTool().run(WriteFileArgs(path="cfg.yaml", content="lr: 0.1\nseed: 0\n"), ctx)

    result = PatchFileTool().run(
        PatchFileArgs(path="cfg.yaml", old_string="lr: 0.1", new_string="lr: 0.3"), ctx
    )

    assert result.ok
    assert (workspace / "cfg.yaml").read_text(encoding="utf-8") == "lr: 0.3\nseed: 0\n"


def test_patch_refuses_ambiguous_and_absent_matches(ctx: ToolContext) -> None:
    WriteFileTool().run(WriteFileArgs(path="dup.txt", content="a\na\n"), ctx)

    with pytest.raises(ToolError):
        PatchFileTool().run(PatchFileArgs(path="dup.txt", old_string="a", new_string="b"), ctx)
    with pytest.raises(ToolError):
        PatchFileTool().run(PatchFileArgs(path="dup.txt", old_string="zzz", new_string="b"), ctx)
