"""Workspace-scoped filesystem read/write/patch; path traversal is impossible
by construction."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..errors import ToolError
from ..utils.io import ensure_dir, safe_join, write_text
from .registry import ToolContext, ToolResult

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 2_000_000
MAX_ENTRIES = 500

_TRUNCATED = "\n... [truncated: file is {size} bytes, showed {shown}]"


def _resolve(ctx: ToolContext, relative: str) -> Path:
    """Every path in this module goes through here. No exceptions.

    ``safe_join`` resolves symlinks and absolute components before comparing,
    so ``..``, ``/etc/passwd`` and a symlink out of the tree all raise
    :class:`SandboxViolation` rather than being sanitized into something that
    silently works.
    """
    return safe_join(ensure_dir(ctx.workspace), relative)


class ReadFileArgs(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")
    max_bytes: int = Field(default=MAX_READ_BYTES, ge=1, le=MAX_READ_BYTES)


class ReadFileTool:
    """Read a UTF-8 text file from the workspace."""

    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. Paths are relative to the workspace root; "
        "absolute paths and '..' are refused. Large files are truncated with an explicit marker."
    )
    parameters = ReadFileArgs

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if not isinstance(args, ReadFileArgs):
            raise ToolError("unexpected argument type", expected="ReadFileArgs")
        path = _resolve(ctx, args.path)
        if not path.is_file():
            raise ToolError("file not found", path=args.path)
        size = path.stat().st_size
        data = path.read_bytes()[: args.max_bytes]
        text = data.decode("utf-8", errors="replace")
        truncated = size > args.max_bytes
        if truncated:
            text += _TRUNCATED.format(size=size, shown=args.max_bytes)
        return ToolResult(
            ok=True,
            output=text,
            metadata={"path": args.path, "bytes": size, "truncated": truncated},
        )


class WriteFileArgs(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")
    content: str = Field(description="Full file contents to write.")
    create_dirs: bool = True


class WriteFileTool:
    """Write a UTF-8 text file inside the workspace."""

    name = "write_file"
    description = (
        "Write a UTF-8 text file inside the workspace, replacing it if it exists. "
        "Paths are relative to the workspace root; absolute paths and '..' are refused."
    )
    parameters = WriteFileArgs

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if not isinstance(args, WriteFileArgs):
            raise ToolError("unexpected argument type", expected="WriteFileArgs")
        encoded = args.content.encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            raise ToolError("content too large", bytes=len(encoded), limit=MAX_WRITE_BYTES)
        path = _resolve(ctx, args.path)
        if args.create_dirs:
            ensure_dir(path.parent)
        elif not path.parent.is_dir():
            raise ToolError("parent directory does not exist", path=args.path)
        write_text(path, args.content)
        return ToolResult(
            ok=True,
            output=f"wrote {len(encoded)} bytes to {args.path}",
            metadata={"path": args.path, "bytes": len(encoded)},
        )


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Directory relative to the workspace root.")
    recursive: bool = False
    max_entries: int = Field(default=MAX_ENTRIES, ge=1, le=MAX_ENTRIES)


class ListDirTool:
    """List a directory inside the workspace."""

    name = "list_dir"
    description = (
        "List files and directories inside the workspace. Set recursive=true to walk the tree. "
        "Output is truncated at max_entries with an explicit marker."
    )
    parameters = ListDirArgs

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if not isinstance(args, ListDirArgs):
            raise ToolError("unexpected argument type", expected="ListDirArgs")
        root = _resolve(ctx, args.path)
        if not root.is_dir():
            raise ToolError("not a directory", path=args.path)
        paths = sorted(root.rglob("*") if args.recursive else root.iterdir())
        entries = [
            f"{p.relative_to(root).as_posix()}{'/' if p.is_dir() else f' ({p.stat().st_size}b)'}"
            for p in paths[: args.max_entries]
        ]
        truncated = len(paths) > args.max_entries
        if truncated:
            entries.append(f"... [truncated: {len(paths) - args.max_entries} more entries]")
        return ToolResult(
            ok=True,
            output="\n".join(entries) or "(empty)",
            metadata={"path": args.path, "count": len(paths), "truncated": truncated},
        )


class PatchFileArgs(BaseModel):
    path: str = Field(description="Path relative to the workspace root.")
    old_string: str = Field(min_length=1, description="Exact text to replace; must be unique.")
    new_string: str = Field(description="Replacement text.")


class PatchFileTool:
    """Replace one exact, unique string in a workspace file."""

    name = "patch_file"
    description = (
        "Replace an exact string in a workspace file. The old_string must appear exactly once, "
        "so include enough surrounding context to make it unique. Fails rather than guessing."
    )
    parameters = PatchFileArgs

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if not isinstance(args, PatchFileArgs):
            raise ToolError("unexpected argument type", expected="PatchFileArgs")
        path = _resolve(ctx, args.path)
        if not path.is_file():
            raise ToolError("file not found", path=args.path)
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(args.old_string)
        if occurrences == 0:
            raise ToolError("old_string not found", path=args.path)
        if occurrences > 1:
            raise ToolError("old_string is not unique", path=args.path, occurrences=occurrences)
        write_text(path, original.replace(args.old_string, args.new_string, 1))
        return ToolResult(
            ok=True,
            output=f"patched {args.path}",
            metadata={
                "path": args.path,
                "delta_chars": len(args.new_string) - len(args.old_string),
            },
        )


__all__ = [
    "MAX_ENTRIES",
    "MAX_READ_BYTES",
    "MAX_WRITE_BYTES",
    "ListDirArgs",
    "ListDirTool",
    "PatchFileArgs",
    "PatchFileTool",
    "ReadFileArgs",
    "ReadFileTool",
    "WriteFileArgs",
    "WriteFileTool",
]
