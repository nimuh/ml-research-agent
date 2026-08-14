"""Guarded shell execution inside a workspace: command allow/deny lists,
timeouts, output capture and truncation."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from ..errors import SandboxViolation, ToolError
from ..utils.io import ensure_dir, safe_join
from .registry import ToolContext, ToolResult

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 1800
MAX_OUTPUT_CHARS = 20_000

# Conservative allow-list of read/inspect-oriented binaries. Anything that runs
# third-party code belongs in experiments/sandbox.py, not here: this tool exists
# so an agent can look around its workspace, not so it can execute a repo.
ALLOWED_BINARIES: frozenset[str] = frozenset(
    {
        "cat",
        "cut",
        "diff",
        "du",
        "echo",
        "find",
        "git",
        "grep",
        "head",
        "ls",
        "md5sum",
        "mkdir",
        "pwd",
        "rg",
        "sed",
        "sha256sum",
        "sort",
        "stat",
        "tail",
        "tree",
        "uniq",
        "wc",
        "which",
    }
)

# Shell metacharacters would let a caller chain past the allow-listed binary,
# so the command is parsed with shlex and executed without a shell at all.
_SHELL_METACHARS = ("|", "&", ";", ">", "<", "`", "$(", "\n")


class ShellArgs(BaseModel):
    command: str = Field(min_length=1, description="A single command, no shell operators.")
    working_dir: str = Field(default=".", description="Relative to the workspace root.")
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT, ge=1, le=MAX_TIMEOUT)


class ShellTool:
    """Run one allow-listed command inside the workspace."""

    name = "shell"
    description = (
        "Run a single read-only shell command inside the workspace (ls, cat, grep, find, git, "
        "wc, ...). No pipes, redirects, or command chaining; no arbitrary binaries. Output is "
        "captured and truncated, and the command is killed at timeout_seconds."
    )
    parameters = ShellArgs

    def __init__(self, allowed: frozenset[str] | None = None) -> None:
        self.allowed = allowed if allowed is not None else ALLOWED_BINARIES

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if not isinstance(args, ShellArgs):
            raise ToolError("unexpected argument type", expected="ShellArgs")
        argv = self._vet(args.command, ctx)
        cwd = safe_join(ensure_dir(ctx.workspace), args.working_dir)
        if not cwd.is_dir():
            raise ToolError("working_dir is not a directory", working_dir=args.working_dir)
        return _execute(argv, cwd=cwd, timeout=args.timeout_seconds)

    def _vet(self, command: str, ctx: ToolContext) -> list[str]:
        stripped = command.strip()
        denied = [d for d in ctx.config.sandbox.denied_commands if d and d in stripped]
        if denied:
            raise SandboxViolation("denied command", command=stripped[:200], matched=denied[0])
        found = [c for c in _SHELL_METACHARS if c in stripped]
        if found:
            raise SandboxViolation(
                "shell operators are not permitted", command=stripped[:200], operator=found[0]
            )
        try:
            argv = shlex.split(stripped)
        except ValueError as exc:
            raise ToolError("could not parse command", detail=str(exc)) from exc
        if not argv:
            raise ToolError("empty command")
        binary = Path(argv[0]).name
        if binary not in self.allowed:
            raise SandboxViolation(
                "command is not on the allow-list",
                binary=binary,
                allowed=sorted(self.allowed),
            )
        return argv


def _execute(argv: list[str], *, cwd: Path, timeout: int) -> ToolResult:
    try:
        completed = subprocess.run(  # noqa: S603 - argv is allow-listed and shell=False
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ToolError("binary not found on this machine", binary=argv[0]) from exc
    except subprocess.TimeoutExpired as exc:
        raise ToolError(
            "command timed out",
            retryable=False,
            command=" ".join(argv)[:200],
            timeout_seconds=timeout,
            stdout=_truncate(exc.stdout),
        ) from exc

    stdout = _truncate(completed.stdout)
    stderr = _truncate(completed.stderr)
    ok = completed.returncode == 0
    return ToolResult(
        ok=ok,
        output=stdout if ok else f"{stdout}\n{stderr}".strip(),
        error=None if ok else f"exit {completed.returncode}: {stderr[:2000]}",
        metadata={
            "command": " ".join(argv)[:200],
            "exit_code": completed.returncode,
            "stdout_chars": len(completed.stdout or ""),
            "truncated": len(completed.stdout or "") > MAX_OUTPUT_CHARS,
        },
    )


def _truncate(value: str | bytes | None, limit: int = MAX_OUTPUT_CHARS) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


__all__ = ["ALLOWED_BINARIES", "MAX_OUTPUT_CHARS", "ShellArgs", "ShellTool"]
