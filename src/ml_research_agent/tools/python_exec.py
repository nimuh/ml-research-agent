"""Scratch Python execution for analysis/plots -- separate from experiment runs,
short-lived, resource-capped."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..errors import ToolError
from ..utils.hashing import hash_text
from ..utils.io import ensure_dir, safe_join, write_text
from .registry import ToolContext, ToolResult

DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 600
DEFAULT_MEMORY_MB = 2048
MAX_OUTPUT_CHARS = 20_000
SCRATCH_DIR = "scratch/python"


class PythonExecArgs(BaseModel):
    code: str = Field(min_length=1, description="Python source to execute.")
    working_dir: str = Field(default=".", description="Relative to the workspace root.")
    timeout_seconds: int = Field(default=DEFAULT_TIMEOUT, ge=1, le=MAX_TIMEOUT)


class PythonExecTool:
    """Run a short Python snippet in a fresh subprocess.

    A subprocess rather than ``exec()``: in-process execution shares our
    interpreter with model-authored code, which means no memory ceiling, no
    timeout that actually kills anything, and a crash takes the orchestrator
    with it. This is for analysis and plots -- experiment runs go through
    ``experiments/sandbox.py``.
    """

    name = "python_exec"
    description = (
        "Execute a short Python snippet in a fresh subprocess scoped to the workspace, for "
        "analysis or plotting. State is not preserved between calls. stdout/stderr are captured "
        "and truncated; the process is killed at timeout_seconds. Not for experiment runs."
    )
    parameters = PythonExecArgs

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if not isinstance(args, PythonExecArgs):
            raise ToolError("unexpected argument type", expected="PythonExecArgs")
        workspace = ensure_dir(ctx.workspace)
        cwd = safe_join(workspace, args.working_dir)
        if not cwd.is_dir():
            raise ToolError("working_dir is not a directory", working_dir=args.working_dir)

        script_dir = ensure_dir(safe_join(workspace, SCRATCH_DIR))
        script = script_dir / f"snippet_{hash_text(args.code, length=12)}.py"
        write_text(script, args.code)

        memory_mb = ctx.config.sandbox.max_memory_mb or DEFAULT_MEMORY_MB
        try:
            completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
                [sys.executable, "-I", str(script)],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
                shell=False,
                env=_child_env(workspace),
                preexec_fn=_limits(memory_mb) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                "python snippet timed out",
                retryable=False,
                timeout_seconds=args.timeout_seconds,
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
                "exit_code": completed.returncode,
                "script": str(script.relative_to(workspace)),
                "memory_mb": memory_mb,
                "truncated": len(completed.stdout or "") > MAX_OUTPUT_CHARS,
            },
        )


def _child_env(workspace: Path) -> dict[str, str]:
    """A minimal environment. Notably: no API keys reach model-authored code."""
    keep = ("PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR", "SYSTEMROOT")
    env = {k: v for k, v in os.environ.items() if k in keep}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MRA_WORKSPACE"] = str(workspace)
    env["MPLBACKEND"] = "Agg"
    return env


def _limits(memory_mb: int) -> Any:
    def _apply() -> None:  # pragma: no cover - runs in the forked child
        import contextlib
        import resource

        limit = memory_mb * 1024 * 1024
        # Best-effort: some platforms refuse individual rlimits, and a snippet
        # that runs uncapped is better than one that cannot start at all.
        for which in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(which, (limit, limit))
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))

    return _apply


def _truncate(value: str | bytes | None, limit: int = MAX_OUTPUT_CHARS) -> str:
    if value is None:
        return ""
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


__all__ = ["DEFAULT_MEMORY_MB", "MAX_OUTPUT_CHARS", "PythonExecArgs", "PythonExecTool"]
