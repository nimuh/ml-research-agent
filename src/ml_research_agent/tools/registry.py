"""Tool registry + allow-lists: which agent may call which tool, with what
argument constraints."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from ..config import Config
from ..errors import MRAError, SandboxViolation
from ..observability.logging import StructuredLogger, get_logger
from ..observability.tracing import Tracer


@dataclass
class ToolResult:
    """Uniform tool outcome. Tools report failure; they do not raise past the loop."""

    ok: bool
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_content(self, *, max_chars: int = 8000) -> str:
        """Render for a ``tool_result`` content block."""
        text = self.error if not self.ok and self.error else str(self.output)
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars]"
        return text


@dataclass
class ToolContext:
    """Everything a tool is allowed to know about the run invoking it."""

    workspace: Path
    config: Config
    agent: str | None = None
    logger: StructuredLogger | None = None
    tracer: Tracer | None = None


@runtime_checkable
class Tool(Protocol):
    """A typed, permissioned, individually disableable capability.

    ``parameters`` is a read-only property rather than a plain attribute so an
    implementation may narrow it to its own args model -- a mutable attribute
    would be invariant and every concrete tool would fail the protocol check.
    """

    name: str
    description: str

    @property
    def parameters(self) -> type[BaseModel]: ...

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...


class ToolRegistry:
    """Holds the tools and enforces the per-agent allow-list.

    ``call`` never raises past the agent loop: a denied tool, bad arguments, or
    a crashing implementation all come back as ``ToolResult(ok=False)`` so the
    model can see what went wrong and adjust. Everything is logged and traced,
    including the denials -- an unlogged refusal is invisible in an audit.
    """

    def __init__(
        self,
        config: Config,
        *,
        logger: StructuredLogger | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or get_logger("tools.registry")
        self.tracer = tracer or Tracer(enabled=False)
        self._tools: dict[str, Tool] = {}
        self._enabled: dict[str, bool] = {}

    def register(self, tool: Tool, *, enabled: bool = True) -> None:
        self._tools[tool.name] = tool
        self._enabled[tool.name] = enabled

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return tool

    def names(self) -> list[str]:
        return sorted(name for name, on in self._enabled.items() if on)

    def schemas(self, names: Iterable[str]) -> list[dict[str, Any]]:
        """Anthropic tool schemas for the subset an agent is allowed to use."""
        schemas: list[dict[str, Any]] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None or not self._enabled.get(name, False):
                self.logger.warning("tool schema requested for unavailable tool", tool=name)
                continue
            schema = tool.parameters.model_json_schema()
            schema.pop("title", None)
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description.strip(),
                    "input_schema": schema,
                }
            )
        return schemas

    def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        agent: str,
        allow: Iterable[str] | None = None,
    ) -> ToolResult:
        allowed = None if allow is None else set(allow)
        logger = self.logger.bind(agent=agent, tool=name)

        if allowed is not None and name not in allowed:
            logger.warning("tool call denied by allow-list", allowed=sorted(allowed))
            return ToolResult(
                ok=False,
                output=None,
                error=f"tool {name!r} is not in this agent's allow-list",
                metadata={"denied": True, "reason": "not_allowed"},
            )
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("unknown tool call", known=sorted(self._tools))
            return ToolResult(
                ok=False,
                output=None,
                error=f"unknown tool {name!r}",
                metadata={"denied": True, "reason": "unknown"},
            )
        if not self._enabled.get(name, False):
            logger.warning("disabled tool call")
            return ToolResult(
                ok=False,
                output=None,
                error=f"tool {name!r} is disabled",
                metadata={"denied": True, "reason": "disabled"},
            )

        ctx = ToolContext(
            workspace=Path(self.config.paths.workspace),
            config=self.config,
            agent=agent,
            logger=logger,
            tracer=self.tracer,
        )
        with self.tracer.span("tool.call", tool=name, agent=agent) as span:
            try:
                parsed = tool.parameters.model_validate(arguments)
            except ValidationError as exc:
                span.set(status="invalid_arguments")
                logger.warning("tool arguments failed validation", detail=str(exc)[:500])
                return ToolResult(
                    ok=False,
                    output=None,
                    error=f"invalid arguments for {name!r}: {exc}",
                    metadata={"reason": "invalid_arguments"},
                )
            try:
                result = tool.run(parsed, ctx)
            except SandboxViolation as exc:
                # A violation is a security event: surfaced to the agent as a
                # failure, but logged at error level so it is never lost.
                span.set(status="sandbox_violation")
                logger.error("sandbox violation", detail=str(exc))
                return ToolResult(
                    ok=False,
                    output=None,
                    error=f"sandbox violation: {exc}",
                    metadata={"reason": "sandbox_violation"},
                )
            except MRAError as exc:
                span.set(status="tool_error")
                logger.warning("tool failed", detail=str(exc))
                return ToolResult(
                    ok=False, output=None, error=str(exc), metadata={"reason": "tool_error"}
                )
            except Exception as exc:
                span.set(status="crash")
                logger.error("tool crashed", error=type(exc).__name__, detail=str(exc)[:500])
                return ToolResult(
                    ok=False,
                    output=None,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata={"reason": "exception"},
                )
            span.set(status="ok" if result.ok else "failed")

        logger.info("tool call", ok=result.ok, **_loggable(result.metadata))
        return result


def default_registry(
    config: Config,
    *,
    logger: StructuredLogger | None = None,
    tracer: Tracer | None = None,
) -> ToolRegistry:
    """The standard tool set: filesystem, shell, scratch python, HTTP."""
    from .fs import ListDirTool, PatchFileTool, ReadFileTool, WriteFileTool
    from .http import HttpTool
    from .python_exec import PythonExecTool
    from .shell import ShellTool

    registry = ToolRegistry(config, logger=logger, tracer=tracer)
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirTool())
    registry.register(PatchFileTool())
    registry.register(ShellTool())
    registry.register(PythonExecTool())
    registry.register(HttpTool(config))
    return registry


def _loggable(metadata: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metadata.items() if isinstance(v, str | int | float | bool)}


__all__ = [
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
]
