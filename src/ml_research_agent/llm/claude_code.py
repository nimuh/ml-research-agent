"""Provider backend that reaches Claude through the Claude Code CLI, using the
developer's existing subscription credentials instead of an API key.

`LLMClient` speaks a single-shot completion contract: given messages and tool
schemas, return text or the tool calls the model chose, having executed
nothing. Claude Code is the opposite shape -- an agent that owns its own loop
and runs tools itself. This module is the adapter between the two, and the two
places it has to work against the grain are worth stating plainly:

*Structured output* maps cleanly. A forced tool call (what
:func:`~ml_research_agent.llm.structured.generate_structured` asks for) becomes
``output_format={"type": "json_schema", ...}``, and the ``structured_output``
that comes back is re-wrapped as the tool call the caller expects. No prompt
coaching, no parsing of prose.

*Tool calls* do not. The framework executes its own tools -- that is what keeps
the allow-list, the audit log, and the workspace sandbox meaningful -- so the
adapter must learn what the model chose without letting Claude Code act on it.
Tools are registered so the model can see their schemas, then every invocation
is denied at the permission gate and reported back to the caller. The handlers
therefore exist to be listed and never to run; the denial is the mechanism, not
a failure. Two consequences follow. A tool entry must stay out of
``allowed_tools``, because an allow rule auto-approves before the gate is
consulted. And the denial must not set ``interrupt``, which aborts the stream
before usage is reported.

Claude Code's own toolset and settings files are both switched off: an agent
that could read the filesystem on its own would route around
`tools/`, and a `CLAUDE.md` picked up from disk would make results depend on
which directory the run started in.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from typing import Any, cast

from ..config import Config
from ..errors import ConfigError
from .client import LLMClient

#: Name of the in-process MCP server the framework's tools are published under.
#: Tool names arrive back prefixed (``mcp__mra__read_file``); the caller only
#: ever knows the bare name, so the prefix is stripped on the way out.
_SERVER = "mra"
_PREFIX = f"mcp__{_SERVER}__"


class ClaudeCodeLLMClient(LLMClient):
    """An :class:`LLMClient` backed by the Claude Code CLI.

    Budget checks, cost accounting, caching, tracing and the retry loop are all
    inherited unchanged -- only the provider call itself differs.
    """

    def cost_for(self, payload: dict[str, Any], *, model: str) -> float:
        """Prefer Claude Code's own figure -- it prices the cache traffic.

        Cache tokens dominate here (the harness prompt is resent every call),
        so a estimate from prompt/completion tokens alone would understate a
        run several-fold.
        """
        reported = float(payload.get("provider_cost_usd", 0.0) or 0.0)
        return reported if reported > 0 else super().cost_for(payload, model=model)

    def budget_usd(self, cost: float) -> float:
        """Subscription usage bills no dollars, so it charges none.

        Claude Code reports what the same tokens would have cost through the
        API. That number is worth recording, but spending it against
        ``budget.usd_per_project`` would halt a run over money nobody was
        charged. Token and wall-clock ceilings still apply, and setting
        ``llm.subscription_cost_is_billed`` restores dollar accounting for
        anyone whose Claude Code usage is in fact metered.
        """
        return cost if self.config.llm.subscription_cost_is_billed else 0.0

    def _invoke(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        payload = _run_sync(self._invoke_async(model=model, messages=messages, params=params))
        return cast("dict[str, Any]", payload)

    async def _invoke_async(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        sdk = _import_sdk()
        forced = _forced_tool(params)
        captured: list[dict[str, Any]] = []

        options = self._options(sdk, model=model, params=params, forced=forced, captured=captured)
        prompt = _render_prompt(messages)

        async def _stream() -> Any:
            yield {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": "default",
            }

        text_parts: list[str] = []
        result: Any = None
        async for message in sdk.query(prompt=_stream(), options=options):
            if isinstance(message, sdk.AssistantMessage):
                for block in message.content:
                    if isinstance(block, sdk.TextBlock):
                        text_parts.append(block.text)
            elif isinstance(message, sdk.ResultMessage):
                result = message

        if result is None:
            raise ConfigError("Claude Code returned no result message", model=model)
        if getattr(result, "is_error", False):
            raise ConfigError(
                "Claude Code reported an error",
                model=model,
                detail=str(getattr(result, "errors", None) or result.terminal_reason)[:300],
            )

        return _payload(
            result,
            model=model,
            text="".join(text_parts),
            forced=forced,
            captured=captured,
        )

    def _options(
        self,
        sdk: Any,
        *,
        model: str,
        params: dict[str, Any],
        forced: dict[str, Any] | None,
        captured: list[dict[str, Any]],
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "system_prompt": params.get("system") or None,
            # An empty toolset removes Claude Code's own Read/Bash/Grep. The
            # framework's tools are the only ones that may run, and they run
            # through `tools/`, not here.
            "tools": [],
            "allowed_tools": [],
            # Settings files would otherwise inject the ambient CLAUDE.md and
            # make a run depend on its working directory.
            "setting_sources": [],
        }

        if forced is not None:
            kwargs["output_format"] = {
                "type": "json_schema",
                "schema": forced["input_schema"],
            }
        elif params.get("tools"):
            kwargs["mcp_servers"] = {_SERVER: _server_for(sdk, params["tools"])}
            kwargs["can_use_tool"] = _capture_gate(sdk, captured)

        return sdk.ClaudeAgentOptions(**kwargs)


def _capture_gate(sdk: Any, captured: list[dict[str, Any]]) -> Any:
    """Record the model's tool choice and refuse to run it.

    The caller executes the tool and supplies the result on the next turn, so
    denial here is the contract rather than an error. ``interrupt`` stays off:
    interrupting aborts the stream before usage is reported, which would lose
    the token accounting for the call.
    """

    async def gate(tool_name: str, input_data: dict[str, Any], context: Any) -> Any:
        captured.append({"name": tool_name, "arguments": dict(input_data or {})})
        return sdk.PermissionResultDeny(
            message=(
                "The host application executes this tool and will supply its result. "
                "Do not retry it; end your turn."
            ),
            interrupt=False,
        )

    return gate


def _server_for(sdk: Any, tools: list[dict[str, Any]]) -> Any:
    """Publish the caller's tool schemas so the model can choose among them."""
    published = []
    for spec in tools:

        @sdk.tool(  # type: ignore[untyped-decorator]  # the SDK ships no stubs
            str(spec["name"]),
            str(spec.get("description", ""))[:1024],
            spec.get("input_schema") or {"type": "object", "properties": {}},
        )
        async def _never_runs(args: dict[str, Any]) -> dict[str, Any]:
            # Unreachable: the permission gate denies every call. Present only
            # so the tool has a schema the model can see.
            raise AssertionError("framework tools must not execute inside Claude Code")

        published.append(_never_runs)
    return sdk.create_sdk_mcp_server(name=_SERVER, version="1.0.0", tools=published)


def _forced_tool(params: dict[str, Any]) -> dict[str, Any] | None:
    """The tool a `tool_choice` pins, when the caller demanded exactly one."""
    choice = params.get("tool_choice") or {}
    if choice.get("type") != "tool":
        return None
    name = choice.get("name")
    specs: list[dict[str, Any]] = params.get("tools") or []
    for spec in specs:
        if spec.get("name") == name:
            return spec
    return None


def _render_prompt(messages: list[dict[str, Any]]) -> str:
    """Flatten the conversation into one prompt.

    Each call is stateless -- the caller owns the transcript and resends it --
    so the whole conversation is rendered rather than resumed from a session.
    """
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        lines.append(f"[{role}]\n{_render_content(message.get('content'))}")
    return "\n\n".join(lines)


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "tool_use":
            args = json.dumps(block.get("input", {}), default=str)
            parts.append(f"(called `{block.get('name')}` with {args})")
        elif kind == "tool_result":
            parts.append(f"(result of `{block.get('tool_use_id', '')}`)\n{block.get('content')}")
        else:
            parts.append(str(block))
    return "\n".join(parts)


def _payload(
    result: Any,
    *,
    model: str,
    text: str,
    forced: dict[str, Any] | None,
    captured: list[dict[str, Any]],
) -> dict[str, Any]:
    usage = getattr(result, "usage", None) or {}
    tool_calls: list[dict[str, Any]] = []

    if forced is not None:
        structured = getattr(result, "structured_output", None)
        if isinstance(structured, dict):
            tool_calls.append(
                {"id": "toolu_claude_code", "name": str(forced["name"]), "arguments": structured}
            )
    else:
        for index, call in enumerate(captured):
            name = str(call["name"])
            tool_calls.append(
                {
                    "id": f"toolu_claude_code_{index}",
                    # Strip the MCP server prefix the SDK adds, so the caller
                    # matches on the name it registered.
                    "name": name[len(_PREFIX) :] if name.startswith(_PREFIX) else name,
                    "arguments": call["arguments"],
                }
            )

    stop_reason = str(getattr(result, "stop_reason", "") or "")
    fresh = int(usage.get("input_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)

    return {
        "text": text,
        "tool_calls": tool_calls,
        "model": model,
        # Claude Code reports almost all input as cache traffic -- its harness
        # prompt is ~20k tokens and is re-sent every call, so `input_tokens`
        # alone came back as 2 on a real run that processed tens of thousands.
        # Counting only that would leave `budget.tokens_per_phase` -- the
        # ceiling that still bites when dollars do not -- off by orders of
        # magnitude. The breakdown is kept beside it for anyone reading the
        # ledger.
        "input_tokens": fresh + cache_creation + cache_read,
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "fresh_input_tokens": fresh,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "stop_reason": "tool_use" if (tool_calls and not forced) else (stop_reason or "end_turn"),
        "provider_cost_usd": float(getattr(result, "total_cost_usd", 0.0) or 0.0),
    }


def _import_sdk() -> Any:
    try:
        import claude_agent_sdk
    except ImportError as exc:
        raise ConfigError(
            "provider `claude_code` needs the Claude Agent SDK: "
            "pip install 'ml-research-agent[claude-code]'"
        ) from exc
    return claude_agent_sdk


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from this synchronous codebase.

    ``asyncio.run`` refuses to nest, so when a loop is already running (a
    notebook, an async caller) the work moves to a worker thread with a loop of
    its own rather than failing.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def build_client(config: Config, **kwargs: Any) -> LLMClient:
    """Pick the backend named by ``llm.provider``.

    This is the only place the provider name is read, so a new backend means a
    new branch here and nowhere else.
    """
    if config.llm.provider == "claude_code":
        return ClaudeCodeLLMClient(config, **kwargs)
    return LLMClient(config, **kwargs)


__all__ = ["ClaudeCodeLLMClient", "build_client"]
