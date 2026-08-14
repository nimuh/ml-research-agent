"""The Claude Code backend, exercised without the SDK or the network.

The adapter's whole job is translating between a single-shot completion
contract and an agent that wants to run its own loop, so these tests pin the
translation in both directions -- and pin the two settings that stop Claude
Code reaching past `tools/`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from ml_research_agent.config import Config
from ml_research_agent.errors import ConfigError
from ml_research_agent.llm.claude_code import ClaudeCodeLLMClient, build_client
from ml_research_agent.llm.client import LLMClient

# -- a fake claude_agent_sdk ------------------------------------------------


@dataclasses.dataclass
class _TextBlock:
    text: str


@dataclasses.dataclass
class _AssistantMessage:
    content: list[Any]


@dataclasses.dataclass
class _ResultMessage:
    usage: dict[str, Any] | None = None
    structured_output: Any = None
    total_cost_usd: float | None = None
    stop_reason: str | None = "end_turn"
    is_error: bool = False
    terminal_reason: str | None = "completed"
    errors: list[str] | None = None


@dataclasses.dataclass
class _Deny:
    message: str
    interrupt: bool


@dataclasses.dataclass
class _SdkTool:
    name: str
    description: str
    input_schema: Any
    handler: Any


class FakeSDK:
    """Records what the adapter asked for and replays a scripted turn."""

    AssistantMessage = _AssistantMessage
    TextBlock = _TextBlock
    ResultMessage = _ResultMessage
    PermissionResultDeny = _Deny

    def __init__(
        self,
        *,
        text: str = "",
        result: _ResultMessage | None = None,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.text = text
        self.result = result or _ResultMessage(usage={"input_tokens": 10, "output_tokens": 5})
        self.tool_calls = tool_calls or []
        self.options: Any = None
        self.registered: list[_SdkTool] = []
        self.gate_replies: list[_Deny] = []

    # -- surface the adapter uses
    def ClaudeAgentOptions(self, **kwargs: Any) -> dict[str, Any]:  # noqa: N802
        self.options = kwargs
        return kwargs

    def tool(self, name: str, description: str, input_schema: Any) -> Any:
        def wrap(handler: Any) -> _SdkTool:
            return _SdkTool(name, description, input_schema, handler)

        return wrap

    def create_sdk_mcp_server(self, *, name: str, version: str, tools: list[_SdkTool]) -> Any:
        self.registered = list(tools)
        return {"name": name, "tools": [t.name for t in tools]}

    async def query(self, *, prompt: Any, options: Any) -> Any:
        async for _ in prompt:  # drain the streaming-mode prompt
            pass
        gate = options.get("can_use_tool")
        for name, args in self.tool_calls:
            assert gate is not None, "tool calls scripted but no permission gate installed"
            self.gate_replies.append(await gate(name, args, None))
        if self.text:
            yield _AssistantMessage(content=[_TextBlock(self.text)])
        yield self.result


@pytest.fixture
def client_factory(monkeypatch: pytest.MonkeyPatch):
    def make(sdk: FakeSDK, config: Config | None = None) -> ClaudeCodeLLMClient:
        monkeypatch.setattr(
            "ml_research_agent.llm.claude_code._import_sdk", lambda: sdk, raising=True
        )
        return ClaudeCodeLLMClient(config or Config())

    return make


# -- structured output ------------------------------------------------------


def test_a_forced_tool_call_becomes_a_json_schema_request(client_factory) -> None:
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    sdk = FakeSDK(
        result=_ResultMessage(
            usage={"input_tokens": 3, "output_tokens": 4},
            structured_output={"title": "a brief"},
        )
    )
    client = client_factory(sdk)

    response = client.complete(
        messages=[{"role": "user", "content": "frame this"}],
        tools=[{"name": "emit", "description": "Emit it.", "input_schema": schema}],
        tool_choice={"type": "tool", "name": "emit"},
    )

    # The forced call is expressed as an output format, not coached in prose.
    assert sdk.options["output_format"] == {"type": "json_schema", "schema": schema}
    assert "can_use_tool" not in sdk.options, "structured output needs no permission gate"
    # ...and comes back shaped as the tool call the caller asked for.
    call = response.tool_call("emit")
    assert call is not None and call.arguments == {"title": "a brief"}


def test_structured_output_that_never_arrives_is_not_invented(client_factory) -> None:
    # A missing structured_output must surface as "no tool call" so the repair
    # loop runs, rather than as an empty object that validates to nonsense.
    sdk = FakeSDK(result=_ResultMessage(usage={}, structured_output=None))
    client = client_factory(sdk)
    response = client.complete(
        messages=[{"role": "user", "content": "x"}],
        tools=[{"name": "emit", "description": "d", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "emit"},
    )
    assert response.tool_calls == []


# -- the tool loop ----------------------------------------------------------


def test_tool_calls_are_captured_and_denied_never_executed(client_factory) -> None:
    sdk = FakeSDK(
        tool_calls=[
            ("mcp__mra__kb_search", {"query": "label smoothing"}),
            ("mcp__mra__read_file", {"path": "a.py"}),
        ]
    )
    client = client_factory(sdk)

    response = client.complete(
        messages=[{"role": "user", "content": "look it up"}],
        tools=[
            {"name": "kb_search", "description": "Search.", "input_schema": {"type": "object"}},
            {"name": "read_file", "description": "Read.", "input_schema": {"type": "object"}},
        ],
    )

    # The caller gets bare names back, not the MCP server prefix.
    assert [(c.name, c.arguments) for c in response.tool_calls] == [
        ("kb_search", {"query": "label smoothing"}),
        ("read_file", {"path": "a.py"}),
    ]
    assert response.stop_reason == "tool_use"
    # Every call was refused, and refused without interrupting -- an interrupt
    # aborts the stream before usage is reported.
    assert [d.interrupt for d in sdk.gate_replies] == [False, False]


def test_registered_tool_handlers_must_never_run(client_factory) -> None:
    # The handlers exist so the model can see a schema. If one is ever actually
    # invoked, a framework tool ran inside Claude Code -- outside the
    # allow-list and the audit log -- so it must fail loudly rather than work.
    sdk = FakeSDK()
    client = client_factory(sdk)
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "write_file", "description": "Write.", "input_schema": {"type": "object"}}],
    )
    (registered,) = sdk.registered
    assert registered.name == "write_file"
    with pytest.raises(AssertionError):
        import asyncio

        asyncio.run(registered.handler({}))


def test_tools_are_kept_out_of_the_allow_list(client_factory) -> None:
    # An allowed_tools entry auto-approves the call *before* the permission
    # gate is consulted, which would let Claude Code execute a framework tool.
    sdk = FakeSDK()
    client = client_factory(sdk)
    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "kb_search", "description": "s", "input_schema": {"type": "object"}}],
    )
    assert sdk.options["allowed_tools"] == []
    assert sdk.options["can_use_tool"] is not None


def test_claude_codes_own_tools_and_settings_are_disabled(client_factory) -> None:
    # Its built-in toolset would route around `tools/`; settings files would
    # inject the ambient CLAUDE.md and make results depend on the cwd.
    sdk = FakeSDK(text="ok")
    client = client_factory(sdk)
    client.complete(messages=[{"role": "user", "content": "hi"}])
    assert sdk.options["tools"] == []
    assert sdk.options["setting_sources"] == []


# -- transcript rendering ---------------------------------------------------


def test_the_whole_transcript_is_resent_including_tool_results(client_factory) -> None:
    # Each call is stateless, so a tool result gathered on an earlier turn only
    # reaches the model if it is rendered into the prompt.
    sdk = FakeSDK(text="ok")
    client = client_factory(sdk)
    captured: list[str] = []

    async def spy(*, prompt: Any, options: Any) -> Any:
        async for chunk in prompt:
            captured.append(chunk["message"]["content"])
        yield sdk.result

    sdk.query = spy  # type: ignore[method-assign]
    client.complete(
        messages=[
            {"role": "user", "content": "find papers"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "name": "kb_search", "input": {"q": "vit"}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "3 hits found"}
                ],
            },
        ]
    )
    (prompt,) = captured
    assert "find papers" in prompt
    assert "kb_search" in prompt
    assert "3 hits found" in prompt


# -- accounting -------------------------------------------------------------


def test_subscription_cost_is_recorded_but_charges_no_budget(client_factory) -> None:
    sdk = FakeSDK(
        text="hi",
        result=_ResultMessage(usage={"input_tokens": 2, "output_tokens": 5}, total_cost_usd=0.187),
    )
    client = client_factory(sdk)
    response = client.complete(messages=[{"role": "user", "content": "hi"}])

    # The API-equivalent figure is kept for visibility...
    assert response.cost_usd == pytest.approx(0.187)
    # ...but nothing is charged against a ceiling nobody is billed against.
    assert client.budget_usd(response.cost_usd) == 0.0


def test_cache_traffic_counts_towards_the_token_ceiling(client_factory) -> None:
    # Claude Code re-sends its ~20k-token harness prompt every call and reports
    # it as cache traffic, so `input_tokens` alone is nearly zero. The token
    # ceiling is the guard that still bites when dollars do not, and counting
    # only fresh input would leave it off by orders of magnitude.
    sdk = FakeSDK(
        text="hi",
        result=_ResultMessage(
            usage={
                "input_tokens": 2,
                "cache_creation_input_tokens": 18_627,
                "cache_read_input_tokens": 560,
                "output_tokens": 5,
            }
        ),
    )
    client = client_factory(sdk)
    response = client.complete(messages=[{"role": "user", "content": "hi"}])
    assert response.input_tokens == 2 + 18_627 + 560
    assert response.raw is not None
    assert response.raw["fresh_input_tokens"] == 2  # breakdown kept for the ledger


def test_metered_claude_code_usage_can_charge_the_budget() -> None:
    config = Config.model_validate({"llm": {"subscription_cost_is_billed": True}})
    client = ClaudeCodeLLMClient(config)
    assert client.budget_usd(0.42) == pytest.approx(0.42)


def test_reported_cost_wins_over_the_token_estimate() -> None:
    # Cache traffic dominates here and the token estimate cannot see it, so the
    # provider's own figure is the honest one.
    client = ClaudeCodeLLMClient(Config())
    payload = {"input_tokens": 1, "output_tokens": 1, "provider_cost_usd": 0.5}
    assert client.cost_for(payload, model="claude-opus-5") == pytest.approx(0.5)


def test_cost_falls_back_to_the_estimate_when_none_is_reported() -> None:
    client = ClaudeCodeLLMClient(Config())
    payload = {"input_tokens": 1000, "output_tokens": 1000, "provider_cost_usd": 0.0}
    assert client.cost_for(payload, model="claude-opus-5") > 0


# -- wiring and failure modes -----------------------------------------------


def test_the_provider_setting_selects_the_backend() -> None:
    assert type(build_client(Config())) is LLMClient
    claude_code = Config.model_validate({"llm": {"provider": "claude_code"}})
    assert isinstance(build_client(claude_code), ClaudeCodeLLMClient)


def test_a_missing_sdk_names_the_install_that_fixes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "claude_agent_sdk":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    client = ClaudeCodeLLMClient(Config())
    with pytest.raises(ConfigError, match="claude-code"):
        client.complete(messages=[{"role": "user", "content": "hi"}])


def test_an_errored_turn_is_raised_not_returned_as_empty_text(client_factory) -> None:
    sdk = FakeSDK(
        result=_ResultMessage(usage={}, is_error=True, errors=["rate limited"], stop_reason=None)
    )
    client = client_factory(sdk)
    with pytest.raises(ConfigError, match="rate limited"):
        client.complete(messages=[{"role": "user", "content": "hi"}])


def test_no_api_key_is_required(client_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    # The point of the backend: it must not touch require_api_key at all.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sdk = FakeSDK(text="PONG")
    client = client_factory(sdk)
    assert client.complete(messages=[{"role": "user", "content": "ping"}]).text == "PONG"
