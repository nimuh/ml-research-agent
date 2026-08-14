"""Provider client wrapper: model tiers (fast/standard/deep), streaming, tool
calling, retries with backoff, and per-call cost accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import Config
from ..errors import AgentFailure, ConfigError
from ..observability.cost import CostLedger, estimate_cost
from ..observability.logging import StructuredLogger, get_logger
from ..observability.tracing import Tracer
from ..types import ModelTier
from ..utils.concurrency import retry_sync
from .budget import BudgetTracker
from .cache import ResponseCache

# Model families that reject `temperature`/`top_p`/`top_k` outright (a non-default
# value is a 400, not a warning). Sending sampling params is opt-in per family
# rather than opt-out so a new deep-tier model fails safe.
_NO_SAMPLING_PARAMS = (
    "claude-opus-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

# Roughly 4 characters per token; only ever used to *project* a call's cost for
# the pre-flight budget check. Actual accounting always uses reported usage.
_CHARS_PER_TOKEN = 4


@dataclass
class ToolCall:
    """A tool the model asked for, already parsed out of the content blocks."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """One completion, with everything the ledger and the trace need."""

    text: str
    tool_calls: list[ToolCall]
    model: str
    tier: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    stop_reason: str
    cached: bool
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def tool_call(self, name: str) -> ToolCall | None:
        return next((c for c in self.tool_calls if c.name == name), None)


class LLMClient:
    """The only thing in the system that talks to a model provider.

    Every call is budget-checked before it happens, priced after it happens,
    traced, logged (metadata only -- never the key), and optionally served from
    a deterministic cache.
    """

    #: Whether this client can reach a provider. ``offline`` mode only has to
    #: block clients that would actually open a connection -- a scripted client
    #: makes no request, so refusing it would only make offline tests
    #: impossible to write.
    reaches_network: bool = True

    def __init__(
        self,
        config: Config,
        *,
        cache: ResponseCache | None = None,
        budget: BudgetTracker | None = None,
        ledger: CostLedger | None = None,
        tracer: Tracer | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.config = config
        self.cache = cache
        self.budget = budget
        self.ledger = ledger
        self.tracer = tracer or Tracer(enabled=False)
        self.logger = logger or get_logger("llm.client")
        self._sdk: Any | None = None

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tier: ModelTier | str = ModelTier.STANDARD,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        tool_choice: dict[str, Any] | None = None,
        phase: str | None = None,
        agent: str | None = None,
    ) -> LLMResponse:
        tier_name = tier.value if isinstance(tier, ModelTier) else str(tier)
        model = self.config.model_for(tier_name)
        params: dict[str, Any] = {
            "max_tokens": max_tokens or self.config.llm.max_tokens,
            "temperature": (self.config.llm.temperature if temperature is None else temperature),
            "system": system,
            "tools": tools,
            "tool_choice": tool_choice,
            "stop_sequences": stop_sequences,
        }

        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.key(model=model, messages=messages, params=params)
            hit = self.cache.get(cache_key)
            if hit is not None:
                self.logger.debug(
                    "llm cache hit", model=model, tier=tier_name, phase=phase, agent=agent
                )
                return _response_from_payload(hit, tier=tier_name, cached=True, cost_usd=0.0)

        if self.config.offline and self.reaches_network:
            raise ConfigError(
                "offline mode: no cached response for this request",
                model=model,
                tier=tier_name,
                phase=phase,
                agent=agent,
            )

        projected_input = _estimate_tokens(messages, system)
        projected_output = int(params["max_tokens"])
        if self.budget is not None:
            self.budget.check(
                projected_usd=estimate_cost(model, projected_input, projected_output),
                projected_tokens=projected_input + projected_output,
                phase=phase,
            )

        with self.tracer.span(
            "llm.complete", model=model, tier=tier_name, phase=phase, agent=agent
        ) as span:
            payload = self._invoke_with_retries(
                model=model, messages=messages, params=params, phase=phase, agent=agent
            )
            cost = estimate_cost(model, payload["input_tokens"], payload["output_tokens"])
            span.set(
                input_tokens=payload["input_tokens"],
                output_tokens=payload["output_tokens"],
                cost_usd=round(cost, 6),
                stop_reason=payload.get("stop_reason"),
                tool_calls=len(payload.get("tool_calls", [])),
            )

        if self.ledger is not None:
            self.ledger.record(
                kind="llm",
                usd=cost,
                model=model,
                input_tokens=payload["input_tokens"],
                output_tokens=payload["output_tokens"],
                phase=phase,
                agent=agent,
            )
        if self.budget is not None:
            self.budget.record(
                usd=cost,
                tokens=payload["input_tokens"] + payload["output_tokens"],
                phase=phase,
            )
        if self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, payload)

        self.logger.info(
            "llm call",
            model=model,
            tier=tier_name,
            phase=phase,
            agent=agent,
            input_tokens=payload["input_tokens"],
            output_tokens=payload["output_tokens"],
            cost_usd=round(cost, 6),
            stop_reason=payload.get("stop_reason"),
        )
        return _response_from_payload(payload, tier=tier_name, cached=False, cost_usd=cost)

    def close(self) -> None:
        client = self._sdk
        self._sdk = None
        if client is not None:
            closer = getattr(client, "close", None)
            if callable(closer):
                closer()

    # -- provider plumbing --------------------------------------------------

    def _invoke_with_retries(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        phase: str | None,
        agent: str | None,
    ) -> dict[str, Any]:
        attempts = max(self.config.llm.max_retries, 0) + 1

        def _attempt() -> dict[str, Any]:
            return self._invoke(model=model, messages=messages, params=params)

        def _on_retry(attempt: int, exc: BaseException) -> None:
            self.logger.warning(
                "llm call failed; retrying",
                model=model,
                phase=phase,
                agent=agent,
                attempt=attempt,
                error=type(exc).__name__,
                detail=str(exc)[:300],
            )

        try:
            return retry_sync(_attempt, attempts=attempts, on_retry=_on_retry)
        except ConfigError:
            raise
        except Exception as exc:
            raise AgentFailure(
                "LLM call failed after retries",
                retryable=True,
                model=model,
                phase=phase,
                agent=agent,
                error=type(exc).__name__,
                detail=str(exc)[:300],
            ) from exc

    def _invoke(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        """One provider call, normalized to a cacheable payload dict."""
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": params["max_tokens"],
            "messages": messages,
        }
        if params.get("system"):
            request["system"] = params["system"]
        if params.get("tools"):
            request["tools"] = params["tools"]
        if params.get("tool_choice"):
            request["tool_choice"] = params["tool_choice"]
        if params.get("stop_sequences"):
            request["stop_sequences"] = params["stop_sequences"]
        if supports_sampling_params(model) and params.get("temperature") is not None:
            request["temperature"] = params["temperature"]

        message = self._client().messages.create(**request)
        return _payload_from_message(message, model=model)

    def _client(self) -> Any:
        """Lazily construct the SDK client so tests never need a key."""
        if self._sdk is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ConfigError("the `anthropic` package is not installed") from exc
            self._sdk = anthropic.Anthropic(
                api_key=self.config.require_api_key(),
                timeout=self.config.llm.timeout_seconds,
                # Our own retry_sync loop owns backoff so retries are logged.
                max_retries=0,
            )
        return self._sdk


class FakeLLMClient(LLMClient):
    """Scripted client for tests: no network, no key, same accounting path.

    ``responses`` is either a list consumed in order or a callable invoked with
    the request. Each item may be a ``str`` (plain text), a ``dict``
    (``{"text": ..., "tool_calls": [...]}``), or a pydantic model instance,
    which becomes a single tool call carrying that model's fields -- the shape
    :func:`~ml_research_agent.llm.structured.generate_structured` expects.
    """

    reaches_network = False

    def __init__(
        self,
        responses: list[Any] | Callable[..., Any],
        *,
        config: Config | None = None,
        cache: ResponseCache | None = None,
        budget: BudgetTracker | None = None,
        ledger: CostLedger | None = None,
        tracer: Tracer | None = None,
        logger: StructuredLogger | None = None,
        tool_name: str = "emit",
    ) -> None:
        super().__init__(
            config or Config(),
            cache=cache,
            budget=budget,
            ledger=ledger,
            tracer=tracer,
            logger=logger or get_logger("llm.fake"),
        )
        self.responses = responses
        self.tool_name = tool_name
        self.calls: list[dict[str, Any]] = []

    def _invoke(
        self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": messages, "params": params})
        scripted = self._next(model=model, messages=messages, params=params)
        return _payload_from_scripted(scripted, model=model, tool_name=self.tool_name)

    def _next(self, **request: Any) -> Any:
        if callable(self.responses):
            return self.responses(**request)
        index = len(self.calls) - 1
        if index >= len(self.responses):
            # ConfigError, not AgentFailure: a short script is a broken test
            # fixture, and retrying it would just burn the backoff schedule.
            raise ConfigError(
                "FakeLLMClient ran out of scripted responses",
                scripted=len(self.responses),
                requested=index + 1,
            )
        return self.responses[index]

    def _client(self) -> Any:  # pragma: no cover - guards against accidental network use
        raise ConfigError("FakeLLMClient must never open a provider connection")


def supports_sampling_params(model: str) -> bool:
    """Whether ``temperature`` may be sent at all for this model.

    The current Opus/Sonnet 5 generation rejects sampling parameters with a
    400, so the client omits them rather than letting every call fail.
    """
    return not any(model.startswith(prefix) for prefix in _NO_SAMPLING_PARAMS)


def _estimate_tokens(messages: list[dict[str, Any]], system: str | None) -> int:
    chars = len(system or "")
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        else:
            chars += len(str(content))
    return max(chars // _CHARS_PER_TOKEN, 1)


def _payload_from_message(message: Any, *, model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in getattr(message, "content", []) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            text_parts.append(str(getattr(block, "text", "")))
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(getattr(block, "id", "")),
                    "name": str(getattr(block, "name", "")),
                    "arguments": dict(getattr(block, "input", {}) or {}),
                }
            )
    usage = getattr(message, "usage", None)
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "model": str(getattr(message, "model", model)),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "stop_reason": str(getattr(message, "stop_reason", "") or ""),
    }


def _payload_from_scripted(scripted: Any, *, model: str, tool_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": "",
        "tool_calls": [],
        "model": model,
        "input_tokens": 0,
        "output_tokens": 0,
        "stop_reason": "end_turn",
    }
    dumper = getattr(scripted, "model_dump", None)
    if isinstance(scripted, str):
        payload["text"] = scripted
        payload["output_tokens"] = max(len(scripted) // _CHARS_PER_TOKEN, 1)
    elif isinstance(scripted, dict):
        payload.update(scripted)
        payload.setdefault("model", model)
    elif callable(dumper):
        payload["tool_calls"] = [
            {"id": "toolu_fake", "name": tool_name, "arguments": dumper(mode="json")}
        ]
        payload["stop_reason"] = "tool_use"
    else:
        raise ConfigError("unsupported scripted response", kind=type(scripted).__name__)
    payload["tool_calls"] = [
        {
            "id": str(call.get("id", "toolu_fake")),
            "name": str(call.get("name", tool_name)),
            "arguments": dict(call.get("arguments", {})),
        }
        for call in payload.get("tool_calls", [])
    ]
    return payload


def _response_from_payload(
    payload: dict[str, Any], *, tier: str, cached: bool, cost_usd: float
) -> LLMResponse:
    return LLMResponse(
        text=str(payload.get("text", "")),
        tool_calls=[
            ToolCall(id=c["id"], name=c["name"], arguments=dict(c["arguments"]))
            for c in payload.get("tool_calls", [])
        ],
        model=str(payload.get("model", "")),
        tier=tier,
        input_tokens=int(payload.get("input_tokens", 0)),
        output_tokens=int(payload.get("output_tokens", 0)),
        cost_usd=round(cost_usd, 6),
        stop_reason=str(payload.get("stop_reason", "")),
        cached=cached,
        raw=payload,
    )


__all__ = [
    "FakeLLMClient",
    "LLMClient",
    "LLMResponse",
    "ToolCall",
    "supports_sampling_params",
]
