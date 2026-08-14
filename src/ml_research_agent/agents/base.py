"""BaseAgent: shared loop (prompt -> tool calls -> structured output -> validate
-> retry/repair), tool allow-listing, token/cost accounting, and tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from ..config import Config
from ..errors import AgentFailure
from ..llm.client import LLMClient
from ..llm.prompts import PromptLibrary
from ..llm.structured import generate_structured, schema_for
from ..observability.logging import StructuredLogger, get_logger
from ..observability.tracing import Tracer
from ..tools.registry import ToolRegistry
from ..types import ModelTier, Phase

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

EMIT_TOOL = "emit_result"


@runtime_checkable
class RetrieverLike(Protocol):
    """The slice of ``knowledge.search.Retriever`` agents are allowed to use.

    Structural, not nominal: agents depend on the retrieval *contract*, which
    keeps the single-retrieval-entrypoint invariant testable with a fake.
    """

    def search(self, query: str, **kwargs: Any) -> Any: ...


@dataclass
class AgentContext:
    """Everything an agent is allowed to touch.

    Agents are stateless: this is handed in per call and never stored on the
    agent. Note what is absent -- no reference to another agent. All
    coordination goes through the orchestrator and ProjectState.
    """

    config: Config
    llm: LLMClient
    prompts: PromptLibrary
    tools: ToolRegistry | None = None
    retriever: RetrieverLike | None = None
    logger: StructuredLogger | None = None
    tracer: Tracer | None = None
    phase: Phase | None = None
    project_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def child(self, **overrides: Any) -> AgentContext:
        data = {**self.__dict__, **overrides}
        return AgentContext(**data)


class BaseAgent(Generic[InputT, OutputT]):
    """One narrow role with a typed contract: ``run(input, ctx) -> output``.

    Subclasses declare their prompt, model tier, tool allow-list and output
    model, then optionally override :meth:`build_context_block` to inject
    retrieved evidence. The loop itself lives here so every agent gets the same
    validation, repair, budgeting and tracing behavior.
    """

    name: ClassVar[str] = "agent"
    description: ClassVar[str] = ""
    tier: ClassVar[ModelTier] = ModelTier.STANDARD
    prompt_name: ClassVar[str] = ""
    prompt_version: ClassVar[str | None] = None
    tool_allowlist: ClassVar[tuple[str, ...]] = ()
    max_tool_turns: ClassVar[int] = 6
    output_model: ClassVar[type[BaseModel]]

    def __init__(self, *, tier: ModelTier | None = None) -> None:
        self._tier_override = tier

    # -- the pieces subclasses customize ------------------------------------

    def render_prompt(self, payload: InputT, ctx: AgentContext) -> tuple[str, str]:
        """Return ``(system, user)``. Prompts are data: loaded, never inlined."""
        prompt = ctx.prompts.get(self.prompt_name or self.name, version=self.prompt_version)
        variables = self.prompt_variables(payload, ctx)
        return prompt.system, prompt.render(**variables)

    def prompt_variables(self, payload: InputT, ctx: AgentContext) -> dict[str, Any]:
        """Variables substituted into the prompt template."""
        return {"input": payload.model_dump_json(indent=2), **self.extra_variables(payload, ctx)}

    def extra_variables(self, payload: InputT, ctx: AgentContext) -> dict[str, Any]:
        return {}

    def postprocess(self, output: OutputT, payload: InputT, ctx: AgentContext) -> OutputT:
        """Hook for deterministic fixups and invariant enforcement after generation."""
        return output

    def validate_output(self, output: OutputT, payload: InputT, ctx: AgentContext) -> list[str]:
        """Return violations. Non-empty triggers one repair round, then AgentFailure.

        This is where an agent enforces *its own* domain invariants -- the ones
        pydantic cannot express, like "every claim carries provenance".
        """
        return []

    # -- the loop ------------------------------------------------------------

    def run(self, payload: InputT, ctx: AgentContext) -> OutputT:
        logger = (ctx.logger or get_logger("agent")).bind(
            agent=self.name, phase=_phase_value(ctx.phase)
        )
        tier = self._tier_override or self.tier
        system, user = self.render_prompt(payload, ctx)
        tracer = ctx.tracer

        span_ctx = tracer.span(f"agent:{self.name}", tier=str(tier)) if tracer else _null_span()
        with span_ctx:
            logger.info("agent_started", tier=str(tier), tools=list(self.tool_allowlist))
            messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

            if self.tool_allowlist and ctx.tools is not None:
                messages = self._tool_loop(messages, system, tier, ctx, logger)

            output = generate_structured(
                ctx.llm,
                self.output_model,
                messages=messages,
                system=system,
                tier=tier,
                phase=_phase_value(ctx.phase),
                agent=self.name,
            )
            typed = self.postprocess(output, payload, ctx)  # type: ignore[arg-type]

            violations = self.validate_output(typed, payload, ctx)
            if violations:
                logger.warning("agent_output_invalid", violations=violations)
                typed = self._repair(messages, system, tier, ctx, violations, payload)

            logger.info("agent_completed", output_type=type(typed).__name__)
            return typed

    def _repair(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tier: ModelTier,
        ctx: AgentContext,
        violations: list[str],
        payload: InputT,
    ) -> OutputT:
        """One more attempt, with the specific violations named.

        Naming the violation is what makes repair work -- "invalid output" gets
        the same invalid output back.
        """
        complaint = "\n".join(f"- {v}" for v in violations)
        repair_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "Your previous output violated these requirements:\n"
                    f"{complaint}\n\n"
                    "Produce a corrected result that satisfies all of them. Do not drop content "
                    "to satisfy a constraint -- fix it."
                ),
            },
        ]
        output = generate_structured(
            ctx.llm,
            self.output_model,
            messages=repair_messages,
            system=system,
            tier=tier,
            phase=_phase_value(ctx.phase),
            agent=self.name,
        )
        typed = self.postprocess(output, payload, ctx)  # type: ignore[arg-type]
        remaining = self.validate_output(typed, payload, ctx)
        if remaining:
            raise AgentFailure(
                "agent could not produce valid output after repair",
                agent=self.name,
                violations=remaining,
            )
        return typed

    def _tool_loop(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tier: ModelTier,
        ctx: AgentContext,
        logger: StructuredLogger,
    ) -> list[dict[str, Any]]:
        """Let the agent gather evidence before it is asked to commit to output.

        Bounded by ``max_tool_turns`` -- an agent that will not stop researching
        is a cost bug, not diligence.
        """
        assert ctx.tools is not None
        schemas = ctx.tools.schemas(self.tool_allowlist)
        if not schemas:
            return messages

        working = list(messages)
        for turn in range(self.max_tool_turns):
            response = ctx.llm.complete(
                messages=working,
                system=system,
                tools=schemas,
                tier=tier,
                phase=_phase_value(ctx.phase),
                agent=self.name,
            )
            if not response.tool_calls:
                if response.text:
                    working.append({"role": "assistant", "content": response.text})
                break

            working.append(
                {
                    "role": "assistant",
                    "content": _assistant_tool_content(response.text, response.tool_calls),
                }
            )
            results: list[dict[str, Any]] = []
            for call in response.tool_calls:
                result = ctx.tools.call(
                    call.name, call.arguments, agent=self.name, allow=self.tool_allowlist
                )
                logger.info("tool_call", tool=call.name, ok=result.ok, turn=turn)
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": _stringify(result.output if result.ok else result.error),
                        "is_error": not result.ok,
                    }
                )
            working.append({"role": "user", "content": results})
        else:
            logger.warning("tool_loop_exhausted", turns=self.max_tool_turns)

        working.append(
            {
                "role": "user",
                "content": "Now produce your final structured result based on what you gathered.",
            }
        )
        return working

    # -- introspection -------------------------------------------------------

    @classmethod
    def output_schema(cls) -> dict[str, Any]:
        return schema_for(cls.output_model)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} tier={self.tier}>"


def _phase_value(phase: Phase | None) -> str | None:
    return phase.value if phase is not None else None


def _assistant_tool_content(text: str, calls: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for call in calls:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return blocks


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value[:20000]
    import json

    try:
        return json.dumps(value, default=str)[:20000]
    except (TypeError, ValueError):
        return str(value)[:20000]


class _null_span:
    """Context manager used when tracing is disabled."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> Literal[False]:
        return False


__all__ = ["EMIT_TOOL", "AgentContext", "BaseAgent", "RetrieverLike"]
