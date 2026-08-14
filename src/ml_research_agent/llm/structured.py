"""Schema-constrained generation: pydantic model -> tool schema, validation,
and a repair loop that re-prompts on invalid output."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import AgentFailure
from ..types import ModelTier
from .client import LLMClient

T = TypeVar("T", bound=BaseModel)


def schema_for(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Turn a pydantic model into an Anthropic tool definition."""
    schema = model_cls.model_json_schema()
    schema.pop("title", None)
    description = (model_cls.__doc__ or f"Emit a {model_cls.__name__}.").strip()
    return {
        "name": _tool_name(model_cls.__name__),
        "description": description[:1024],
        "input_schema": schema,
    }


def generate_structured(
    client: LLMClient,
    output_model: type[T],
    *,
    messages: list[dict[str, Any]],
    system: str | None = None,
    tier: ModelTier | str = ModelTier.STANDARD,
    phase: str | None = None,
    agent: str | None = None,
    max_repairs: int | None = None,
    tool_name: str = "emit",
) -> T:
    """Force one tool call shaped like ``output_model`` and validate it.

    On :class:`ValidationError` the *specific* errors are appended to the
    conversation and the model is asked again. Feeding back the exact field
    paths is what makes the repair loop converge -- "that was invalid" alone
    does not.
    """
    tool = schema_for(output_model)
    tool["name"] = tool_name
    tools = [tool]
    tool_choice_hint = (
        f"Reply by calling the `{tool_name}` tool exactly once with the required fields. "
        "Do not answer in prose."
    )
    system_prompt = f"{system}\n\n{tool_choice_hint}" if system else tool_choice_hint

    attempts = client.config.llm.repair_attempts if max_repairs is None else max_repairs
    conversation = list(messages)
    last_error: str = ""

    for attempt in range(max(attempts, 0) + 1):
        response = client.complete(
            messages=conversation,
            tier=tier,
            system=system_prompt,
            tools=tools,
            # Force the tool at the API level. The system-prompt instruction
            # above is the fallback, not the primary mechanism.
            tool_choice={"type": "tool", "name": tool_name},
            phase=phase,
            agent=agent,
        )
        call = response.tool_call(tool_name) or (
            response.tool_calls[0] if response.tool_calls else None
        )
        if call is None:
            last_error = (
                f"No `{tool_name}` tool call was made. The entire answer must be the tool call."
            )
            conversation = _with_repair(conversation, response.text, last_error)
            continue
        try:
            return output_model.model_validate(call.arguments)
        except ValidationError as exc:
            last_error = _format_errors(exc)
            client.logger.warning(
                "structured output failed validation; repairing",
                model=output_model.__name__,
                phase=phase,
                agent=agent,
                attempt=attempt + 1,
                errors=last_error[:500],
            )
            conversation = _with_repair(conversation, json.dumps(call.arguments)[:4000], last_error)

    raise AgentFailure(
        "structured output did not validate after repair attempts",
        retryable=True,
        model=output_model.__name__,
        attempts=max(attempts, 0) + 1,
        phase=phase,
        agent=agent,
        errors=last_error[:1000],
    )


def _with_repair(
    conversation: list[dict[str, Any]], attempted: str, errors: str
) -> list[dict[str, Any]]:
    """Append the bad output and the validator's complaint as a new turn."""
    return [
        *conversation,
        {"role": "assistant", "content": attempted or "(no output)"},
        {
            "role": "user",
            "content": (
                "That output was rejected by the schema validator:\n"
                f"{errors}\n\n"
                "Emit the tool call again, corrected. Change only what the errors require."
            ),
        },
    ]


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "(root)"
        lines.append(f"- {location}: {error.get('msg', 'invalid')}")
    return "\n".join(lines)


def _tool_name(class_name: str) -> str:
    out: list[str] = []
    for index, char in enumerate(class_name):
        if char.isupper() and index > 0:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


__all__ = ["generate_structured", "schema_for"]
