"""Structured generation and the repair loop, exercised without a network."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from ml_research_agent.config import Config
from ml_research_agent.errors import AgentFailure
from ml_research_agent.llm.client import FakeLLMClient
from ml_research_agent.llm.structured import generate_structured, schema_for


class Brief(BaseModel):
    """A tiny stand-in for a real agent output."""

    title: str
    claims: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


def tool_call(arguments: dict[str, Any], *, name: str = "emit") -> dict[str, Any]:
    return {
        "tool_calls": [{"id": "toolu_1", "name": name, "arguments": arguments}],
        "stop_reason": "tool_use",
    }


VALID = {"title": "Does X help?", "claims": ["X improves Y"], "confidence": 0.4}


def test_schema_for_produces_an_anthropic_tool() -> None:
    schema = schema_for(Brief)

    assert schema["name"] == "brief"
    assert "stand-in" in schema["description"]
    assert schema["input_schema"]["type"] == "object"
    assert set(schema["input_schema"]["properties"]) == {"title", "claims", "confidence"}


def test_valid_output_returns_the_model() -> None:
    client = FakeLLMClient([tool_call(VALID)])

    result = generate_structured(client, Brief, messages=[{"role": "user", "content": "go"}])

    assert isinstance(result, Brief)
    assert result.title == "Does X help?"
    assert len(client.calls) == 1


def test_a_pydantic_instance_is_accepted_as_a_scripted_response() -> None:
    client = FakeLLMClient([Brief(**VALID)])

    result = generate_structured(client, Brief, messages=[{"role": "user", "content": "go"}])

    assert result.confidence == 0.4


def test_invalid_output_is_repaired_with_the_specific_errors() -> None:
    invalid = {"title": "T", "claims": [], "confidence": 9.0}
    client = FakeLLMClient([tool_call(invalid), tool_call(VALID)])

    result = generate_structured(client, Brief, messages=[{"role": "user", "content": "go"}])

    assert result.title == "Does X help?"
    assert len(client.calls) == 2

    # The repair turn must name the failing fields, not just say "invalid".
    repair_prompt = client.calls[1]["messages"][-1]["content"]
    assert "claims" in repair_prompt
    assert "confidence" in repair_prompt


def test_a_prose_answer_is_repaired_into_a_tool_call() -> None:
    client = FakeLLMClient(["Sure! Here is the brief: ...", tool_call(VALID)])

    result = generate_structured(client, Brief, messages=[{"role": "user", "content": "go"}])

    assert result.title == "Does X help?"
    assert "tool call" in client.calls[1]["messages"][-1]["content"]


def test_exhausted_repairs_raise_agent_failure() -> None:
    invalid = tool_call({"title": "T", "claims": [], "confidence": 9.0})
    client = FakeLLMClient([invalid, invalid, invalid])

    with pytest.raises(AgentFailure) as excinfo:
        generate_structured(
            client, Brief, messages=[{"role": "user", "content": "go"}], max_repairs=2
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.context["model"] == "Brief"
    assert len(client.calls) == 3


def test_max_repairs_zero_means_one_attempt() -> None:
    client = FakeLLMClient([tool_call({"title": "T", "claims": [], "confidence": 0.1})])

    with pytest.raises(AgentFailure):
        generate_structured(
            client, Brief, messages=[{"role": "user", "content": "go"}], max_repairs=0
        )
    assert len(client.calls) == 1


def test_repair_attempts_default_comes_from_config() -> None:
    config = Config()
    invalid = tool_call({"title": "T", "claims": [], "confidence": 9.0})
    client = FakeLLMClient([invalid] * 10, config=config)

    with pytest.raises(AgentFailure):
        generate_structured(client, Brief, messages=[{"role": "user", "content": "go"}])

    assert len(client.calls) == config.llm.repair_attempts + 1


def test_the_tool_name_is_what_the_system_prompt_demands() -> None:
    client = FakeLLMClient([tool_call(VALID, name="emit_brief")], tool_name="emit_brief")

    generate_structured(
        client,
        Brief,
        messages=[{"role": "user", "content": "go"}],
        system="You are the Framer.",
        tool_name="emit_brief",
    )

    call = client.calls[0]
    assert call["params"]["tools"][0]["name"] == "emit_brief"
    assert call["params"]["tool_choice"] == {"type": "tool", "name": "emit_brief"}
    assert "emit_brief" in call["params"]["system"]
    assert "You are the Framer." in call["params"]["system"]
