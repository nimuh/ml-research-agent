"""Golden-prompt tests (docs/PLAN.md §7).

Prompts are data, and data drifts away from the code that fills it. These tests
pin the contract between a prompt asset and its agent: the agent supplies
exactly the variables the prompt declares, every declared variable is actually
used, and no prompt can raise at render time. A prompt that only fails in
production -- mid-run, after the budget is spent -- is the failure mode this
file exists to prevent.

Everything here is offline: no LLM call, no network, no key.
"""

from __future__ import annotations

import string
from typing import Any

import pytest
from agent_payloads import payload_for

from ml_research_agent.agents.base import AgentContext, BaseAgent
from ml_research_agent.agents.registry import get_agent_class, registered
from ml_research_agent.config import Config
from ml_research_agent.llm.client import FakeLLMClient
from ml_research_agent.llm.prompts import Prompt, PromptLibrary

LIBRARY = PromptLibrary()
PROMPT_NAMES = LIBRARY.list()
AGENT_NAMES = registered()


def placeholders(template: str) -> set[str]:
    """The ``{name}`` fields ``str.format`` will try to substitute."""
    return {field for _, field, _, _ in string.Formatter().parse(template) if field}


def declared_inputs(prompt: Prompt) -> set[str]:
    return set(prompt.metadata.get("inputs") or [])


def prompt_for_agent(agent_cls: type[BaseAgent[Any, Any]]) -> Prompt:
    return LIBRARY.get(agent_cls.prompt_name or agent_cls.name, version=agent_cls.prompt_version)


@pytest.fixture(scope="module")
def ctx() -> AgentContext:
    """A context with a scripted client -- prompt rendering must not need a model."""
    config = Config()
    return AgentContext(
        config=config,
        llm=FakeLLMClient([], config=config),
        prompts=LIBRARY,
    )


def test_the_library_is_not_empty() -> None:
    assert PROMPT_NAMES, "no prompt assets found; the library root is wrong"


# -- every prompt asset, agent or not ---------------------------------------


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_loads_with_a_system_and_a_body(name: str) -> None:
    prompt = LIBRARY.get(name)

    assert prompt.system.strip(), f"{name}: empty system message"
    assert prompt.template.strip(), f"{name}: empty template body"
    assert prompt.version.startswith("v")


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_every_literal_brace_is_doubled(name: str) -> None:
    """An undoubled brace makes ``render()`` raise mid-run, not at import.

    ``Formatter.parse`` raises on an unbalanced brace, and an undoubled JSON
    example such as ``{"metric": ...}`` survives parsing but yields a field
    name that is not an identifier -- so both checks are needed.
    """
    prompt = LIBRARY.get(name)
    try:
        fields = [field for _, field, _, _ in string.Formatter().parse(prompt.template)]
    except ValueError as exc:  # unbalanced or stray brace
        pytest.fail(
            f"{name}: unbalanced brace in the template body ({exc}); double it as {{{{ }}}}"
        )

    for field in fields:
        if field is None:
            continue
        assert field.isidentifier(), (
            f"{name}: {{{field}}} is not a variable name. If it is literal text "
            "(a JSON example, a code snippet), double the braces."
        )


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_declared_inputs_cover_every_placeholder(name: str) -> None:
    prompt = LIBRARY.get(name)
    undeclared = placeholders(prompt.template) - declared_inputs(prompt)

    assert not undeclared, f"{name}: used but not declared in front matter: {sorted(undeclared)}"


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_declared_inputs_are_all_used(name: str) -> None:
    """A declared-but-unused variable means the prompt was edited and the front
    matter was not -- the same drift, seen from the other side."""
    prompt = LIBRARY.get(name)
    unused = declared_inputs(prompt) - placeholders(prompt.template)

    assert not unused, f"{name}: declared in front matter but never used: {sorted(unused)}"


@pytest.mark.parametrize("name", PROMPT_NAMES)
def test_prompt_renders_from_its_declared_inputs_alone(name: str) -> None:
    prompt = LIBRARY.get(name)
    variables = dict.fromkeys(declared_inputs(prompt), "FIXTURE")

    rendered = prompt.render(**variables)

    assert "FIXTURE" in rendered or not variables


# -- the prompt/agent contract ----------------------------------------------


@pytest.mark.parametrize("agent_name", AGENT_NAMES)
def test_agent_supplies_exactly_the_declared_inputs(agent_name: str, ctx: AgentContext) -> None:
    """The drift check: the agent's real ``prompt_variables()`` against the asset.

    Strict equality in both directions. A missing variable raises ``ConfigError``
    at render time; an extra one means the agent is computing something the
    prompt no longer reads, which is how a prompt silently stops using evidence
    the agent still pays to assemble.
    """
    agent_cls = get_agent_class(agent_name)
    prompt = prompt_for_agent(agent_cls)
    supplied = set(agent_cls().prompt_variables(payload_for(agent_name), ctx))
    declared = declared_inputs(prompt)

    assert supplied == declared, (
        f"{agent_name}: prompt/agent drift. "
        f"supplied but not declared={sorted(supplied - declared)}, "
        f"declared but not supplied={sorted(declared - supplied)}"
    )


@pytest.mark.parametrize("agent_name", AGENT_NAMES)
def test_prompt_renders_with_the_agents_own_variables(agent_name: str, ctx: AgentContext) -> None:
    agent_cls = get_agent_class(agent_name)
    agent = agent_cls()
    prompt = prompt_for_agent(agent_cls)

    system, user = agent.render_prompt(payload_for(agent_name), ctx)

    assert system.strip()
    assert user.strip()
    # Checking the specific `{name}` tokens rather than any brace: rendered
    # values legitimately contain braces (JSON payloads, code excerpts).
    for field in placeholders(prompt.template):
        assert f"{{{field}}}" not in user, f"{agent_name}: {{{field}}} was not substituted"


@pytest.mark.parametrize("agent_name", AGENT_NAMES)
def test_rendered_prompt_carries_the_fixture_content(agent_name: str, ctx: AgentContext) -> None:
    """Substitution must actually happen -- a template of pure boilerplate would
    pass every check above while telling the model nothing about the task."""
    agent_cls = get_agent_class(agent_name)
    variables = agent_cls().prompt_variables(payload_for(agent_name), ctx)
    _, user = agent_cls().render_prompt(payload_for(agent_name), ctx)

    substantive = [str(v) for v in variables.values() if str(v).strip()]
    assert substantive, f"{agent_name}: every prompt variable rendered empty"
    assert any(value[:60] in user for value in substantive), (
        f"{agent_name}: no variable value reached the rendered prompt"
    )


# -- coverage in both directions --------------------------------------------


def test_every_registered_agent_has_a_prompt_asset() -> None:
    missing = [
        name
        for name in AGENT_NAMES
        if (get_agent_class(name).prompt_name or name) not in PROMPT_NAMES
    ]

    assert not missing, f"agents with no prompt asset: {missing}"


def test_every_prompt_asset_belongs_to_an_agent() -> None:
    """Catches an orphan left behind by a rename -- a prompt nothing loads is
    dead weight that still looks maintained."""
    claimed = {get_agent_class(name).prompt_name or name for name in AGENT_NAMES}
    orphans = sorted(set(PROMPT_NAMES) - claimed)

    assert not orphans, f"prompt assets no agent references: {orphans}"


def test_every_agent_declares_its_output_model(ctx: AgentContext) -> None:
    """The front matter's ``output_model`` is documentation; it must not lie."""
    for agent_name in AGENT_NAMES:
        agent_cls = get_agent_class(agent_name)
        declared = prompt_for_agent(agent_cls).metadata.get("output_model")
        if declared is None:
            continue
        assert declared == agent_cls.output_model.__name__, (
            f"{agent_name}: prompt declares output_model={declared} but the agent emits "
            f"{agent_cls.output_model.__name__}"
        )
