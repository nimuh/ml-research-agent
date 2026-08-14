"""Prompts are data: they load from disk, version, render, and fail loudly."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_research_agent.errors import ConfigError
from ml_research_agent.llm.prompts import DEFAULT_ROOT, PromptLibrary


def write(root: Path, name: str, version: str, body: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{version}.md"
    path.write_text(body, encoding="utf-8")
    return path


SIMPLE = """---
system: |
  You are a test prompt.
description: A fixture.
---

Hello {who}, about {topic}.
"""


def test_loads_system_body_and_metadata(tmp_path: Path) -> None:
    write(tmp_path, "tester", "v1", SIMPLE)

    prompt = PromptLibrary(tmp_path).get("tester")

    assert prompt.system == "You are a test prompt."
    assert prompt.template == "Hello {who}, about {topic}."
    assert prompt.metadata["description"] == "A fixture."
    assert prompt.ref == "tester@v1"


def test_render_substitutes_variables(tmp_path: Path) -> None:
    write(tmp_path, "tester", "v1", SIMPLE)

    rendered = PromptLibrary(tmp_path).get("tester").render(who="world", topic="scaling")

    assert rendered == "Hello world, about scaling."


def test_missing_variable_names_itself(tmp_path: Path) -> None:
    write(tmp_path, "tester", "v1", SIMPLE)
    prompt = PromptLibrary(tmp_path).get("tester")

    with pytest.raises(ConfigError) as excinfo:
        prompt.render(who="world")

    assert excinfo.value.context["variable"] == "topic"


def test_latest_version_wins_and_numbers_sort_numerically(tmp_path: Path) -> None:
    for version in ("v1", "v2", "v10"):
        write(tmp_path, "tester", version, SIMPLE.replace("test prompt", f"prompt {version}"))
    library = PromptLibrary(tmp_path)

    assert library.versions("tester") == ["v1", "v2", "v10"]
    assert library.get("tester").version == "v10"
    assert library.get("tester", version="v2").version == "v2"


def test_unknown_prompt_and_version_raise(tmp_path: Path) -> None:
    write(tmp_path, "tester", "v1", SIMPLE)
    library = PromptLibrary(tmp_path)

    with pytest.raises(ConfigError):
        library.get("nobody")
    with pytest.raises(ConfigError):
        library.get("tester", version="v99")


def test_prompt_without_a_system_message_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "broken", "v1", "---\ndescription: no system\n---\n\nbody\n")

    with pytest.raises(ConfigError):
        PromptLibrary(tmp_path).get("broken")


def test_prompt_with_an_empty_body_is_rejected(tmp_path: Path) -> None:
    write(tmp_path, "broken", "v1", "---\nsystem: hi\n---\n\n")

    with pytest.raises(ConfigError):
        PromptLibrary(tmp_path).get("broken")


def test_listing_and_caching(tmp_path: Path) -> None:
    write(tmp_path, "alpha", "v1", SIMPLE)
    write(tmp_path, "beta", "v1", SIMPLE)
    library = PromptLibrary(tmp_path)

    assert library.list() == ["alpha", "beta"]
    assert library.get("alpha") is library.get("alpha")


def test_the_shipped_framer_prompt_loads_and_renders() -> None:
    prompt = PromptLibrary().get("framer", version="v1")

    # The Framer emits a BriefDraft; code assembles the ResearchBrief from it,
    # stamping the id and provenance an LLM must not author.
    assert prompt.metadata["output_model"] == "BriefDraft"
    rendered = prompt.render(
        idea_text="does dropout help tiny transformers?",
        domain="nlp",
        constraints="- one GPU",
        context="none",
    )
    assert "does dropout help tiny transformers?" in rendered
    assert "{" not in rendered  # every placeholder was filled
    assert (DEFAULT_ROOT / "framer" / "v1.md").exists()
