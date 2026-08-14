"""The dependency rule, enforced instead of documented.

CLAUDE.md states it: platform <- capability layers <- agents <- orchestrator <-
cli, arrows never pointing back up, and capability layers never importing each
other. Every violation of it is cheap to introduce (one convenient import) and
expensive to undo once other code leans on it.

The most load-bearing case is ``literature/``: it returns candidates and never
judges relevance, which is exactly what an LLM import there would quietly end.
Ranking must stay a cheap filter, so that retrieval recall and screening
precision can be tuned independently.

Imports are read from the AST rather than by importing the modules: this sees
lazily-imported names inside functions too, and runs no module side effects.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ml_research_agent

ROOT = Path(ml_research_agent.__file__).parent
PLATFORM = {"llm", "tools", "observability", "utils", "reporting"}
CAPABILITY = {"literature", "knowledge", "code", "experiments"}
UPPER = {"agents", "orchestrator", "cli"}


def _internal_imports(module: Path) -> set[str]:
    """First-level package names this module imports from within the project.

    Relative imports are resolved against the module's own package, so
    ``from ..llm.client import X`` inside ``literature/rank.py`` resolves to
    ``llm``.
    """
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    package_parts = module.relative_to(ROOT).parent.parts
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package_parts[: len(package_parts) - (node.level - 1)])
                target = base + (node.module.split(".") if node.module else [])
            elif node.module and node.module.startswith("ml_research_agent."):
                target = node.module.split(".")[1:]
            else:
                continue
            if target:
                found.add(target[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ml_research_agent."):
                    found.add(alias.name.split(".")[1])
    return found


def _modules(layer: str) -> list[Path]:
    files = sorted((ROOT / layer).rglob("*.py"))
    assert files, f"no modules found for layer {layer!r}"
    return files


@pytest.mark.parametrize("layer", sorted(CAPABILITY))
def test_a_capability_layer_never_imports_another_one(layer: str) -> None:
    forbidden = (CAPABILITY - {layer}) | UPPER
    for module in _modules(layer):
        offending = _internal_imports(module) & forbidden
        assert not offending, (
            f"{module.relative_to(ROOT)} imports {sorted(offending)}; "
            "capability layers are composed by the orchestrator, never by each other"
        )


def test_literature_never_reaches_for_an_llm() -> None:
    """Retrieval is not relevance: screening is the Curator's judgment call."""
    for module in _modules("literature"):
        imports = _internal_imports(module)
        assert "llm" not in imports, f"{module.relative_to(ROOT)} imports llm"
        source = module.read_text(encoding="utf-8")
        assert "anthropic" not in source, f"{module.relative_to(ROOT)} reaches for a model client"


def test_ranking_is_a_cheap_deterministic_filter() -> None:
    """``rank.py`` in particular: it runs over every candidate before screening."""
    source = (ROOT / "literature" / "rank.py").read_text(encoding="utf-8")
    imports = _internal_imports(ROOT / "literature" / "rank.py")

    assert imports <= {"config", "types", "literature", "utils", "errors"}
    for banned in ("llm", "anthropic", "openai", "requests", "httpx"):
        assert banned not in source, f"rank.py mentions {banned}"


def test_nothing_queries_the_index_around_the_single_retrieval_entrypoint() -> None:
    """Agents retrieve through ``search.py`` and nowhere else.

    Constructing a ``HybridIndex`` is composition and the orchestrator does it;
    *querying* one outside knowledge/ is the thing that would leave retrieval
    uninstrumented and unevaluable.
    """
    for module in sorted(ROOT.rglob("*.py")):
        if module.relative_to(ROOT).parts[0] == "knowledge":
            continue
        source = module.read_text(encoding="utf-8")
        for call in (".vector_search(", ".keyword_search("):
            assert call not in source, (
                f"{module.relative_to(ROOT)} calls {call.strip('.(')} directly; "
                "all retrieval goes through knowledge/search.py"
            )


def test_only_the_orchestrator_builds_the_index_itself() -> None:
    for layer in sorted((CAPABILITY - {"knowledge"}) | {"agents", "reporting"}):
        target = ROOT / layer
        if not target.exists():
            continue
        for module in sorted(target.rglob("*.py")):
            assert "HybridIndex" not in module.read_text(encoding="utf-8"), (
                f"{module.relative_to(ROOT)} builds the index; "
                "the orchestrator composes it and passes it in"
            )


def test_the_platform_never_imports_upward() -> None:
    for layer in sorted(PLATFORM):
        for module in _modules(layer):
            offending = _internal_imports(module) & (CAPABILITY | UPPER)
            assert not offending, (
                f"{module.relative_to(ROOT)} imports {sorted(offending)}; "
                "the platform is the bottom of the stack"
            )


def test_the_orchestrator_may_compose_everything_below_it() -> None:
    """The positive control: the rule above must not be vacuously satisfied."""
    seen: set[str] = set()
    for module in _modules("orchestrator"):
        seen |= _internal_imports(module)
    assert seen & CAPABILITY, "the orchestrator composes the capability layers"
    assert "agents" in seen
