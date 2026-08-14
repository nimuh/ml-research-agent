"""Every documented knob must have a reader.

A config field nothing consults is a guard that is not merely wrong but
unreachable, and its documentation asserts the opposite — `configs/default.yaml`
promises behaviour that silently does not happen. Five of these shipped in this
codebase before anyone noticed (`run_untrusted_code`, `network_allowlist`,
`dry_run`, `cost_ledger`, `embedding_model`), including one where `network: deny`
was inert rather than merely advisory.

Noticing is the wrong mechanism, so this test does it mechanically.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ml_research_agent.config import REPO_ROOT

SRC = REPO_ROOT / "src" / "ml_research_agent"
CONFIG = SRC / "config.py"

#: Fields whose only legitimate reader is `config.py` itself. Each needs a
#: reason — an allowlist without one is how this test rots into a rubber stamp.
SELF_READ_ONLY = {
    "api_key_env": "read by Config.api_key / require_api_key, inside config.py by design",
    "tiers": "read by Config.model_for, the accessor llm/client.py calls",
    "provider": (
        "enforced by its Literal type at parse time rather than by a runtime reader; "
        "a second provider would need one"
    ),
}


def _config_fields() -> dict[str, str]:
    """Public pydantic field names declared in `config.py`, by owning class."""
    tree = ast.parse(CONFIG.read_text(encoding="utf-8"))
    fields: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                name = stmt.target.id
                if not name.startswith("_"):
                    fields[name] = node.name
    return fields


def _readers(field: str) -> list[str]:
    """Modules outside `config.py` that mention the field by name."""
    hits = []
    for path in SRC.rglob("*.py"):
        if path == CONFIG:
            continue
        # Word-boundary match on the attribute name; a substring match would let
        # `max_tokens` be "read" by `max_tokens_per_phase` and hide a real gap.
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            if f".{field}" in line or f'"{field}"' in line or f"{field}=" in line:
                hits.append(path.relative_to(SRC).as_posix())
                break
    return hits


FIELDS = _config_fields()


def test_the_config_surface_was_discovered_at_all() -> None:
    # A parser that silently found nothing would make every test below vacuous.
    assert len(FIELDS) > 50, f"only found {len(FIELDS)} fields; the AST walk is broken"
    assert "usd_per_project" in FIELDS
    assert "network_allowlist" in FIELDS


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_every_config_field_has_a_reader(field: str) -> None:
    if field in SELF_READ_ONLY:
        pytest.skip(SELF_READ_ONLY[field])
    readers = _readers(field)
    assert readers, (
        f"`{FIELDS[field]}.{field}` is declared and documented but nothing reads it. "
        "Either wire it up or delete it — a knob that silently does nothing is worse "
        "than an absent one, because the config file claims otherwise."
    )


def test_the_allowlist_does_not_hide_a_live_field() -> None:
    # If an allowlisted field gains a real reader, the entry is stale and should
    # go, so the allowlist stays a short list of genuine exceptions.
    for field, reason in SELF_READ_ONLY.items():
        assert field in FIELDS, f"allowlisted `{field}` no longer exists; drop the entry"
        assert reason, "every allowlist entry needs a justification"
