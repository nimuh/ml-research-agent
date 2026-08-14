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
import re

import pytest

from ml_research_agent.config import REPO_ROOT

SRC = REPO_ROOT / "src" / "ml_research_agent"
CONFIG = SRC / "config.py"

#: Fields whose only legitimate reader is `config.py` itself. Each needs a
#: reason — an allowlist without one is how this test rots into a rubber stamp.
SELF_READ_ONLY = {
    "api_key_env": "read by Config.api_key / require_api_key, inside config.py by design",
    "tiers": "read by Config.model_for, the accessor llm/client.py calls",
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


def _reader_pattern(field: str) -> re.Pattern[str]:
    """Match a read of `field`, not a longer name that merely contains it.

    The trailing `(?![A-Za-z0-9_])` is what makes this more than a substring
    test: without it `max_tokens` is "read" by any line mentioning
    `max_tokens_per_phase`, and a genuinely unwired knob hiding behind a longer
    name is exactly what this test exists to catch.

    The bare-string-literal alternative is deliberately loose, and it is
    load-bearing: `gates.py` reads its fields as `getattr(config.gates, name)`
    and `config.py` resolves tiers the same way, so for six fields the literal
    `"after_survey"` / `"fast"` *is* the read — there is no attribute access to
    find. Requiring a `["field"]` subscript instead would fail all six.

    The cost of keeping it: a field whose name is also ordinary vocabulary can
    be credited to an unrelated string. A config field named `retry` would be
    counted as read by `runner_agent.py`'s `"retry"` action enum. So this test
    proves a knob is *mentioned* somewhere plausible, not that it changes
    behaviour — it catches the dead-knob class of bug (five real ones so far)
    and does not replace reading the code.
    """
    name = re.escape(field)
    return re.compile(
        # `cfg.field` attribute access, or `"field"` / `'field'` as a key...
        rf"""(?:\.|["']){name}(?![A-Za-z0-9_])"""
        # ...or `field=` as a keyword argument (but not `field ==` a comparison).
        rf"""|(?<![A-Za-z0-9_]){name}\s*=(?!=)"""
    )


def _readers(field: str) -> list[str]:
    """Modules outside `config.py` that mention the field by name."""
    pattern = _reader_pattern(field)
    hits = []
    for path in SRC.rglob("*.py"):
        if path == CONFIG:
            continue
        source = path.read_text(encoding="utf-8")
        for line in source.splitlines():
            if pattern.search(line):
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


def test_a_longer_field_name_does_not_count_as_a_reader() -> None:
    # The failure mode this guard is built to survive: `max_tokens` must not be
    # satisfied by a line that only ever mentions `max_tokens_per_phase`. If this
    # regresses, `test_every_config_field_has_a_reader` still passes while a
    # genuinely unwired knob sits behind a longer name that happens to be used.
    pattern = _reader_pattern("max_tokens")
    assert not pattern.search("limit = cfg.max_tokens_per_phase")
    assert not pattern.search('cfg["max_tokens_per_phase"]')
    assert not pattern.search("max_tokens_per_phase=4")
    # ...while the genuine forms still register.
    assert pattern.search("limit = cfg.max_tokens")
    assert pattern.search('cfg["max_tokens"]')
    assert pattern.search("max_tokens=4")
    # A comparison is not a read of the config knob it names.
    assert not pattern.search("if max_tokens == 4:")


@pytest.mark.parametrize(
    "field", ["after_survey", "before_run", "before_report", "fast", "standard", "deep"]
)
def test_getattr_read_fields_are_still_recognised(field: str) -> None:
    # These six are never accessed as `config.gates.after_survey`; they are read
    # via `getattr(config.gates, name)` with the name supplied as a literal. The
    # string literal is the only trace of the read, which is why
    # `_reader_pattern` accepts a bare literal. Tightening it to require a
    # `["field"]` subscript would fail all six for no gain — this test is here to
    # say so at the point of change rather than leave it to be rediscovered.
    assert _readers(field), (
        f"`{field}` is read dynamically via getattr; if this now fails, the "
        "reader pattern was tightened past what dynamic access can express"
    )


def test_the_allowlist_does_not_hide_a_live_field() -> None:
    # An allowlist that is never re-checked is how this test rots into a rubber
    # stamp: `provider` sat here reasoned as "no runtime reader, a second
    # provider would need one" until `claude_code` added exactly that reader,
    # and nothing noticed. Assert the absence of a reader, not just that the
    # entry is well-formed -- an entry that stops being true has to fail here,
    # which is the whole point of writing the check down.
    for field, reason in SELF_READ_ONLY.items():
        assert field in FIELDS, f"allowlisted `{field}` no longer exists; drop the entry"
        assert reason, "every allowlist entry needs a justification"
        readers = _readers(field)
        assert not readers, (
            f"`{field}` is allowlisted as read only inside config.py, but {readers} now reads it. "
            "The exemption is stale -- delete the entry and let the normal check cover it."
        )
