"""`mra kb search|show|stats|health` end to end against a seeded tmp workspace."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from kb_helpers import synthetic_papers
from typer.testing import CliRunner

from ml_research_agent.cli import app
from ml_research_agent.config import Config
from ml_research_agent.knowledge.embed import get_embedder
from ml_research_agent.knowledge.index import HybridIndex
from ml_research_agent.knowledge.ingest import Ingestor
from ml_research_agent.knowledge.store import KnowledgeStore
from ml_research_agent.types import Claim, Note, Provenance

runner = CliRunner()


def _write_config(root: Path) -> Path:
    """A YAML config the CLI can load, with every path under ``root``.

    Each sub-path is written out explicitly: `configs/default.yaml` sets them
    all, so overriding only `workspace` would leave `kb_wiki` and friends
    pointing at the repo's real workspace.
    """
    workspace = root / "workspace"
    config_path = root / "mra.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "workspace": str(workspace),
                    "kb_raw": str(workspace / "kb" / "raw"),
                    "kb_wiki": str(workspace / "kb" / "wiki"),
                    "kb_reports": str(workspace / "kb" / "reports"),
                    "runs": str(workspace / "runs"),
                    "repos": str(workspace / "repos"),
                    "cache": str(workspace / "cache"),
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def kb(tmp_path) -> tuple[Path, Config, str]:
    """A config file the CLI can load, plus a small KB already sitting behind it."""
    config_path = _write_config(tmp_path)
    config = Config.load(config_file=config_path).ensure_paths()
    assert config.paths.kb_wiki.is_relative_to(tmp_path)

    store = KnowledgeStore(config)
    index = HybridIndex(config.paths.kb_index, get_embedder(config), config.knowledge)
    ingestor = Ingestor(config, store, index)
    note_id = ""
    for paper in synthetic_papers()[:3]:
        result = ingestor.ingest_paper(paper)
        stub = result.note_stub
        assert stub is not None
        store.save_note(stub)
        note_id = note_id or stub.id
    store.close()
    return config_path, config, note_id


def _run(config_path: Path, *args: str):
    result = runner.invoke(app, [*args, "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    return result.output


def test_kb_search_prints_cited_passages(kb):
    config_path, _config, _note_id = kb
    output = _run(config_path, "kb", "search", "sparse attention long range arena", "-k", "3")

    assert "Sato" in output
    assert "score=" in output
    assert "sparse" in output.lower()


def test_kb_search_below_the_relevance_floor_reports_no_hits(kb):
    config_path, _config, _note_id = kb
    # The KB is full; this query simply has nothing behind it. Without the floor
    # the nearest neighbour comes back at ~0.008 and reads like a real citation.
    assert "no hits" in _run(config_path, "kb", "search", "zzzz qqqq wwww")
    assert "score=" in _run(config_path, "kb", "search", "zzzz qqqq wwww", "--min-score", "0")


def test_kb_search_on_an_empty_kb_says_so_rather_than_crashing(tmp_path):
    config_path = _write_config(tmp_path / "empty")
    Config.load(config_file=config_path).ensure_paths()
    assert "no hits" in _run(config_path, "kb", "search", "anything at all")


def test_kb_search_honours_k(kb):
    config_path, _config, _note_id = kb
    query = "sparse attention training accuracy"
    assert _run(config_path, "kb", "search", query, "-k", "1").count("score=") == 1
    assert _run(config_path, "kb", "search", query, "-k", "3").count("score=") == 3


def test_kb_show_prints_the_note_as_stored(kb):
    config_path, config, note_id = kb
    output = _run(config_path, "kb", "show", note_id)

    assert "---" in output  # front matter
    assert note_id in output
    assert "Summary" in output


def test_kb_show_fails_loudly_on_an_unknown_note(kb):
    config_path, _config, _note_id = kb
    result = runner.invoke(app, ["kb", "show", "note_missing", "--config", str(config_path)])
    assert result.exit_code == 1
    assert "no such note" in result.output


def test_kb_stats_reports_the_counts(kb):
    config_path, _config, _note_id = kb
    output = _run(config_path, "kb", "stats")

    assert "documents" in output
    assert "passages" in output
    assert "notes" in output
    assert "schema_version" in output


def test_kb_health_renders_the_report(kb):
    config_path, config, _note_id = kb
    output = _run(config_path, "kb", "health")

    assert "KB health" in output
    assert "Stats" in output
    assert not list(config.paths.kb_wiki.glob("todo-*.md"))


def test_kb_health_can_write_todos(kb):
    config_path, config, _note_id = kb
    with KnowledgeStore(config) as store:
        store.add_claim(Claim(statement="nothing backs this"))
        store.save_note(Note(type="method", title="Floating", summary="No provenance."))

    before = set(config.paths.kb_wiki.glob("*.md"))
    output = _run(config_path, "kb", "health", "--write-todos")
    after = set(config.paths.kb_wiki.glob("*.md"))

    assert "wrote" in output
    assert after - before
    with KnowledgeStore(config) as store:
        todos = store.list_notes(type="todo")
    assert todos
    assert all(note.title.startswith("TODO:") for note in todos)


def test_kb_health_flags_a_note_whose_provenance_is_missing(kb):
    config_path, config, _note_id = kb
    with KnowledgeStore(config) as store:
        store.save_note(
            Note(
                type="paper",
                title="Ghost",
                summary="Points at a paper the KB does not have.",
                paper_key="arxiv:9999.99999",
                provenance=[Provenance(source="arxiv:9999.99999", locator="§1")],
                confidence=0.5,
            )
        )
    output = _run(config_path, "kb", "health")
    assert "orphan_note" in output


def test_the_cli_never_touches_the_real_workspace(kb):
    config_path, config, note_id = kb
    for args in (
        ("kb", "search", "sparse attention"),
        ("kb", "show", note_id),
        ("kb", "stats"),
        ("kb", "health"),
    ):
        _run(config_path, *args)
    assert (config.paths.workspace / "kb" / "kb.sqlite").exists()
    assert config.paths.workspace.is_relative_to(config_path.parent)
