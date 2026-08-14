"""KnowledgeStore: CRUD, idempotency, note history, stats."""

from __future__ import annotations

import pytest
from kb_helpers import kb_config, synthetic_papers

from ml_research_agent.errors import MRAError
from ml_research_agent.knowledge.chunk import chunk_sections
from ml_research_agent.knowledge.store import KnowledgeStore, paper_content_hash
from ml_research_agent.types import (
    Claim,
    ClaimRelation,
    Evidence,
    EvidenceKind,
    GraphEdge,
    Note,
    Provenance,
)


@pytest.fixture
def store(tmp_path):
    config = kb_config(tmp_path)
    with KnowledgeStore(config) as opened:
        yield opened


def _note(**overrides) -> Note:
    fields = {
        "title": "Sparse attention",
        "summary": "A sparse attention pattern with linear cost.",
        "paper_key": "arxiv:2101.00001",
        "provenance": [Provenance(source="arxiv:2101.00001", locator="§Method p.2", quote="mask")],
        "confidence": 0.6,
    }
    fields.update(overrides)
    return Note(**fields)


def test_documents_round_trip_and_upsert_by_key(store):
    record = store.add_document(
        kind="web",
        key="blog/post",
        title="A post",
        path=None,
        content_hash="abc",
        metadata={"a": 1},
    )
    assert store.get_document(record.id) == record
    assert store.document_by_hash("abc") == record
    assert store.document_by_hash("nope") is None

    again = store.add_document(
        kind="web", key="blog/post", title="A post, revised", path=None, content_hash="def"
    )
    assert again.id == record.id
    assert len(store.list_documents(kind="web")) == 1
    assert store.get_document(record.id).title == "A post, revised"


def test_papers_upsert_by_key_and_survive_the_round_trip(store):
    paper = synthetic_papers()[0]
    doc_id = store.add_paper(paper)
    assert store.add_paper(paper) == doc_id
    assert len(store.list_papers()) == 1

    loaded = store.get_paper(paper.key)
    assert loaded is not None
    assert loaded.title == paper.title
    assert [s.title for s in loaded.sections] == [s.title for s in paper.sections]
    assert store.paper_for_doc(doc_id).key == paper.key
    assert store.get_paper("arxiv:nope") is None


def test_paper_content_hash_ignores_identity_but_not_content():
    a, b = synthetic_papers()[0], synthetic_papers()[0]
    assert a.id != b.id
    assert paper_content_hash(a) == paper_content_hash(b)
    assert paper_content_hash(a.model_copy(update={"abstract": "changed"})) != paper_content_hash(a)


def test_passages_round_trip_and_upsert(store):
    paper = synthetic_papers()[0]
    doc_id = store.add_paper(paper)
    passages = chunk_sections(paper.sections, doc_id, store.config.knowledge)
    assert store.add_passages(passages) == len(passages)
    assert store.add_passages(passages) == len(passages)
    assert len(store.list_passages()) == len(passages)

    fetched = store.get_passage(passages[0].id)
    assert fetched == passages[0]
    assert store.get_passage("psg_missing") is None
    assert store.list_passages(doc_id="doc_other") == []
    assert store.delete_passages(doc_id=doc_id) == len(passages)
    assert store.list_passages() == []


def test_notes_round_trip_through_the_markdown_file(store):
    note = _note()
    path = store.save_note(note)
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("---")
    assert store.get_note(note.id) == note
    assert store.get_note("note_missing") is None
    assert [n.id for n in store.list_notes(type="paper")] == [note.id]
    assert store.list_notes(type="concept") == []


def test_a_note_whose_markdown_vanished_is_an_error_not_a_silent_none(store):
    # The Markdown is truth and SQLite is the index over it; a row pointing at a
    # file that is gone must be loud, or the KB quietly forgets a note it lists.
    note = _note()
    path = store.save_note(note)
    path.unlink()
    with pytest.raises(MRAError, match="wiki is truth"):
        store.get_note(note.id)


def test_supersede_keeps_the_old_note_readable(store):
    old = _note()
    old_path = store.save_note(old)
    new = _note(summary="Revised: the gain shrinks at longer context.", confidence=0.8)

    successor = store.supersede_note(old.id, new)
    assert successor.supersedes == old.id
    assert store.superseded_by(old.id) == successor.id
    assert old_path.exists()
    assert store.get_note(old.id) == old
    assert {n.id for n in store.list_notes()} == {old.id, successor.id}
    assert [n.id for n in store.list_notes(include_superseded=False)] == [successor.id]


def test_superseding_an_unknown_note_is_an_error(store):
    with pytest.raises(Exception, match="supersede"):
        store.supersede_note("note_missing", _note())


def test_claims_and_edges(store):
    claim = Claim(
        statement="sparse attention matches dense attention on long context",
        evidence=[
            Evidence(
                kind=EvidenceKind.PASSAGE,
                provenance=Provenance(source="arxiv:2101.00001", locator="§Results p.5"),
            )
        ],
    )
    assert store.add_claim(claim) == claim.id
    assert store.get_claim(claim.id) == claim
    assert store.list_claims() == [claim]
    assert store.get_claim("claim_missing") is None

    edge = GraphEdge(source="claim:a", target="claim:b", relation=ClaimRelation.CONTRADICTS)
    store.add_edge(edge)
    store.add_edge(edge)  # same typed edge upserts rather than duplicating
    assert store.list_edges() == [edge]
    assert store.list_edges(relation=ClaimRelation.CONTRADICTS) == [edge]
    assert store.list_edges(relation=ClaimRelation.CITES) == []


def test_stats_counts_what_the_cli_reports(store):
    paper = synthetic_papers()[0]
    doc_id = store.add_paper(paper)
    store.add_passages(chunk_sections(paper.sections, doc_id, store.config.knowledge))
    store.save_note(_note())
    store.log_ingest(
        content_hash="abc",
        doc_id=doc_id,
        kind="paper",
        key=paper.key,
        passages=2,
        status="ingested",
    )

    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["documents_by_kind"] == {"paper": 1}
    assert stats["passages"] > 0
    assert stats["passage_tokens"] > 0
    assert stats["notes"] == 1
    assert stats["notes_by_type"] == {"paper": 1}
    assert stats["last_ingest_at"] is not None
    assert stats["sqlite_bytes"] > 0
    assert stats["schema_version"] >= 1


def test_reopening_the_store_sees_everything(tmp_path):
    config = kb_config(tmp_path)
    with KnowledgeStore(config) as first:
        first.add_paper(synthetic_papers()[0])
        first.save_note(_note())
    with KnowledgeStore(config) as second:
        assert len(second.list_papers()) == 1
        assert len(second.list_notes()) == 1
