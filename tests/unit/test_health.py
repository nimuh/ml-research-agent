"""KB health: what it catches, and the TODOs it leaves behind."""

from __future__ import annotations

from datetime import timedelta

from kb_helpers import kb_config, synthetic_papers

from ml_research_agent.knowledge.chunk import chunk_sections
from ml_research_agent.knowledge.embed import get_embedder
from ml_research_agent.knowledge.health import STALE_AFTER_DAYS, KnowledgeHealth
from ml_research_agent.knowledge.index import HybridIndex
from ml_research_agent.knowledge.schema import note_from_markdown
from ml_research_agent.knowledge.store import KnowledgeStore
from ml_research_agent.types import Claim, Evidence, EvidenceKind, Note, Provenance, utcnow

PROV = Provenance(source="arxiv:2101.00001", locator="§Results p.5")


def _healthy_kb(tmp_path):
    config = kb_config(tmp_path)
    store = KnowledgeStore(config)
    index = HybridIndex(config.paths.kb_index, get_embedder(config), config.knowledge)
    for paper in synthetic_papers()[:2]:
        doc_id = store.add_paper(paper)
        passages = chunk_sections(paper.sections, doc_id, config.knowledge)
        store.add_passages(passages)
        index.upsert(passages)
        store.save_note(
            Note(
                type="paper",
                title=paper.title,
                summary=paper.abstract,
                paper_key=paper.key,
                method="sparse attention",
                datasets=["long range arena"],
                provenance=[Provenance(source=paper.key, locator="§Method p.1")],
                confidence=0.6,
            )
        )
    index.save()
    return config, store, index


def _kinds(report) -> set[str]:
    return {issue.kind for issue in report.issues}


def test_a_healthy_kb_reports_no_errors(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    report = KnowledgeHealth(config, store).check()
    assert report.errors == []
    assert report.ok
    assert "index_drift" not in _kinds(report)
    assert report.stats["notes"] == 2
    store.close()


def test_a_note_without_provenance_is_an_error(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    store.save_note(Note(type="method", title="Floating claim", summary="Nothing backs this."))
    report = KnowledgeHealth(config, store).check()
    assert "invalid_note" in _kinds(report)
    assert any("provenance" in issue.message for issue in report.errors)
    store.close()


def test_a_note_pointing_at_a_paper_the_kb_lacks_is_an_orphan(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    store.save_note(
        Note(
            type="paper",
            title="Ghost paper",
            summary="Cites a paper nobody ingested.",
            paper_key="arxiv:9999.99999",
            provenance=[PROV],
            confidence=0.4,
        )
    )
    report = KnowledgeHealth(config, store).check()
    assert "orphan_note" in _kinds(report)
    store.close()


def test_an_unbacked_claim_is_an_error(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    store.add_claim(Claim(statement="asserted by nobody"))
    store.add_claim(
        Claim(
            statement="vaguely sourced",
            evidence=[
                Evidence(
                    kind=EvidenceKind.PASSAGE, provenance=Provenance(source="arxiv:2101.00001")
                )
            ],
        )
    )
    report = KnowledgeHealth(config, store).check()
    assert "uncited_claim" in _kinds(report)
    assert any(i.severity == "error" for i in report.issues if i.kind == "uncited_claim")
    assert any(i.severity == "warning" for i in report.issues if i.kind == "uncited_claim")
    store.close()


def test_index_drift_is_reported_in_both_directions(tmp_path):
    config, store, index = _healthy_kb(tmp_path)
    paper = synthetic_papers()[2]
    doc_id = store.add_paper(paper)
    store.add_passages(chunk_sections(paper.sections, doc_id, config.knowledge))

    report = KnowledgeHealth(config, store).check()
    drift = [i for i in report.issues if i.kind == "index_drift"]
    assert len(drift) == 1
    assert "not indexed" in drift[0].message

    index.rebuild(store)
    index.remove([])
    assert not [i for i in KnowledgeHealth(config, store).check().issues if i.kind == "index_drift"]

    store.delete_passages(doc_id=doc_id)
    stale = [i for i in KnowledgeHealth(config, store).check().issues if i.kind == "index_drift"]
    assert any("missing from the store" in i.message for i in stale)
    store.close()


def test_a_document_with_no_passages_cannot_be_cited(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    store.add_document(kind="web", key="page", title="Empty", path=None, content_hash="zz")
    report = KnowledgeHealth(config, store).check()
    assert "unchunked_document" in _kinds(report)
    store.close()


def test_stale_notes_are_information_not_errors(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    old = Note(
        type="paper",
        title="Old reading",
        summary="Written long ago.",
        paper_key=synthetic_papers()[0].key,
        provenance=[PROV],
        confidence=0.5,
        added_at=utcnow() - timedelta(days=STALE_AFTER_DAYS + 1),
    )
    store.save_note(old)
    report = KnowledgeHealth(config, store).check()
    stale = [i for i in report.issues if i.kind == "stale_note"]
    assert [i.target for i in stale] == [old.id]
    assert stale[0].severity == "info"
    store.close()


def test_contradictions_and_gaps_come_through_the_graph(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    a = Claim(
        statement="sparse attention matches dense",
        evidence=[Evidence(kind=EvidenceKind.PASSAGE, provenance=PROV)],
    )
    b = Claim(
        statement="sparse attention loses accuracy",
        evidence=[Evidence(kind=EvidenceKind.PASSAGE, provenance=PROV)],
        contradicts=[a.id],
    )
    store.add_claim(a)
    store.add_claim(b)
    store.save_note(
        Note(
            type="paper",
            title="Quantization",
            summary="Eight bit training.",
            paper_key=synthetic_papers()[2].key,
            method="quantization",
            datasets=["imagenet"],
            provenance=[PROV],
            confidence=0.5,
        )
    )
    report = KnowledgeHealth(config, store).check()
    assert "contradiction" in _kinds(report)
    assert "coverage_gap" in _kinds(report)
    store.close()


def test_write_todos_is_idempotent_and_produces_admissible_notes(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    store.save_note(Note(type="method", title="Floating claim", summary="Nothing backs this."))
    health = KnowledgeHealth(config, store)

    first = health.write_todos(health.check())
    assert first
    todos = store.list_notes(type="todo")
    assert len(todos) == len(set(first))
    assert all(note_from_markdown(p.read_text(encoding="utf-8")).type == "todo" for p in first)

    second = health.write_todos(health.check())
    assert set(second) == set(first)
    assert len(store.list_notes(type="todo")) == len(set(second))
    store.close()


def test_the_report_renders_as_markdown(tmp_path):
    config, store, _index = _healthy_kb(tmp_path)
    store.add_claim(Claim(statement="asserted by nobody"))
    report = KnowledgeHealth(config, store).check()
    markdown = report.to_markdown()
    assert markdown.startswith("# KB health")
    assert "## Errors" in markdown
    assert "uncited_claim" in markdown
    assert report.warnings == [i for i in report.issues if i.severity == "warning"]
    store.close()
