"""Notes: front-matter round trip, admissibility rules, supersede, wiki pages."""

from __future__ import annotations

from kb_helpers import synthetic_papers

from ml_research_agent.config import KnowledgeConfig
from ml_research_agent.knowledge.chunk import chunk_text
from ml_research_agent.knowledge.notes import (
    comparison_table,
    concept_page,
    merge_notes,
    note_stub_from_paper,
    validate_note,
)
from ml_research_agent.knowledge.schema import (
    NoteFrontMatter,
    note_from_markdown,
    note_to_markdown,
)
from ml_research_agent.types import Claim, MetricReport, Note, Provenance

CONFIG = KnowledgeConfig()


def _rich_note() -> Note:
    prov = Provenance(source="arxiv:2101.00001", locator="§Results p.5", quote="58.1 vs 58.4")
    return Note(
        type="paper",
        paper_key="arxiv:2101.00001",
        title="Sparse attention for long context transformers",
        summary="Sparse attention matches dense attention at 40% less memory.\n\nTwo paragraphs.",
        task="long context language modelling",
        method="sparse attention",
        datasets=["long range arena"],
        baselines=["dense attention"],
        metrics=[
            MetricReport(
                name="accuracy",
                value=58.1,
                unit="%",
                dataset="long range arena",
                split="test",
                method="sparse attention",
                provenance=prov,
            )
        ],
        compute="8 x A100 for 6 hours",
        limitations=["only tested up to 8k context"],
        relevance_to_brief="direct baseline",
        confidence=0.7,
        claims=["claim_abc123"],
        sources=["https://arxiv.org/abs/2101.00001"],
        provenance=[prov],
        tags=["attention", "efficiency"],
    )


def test_front_matter_round_trips_exactly():
    note = _rich_note()
    assert note_from_markdown(note_to_markdown(note)) == note


def test_round_trip_survives_a_summary_containing_headings():
    note = _rich_note().model_copy(
        update={"summary": "First line.\n\n## Not a real section\n\nStill the summary."}
    )
    assert note_from_markdown(note_to_markdown(note)) == note


def test_round_trip_of_a_minimal_note():
    note = Note(type="concept", title="Attention", summary="")
    assert note_from_markdown(note_to_markdown(note)) == note


def test_front_matter_contract_keys_are_present():
    markdown = note_to_markdown(_rich_note())
    for key in (
        "id:",
        "type:",
        "sources:",
        "claims:",
        "tags:",
        "confidence:",
        "added_at:",
        "supersedes:",
    ):
        assert key in markdown
    assert markdown.count("---") >= 2


def test_front_matter_model_round_trips_through_a_dict():
    note = _rich_note()
    front = NoteFrontMatter.from_note(note)
    assert NoteFrontMatter.from_dict(front.to_dict()) == front
    assert NoteFrontMatter.from_dict({**front.to_dict(), "unknown_key": 1}) == front


def test_a_note_without_provenance_is_rejected():
    note = Note(type="method", title="Sparse attention", summary="A method page.")
    violations = validate_note(note, CONFIG)
    assert any("provenance" in v for v in violations)


def test_provenance_without_a_locator_is_rejected_when_citations_are_required():
    note = Note(
        type="method",
        title="Sparse attention",
        summary="A method page.",
        provenance=[Provenance(source="arxiv:2101.00001")],
    )
    assert any("locator" in v for v in validate_note(note, CONFIG))
    assert validate_note(note, KnowledgeConfig(require_citations=False)) == []


def test_a_well_formed_note_is_admissible():
    assert validate_note(_rich_note(), CONFIG) == []


def test_metrics_need_a_passage_locator():
    note = _rich_note()
    loose = note.metrics[0].model_copy(update={"provenance": Provenance(source="arxiv:2101.00001")})
    note = note.model_copy(update={"metrics": [loose]})
    assert any("passage locator" in v for v in validate_note(note, CONFIG))


def test_todo_notes_are_exempt_from_the_provenance_rule():
    todo = Note(type="todo", title="TODO: ingest the missing paper", summary="", confidence=0.0)
    assert validate_note(todo, CONFIG) == []


def test_a_stub_from_a_paper_is_traceable_from_the_start():
    paper = synthetic_papers()[0]
    passages = chunk_text(paper.sections[0].text, "doc_1", CONFIG, section="Method", page=1)
    stub = note_stub_from_paper(paper, passages=passages)
    assert stub.paper_key == paper.key
    assert stub.provenance and all(p.locator for p in stub.provenance)
    assert validate_note(stub, CONFIG) == []
    assert "stub" in stub.tags


def test_a_stub_without_passages_still_cites_the_abstract():
    stub = note_stub_from_paper(synthetic_papers()[1])
    assert stub.provenance[0].locator == "§Abstract"
    assert validate_note(stub, CONFIG) == []


def test_merge_keeps_history_and_unions_the_lists():
    old = _rich_note()
    new = old.model_copy(
        update={
            "id": "note_newer",
            "summary": "Revised reading.",
            "datasets": ["wikitext"],
            "limitations": ["single seed"],
            "provenance": [Provenance(source="arxiv:2101.00001", locator="§Appendix p.11")],
            "task": None,
        }
    )
    merged = merge_notes(old, new)
    assert merged.supersedes == old.id
    assert merged.summary == "Revised reading."
    assert merged.datasets == ["long range arena", "wikitext"]
    assert merged.limitations == ["only tested up to 8k context", "single seed"]
    assert len(merged.provenance) == 2
    assert merged.task == old.task  # not restated is not the same as retracted


def test_concept_page_gathers_notes_and_claims():
    notes = [_rich_note()]
    claim = Claim(statement="sparse attention matches dense attention", contradicts=["claim_x"])
    page = concept_page("Sparse attention", notes, claims=[claim])
    assert page.type == "concept"
    assert page.claims == [claim.id]
    assert notes[0].id in page.sources
    assert "long range arena" in page.summary
    assert "[contested]" in page.summary
    assert validate_note(page, CONFIG) == []


def test_comparison_table_is_valid_markdown():
    note = _rich_note()
    other = note.model_copy(update={"id": "note_2", "title": "Dense baseline", "metrics": []})
    table = comparison_table([note, other], metric="accuracy")
    lines = table.splitlines()
    assert lines[0].startswith("| Paper | Method | Datasets | accuracy |")
    assert len(lines) == 4
    assert "58.1" in lines[2]
    assert lines[3].endswith("|")
