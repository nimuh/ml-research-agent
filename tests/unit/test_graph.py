"""Knowledge graph: contradictions, coverage gaps, lineage, serialization."""

from __future__ import annotations

import pytest
from kb_helpers import kb_config

from ml_research_agent.knowledge.graph import KnowledgeGraph, node_id
from ml_research_agent.knowledge.store import KnowledgeStore
from ml_research_agent.types import (
    Claim,
    ClaimRelation,
    Evidence,
    EvidenceKind,
    GraphEdge,
    MetricReport,
    Note,
    Provenance,
)

PROV = Provenance(source="arxiv:2101.00001", locator="§Results p.5")


def _note(title: str, method: str, datasets: list[str], **overrides) -> Note:
    fields = {
        "type": "paper",
        "title": title,
        "summary": f"{method} on {', '.join(datasets)}",
        "paper_key": f"arxiv:{title.lower().replace(' ', '-')}",
        "method": method,
        "datasets": datasets,
        "metrics": [MetricReport(name="accuracy", value=1.0, provenance=PROV)],
        "provenance": [PROV],
        "confidence": 0.6,
    }
    fields.update(overrides)
    return Note(**fields)


@pytest.fixture
def graph(tmp_path):
    config = kb_config(tmp_path)
    store = KnowledgeStore(config)
    store.save_note(_note("Sparse attention", "sparse attention", ["long range arena"]))
    store.save_note(_note("Dense baseline", "dense attention", ["long range arena", "wikitext"]))
    store.save_note(_note("Curriculum MT", "curriculum learning", ["wmt14"]))
    yield KnowledgeGraph(store), store
    store.close()


def test_notes_link_to_the_entities_they_mention(graph):
    kg, _store = graph
    built = kg.build()
    assert node_id("method", "sparse attention") in built
    assert node_id("dataset", "long range arena") in built
    assert node_id("metric", "accuracy") in built

    dataset_neighbors = kg.neighbors(node_id("dataset", "long range arena"))
    assert len(dataset_neighbors) == 2
    assert all(n.startswith("note:") for n in dataset_neighbors)


def test_contradicting_claims_surface_as_candidate_replications(graph):
    kg, store = graph
    a = Claim(
        statement="sparse attention matches dense attention",
        evidence=[Evidence(kind=EvidenceKind.PASSAGE, provenance=PROV)],
    )
    b = Claim(statement="sparse attention loses 2 points", contradicts=[a.id])
    store.add_claim(a)
    store.add_claim(b)
    kg.build()

    pairs = kg.contradictions()
    assert pairs == [tuple(sorted((node_id("claim", a.id), node_id("claim", b.id))))]
    assert kg.neighbors(node_id("claim", a.id), relation=ClaimRelation.CONTRADICTS) == [
        node_id("claim", b.id)
    ]
    assert kg.neighbors(node_id("claim", a.id), relation=ClaimRelation.REPRODUCES) == []


def test_stored_edges_are_part_of_the_graph(graph):
    kg, _store = graph
    kg.add_edge(
        GraphEdge(
            source=node_id("paper", "arxiv:a"),
            target=node_id("paper", "arxiv:b"),
            relation=ClaimRelation.REPRODUCES,
        )
    )
    assert node_id("paper", "arxiv:b") in kg.neighbors(node_id("paper", "arxiv:a"))
    rebuilt = KnowledgeGraph(kg.store)
    rebuilt.build()
    assert node_id("paper", "arxiv:b") in rebuilt.neighbors(node_id("paper", "arxiv:a"))


def test_re_adding_the_same_typed_edge_does_not_duplicate_it(graph):
    # Ingest is idempotent by content hash, which is worth nothing if replaying
    # the same relation piles up parallel edges the gap finder then miscounts.
    kg, _store = graph
    edge = GraphEdge(
        source=node_id("paper", "arxiv:a"),
        target=node_id("paper", "arxiv:b"),
        relation=ClaimRelation.REPRODUCES,
    )
    before = kg.build().number_of_edges()
    kg.add_edge(edge)
    once = kg.graph.number_of_edges()
    assert once == before + 1

    kg.add_edge(edge)
    assert kg.graph.number_of_edges() == once
    assert len(kg.to_dict()["edges"]) == once
    assert len(KnowledgeGraph(kg.store).to_dict()["edges"]) == once


def test_coverage_matrix_counts_method_by_dataset(graph):
    kg, _store = graph
    matrix = kg.coverage_matrix()
    assert matrix[("sparse attention", "long range arena")] == 1
    assert matrix[("dense attention", "wikitext")] == 1
    assert ("sparse attention", "wikitext") not in matrix


def test_gaps_are_the_holes_in_that_matrix(graph):
    kg, _store = graph
    gaps = kg.gaps()
    cells = {(g["row"], g["col"]) for g in gaps}
    assert ("sparse attention", "wikitext") in cells
    assert ("curriculum learning", "long range arena") in cells
    assert ("sparse attention", "long range arena") not in cells
    assert all(g["support"] == 0 for g in gaps)
    assert gaps[0]["rows"] == "method" and gaps[0]["cols"] == "dataset"


def test_a_higher_support_bar_finds_more_gaps(graph):
    kg, _store = graph
    assert len(kg.gaps(min_support=2)) > len(kg.gaps(min_support=1))


def test_other_axes_are_available(graph):
    kg, _store = graph
    matrix = kg.coverage_matrix(rows="dataset", cols="metric")
    assert matrix[("wmt14", "accuracy")] == 1
    with pytest.raises(ValueError, match="unknown coverage axis"):
        kg.coverage_matrix(rows="nonsense")


def test_lineage_follows_supersession(graph):
    kg, store = graph
    old = _note("Sparse attention", "sparse attention", ["long range arena"])
    store.save_note(old)
    newer = store.supersede_note(
        old.id, _note("Sparse attention, revised", "sparse attention", ["long range arena"])
    )
    kg.build()

    lineage = kg.lineage(node_id("note", newer.id))
    assert lineage[0] == node_id("note", newer.id)
    assert node_id("note", old.id) in lineage
    assert kg.lineage("note:does_not_exist") == []


def test_to_dict_is_serializable_and_counts_itself(graph):
    kg, _store = graph
    payload = kg.to_dict()
    assert payload["stats"]["nodes"] == len(payload["nodes"])
    assert payload["stats"]["edges"] == len(payload["edges"])
    assert all(isinstance(edge["relation"], str) for edge in payload["edges"])
    assert {n["kind"] for n in payload["nodes"]} >= {"note", "method", "dataset"}
