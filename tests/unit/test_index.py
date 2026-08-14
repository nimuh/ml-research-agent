"""Hybrid index: upsert idempotency, recall, filters, rebuild from disk."""

from __future__ import annotations

import numpy as np
from kb_helpers import kb_config, synthetic_papers

from ml_research_agent.knowledge.chunk import chunk_sections
from ml_research_agent.knowledge.embed import HashingEmbedder
from ml_research_agent.knowledge.index import HybridIndex, load_index_ids
from ml_research_agent.knowledge.store import KnowledgeStore
from ml_research_agent.types import Passage


def _passages() -> list[Passage]:
    return [
        Passage(
            id="psg_1",
            doc_id="doc_a",
            text="sparse attention reduces the quadratic cost of long context transformers",
            section="Method",
            page=2,
            order=0,
        ),
        Passage(
            id="psg_2",
            doc_id="doc_a",
            text="the sparse model matches dense attention on the long range arena benchmark",
            section="Results",
            page=5,
            order=1,
        ),
        Passage(
            id="psg_3",
            doc_id="doc_b",
            text="curriculum learning orders sentence pairs by difficulty for machine translation",
            section="Method",
            page=2,
            order=0,
        ),
    ]


def _index(tmp_path, dim: int = 256) -> HybridIndex:
    config = kb_config(tmp_path, embedding_dim=dim)
    return HybridIndex(config.paths.kb_index, HashingEmbedder(dim=dim), config.knowledge)


def test_upsert_is_idempotent_by_passage_id(tmp_path):
    index = _index(tmp_path)
    passages = _passages()
    index.upsert(passages)
    assert index.size == 3
    index.upsert(passages)
    assert index.size == 3

    edited = passages[0].model_copy(update={"text": "completely different wording here"})
    index.upsert([edited])
    assert index.size == 3
    record = index.get_record("psg_1")
    assert record is not None and record["text"] == "completely different wording here"


def _corpus_passages(config) -> list[Passage]:
    passages: list[Passage] = []
    for i, paper in enumerate(synthetic_papers()):
        passages += chunk_sections(paper.sections, f"doc_{i}", config.knowledge)
    return passages


def test_replacing_a_passage_refreshes_both_rankers(tmp_path):
    # Upserting over an existing id must leave the index in the state a
    # from-scratch build would produce: a stale vector or a stale BM25 corpus
    # keeps serving text the KB no longer holds.
    config = kb_config(tmp_path, embedding_dim=256)
    passages = _corpus_passages(config)
    target = next(p for p in passages if "curriculum" in p.text)
    edited = target.model_copy(update={"text": "omicron pi rho sigma tau upsilon"})

    incremental = HybridIndex(config.paths.kb_index, HashingEmbedder(dim=256), config.knowledge)
    incremental.upsert(passages)
    assert incremental.keyword_search("curriculum", 5)[0][0] == target.id  # warms the BM25 cache
    incremental.upsert([edited])

    fresh = HybridIndex(config.paths.kb_index, HashingEmbedder(dim=256), config.knowledge)
    fresh.upsert([edited if p.id == target.id else p for p in passages])

    assert incremental.vector_search("omicron pi rho sigma", 5) == fresh.vector_search(
        "omicron pi rho sigma", 5
    )
    assert incremental.keyword_search("omicron", 5) == fresh.keyword_search("omicron", 5) != []
    gone = incremental.keyword_search("curriculum", 5)
    assert gone == fresh.keyword_search("curriculum", 5) == []


def test_vector_search_finds_the_right_passage(tmp_path):
    index = _index(tmp_path)
    index.upsert(_passages())
    hits = index.vector_search("sparse attention quadratic cost", 3)
    assert hits[0][0] == "psg_1"
    assert hits[0][1] > hits[-1][1]


def test_keyword_search_finds_exact_terms(tmp_path):
    index = _index(tmp_path)
    index.upsert(_passages())
    hits = index.keyword_search("curriculum difficulty", 3)
    assert hits[0][0] == "psg_3"
    assert index.keyword_search("nonexistentterm", 3) == []


def test_filters_are_applied_before_scoring(tmp_path):
    index = _index(tmp_path)
    index.upsert(_passages())
    assert [pid for pid, _ in index.vector_search("attention", 5, filters={"doc_id": "doc_b"})] == [
        "psg_3"
    ]
    assert [
        pid for pid, _ in index.keyword_search("attention", 5, filters={"doc_id": "doc_b"})
    ] == []
    both = index.vector_search("attention", 5, filters={"doc_id": ["doc_a", "doc_b"]})
    assert len(both) == 3
    assert [pid for pid, _ in index.vector_search("method", 5, filters={"section": "Results"})] == [
        "psg_2"
    ]


def test_remove_drops_rows_and_keeps_the_rest_searchable(tmp_path):
    index = _index(tmp_path)
    index.upsert(_passages())
    assert index.remove(["psg_1", "missing"]) == 1
    assert index.size == 2
    assert all(pid != "psg_1" for pid, _ in index.vector_search("sparse attention", 5))
    assert index.get_record("psg_1") is None


def test_save_and_load_round_trip(tmp_path):
    index = _index(tmp_path)
    index.upsert(_passages())
    before = index.vector_search("sparse attention", 3)
    index.save()

    reopened = _index(tmp_path)
    reopened.load()
    assert reopened.size == 3
    assert reopened.vector_search("sparse attention", 3) == before
    assert reopened.keyword_search("curriculum", 3)[0][0] == "psg_3"
    assert load_index_ids(reopened.root) == {"psg_1", "psg_2", "psg_3"}


def test_a_dimension_mismatch_drops_the_index_instead_of_serving_garbage(tmp_path):
    index = _index(tmp_path, dim=256)
    index.upsert(_passages())
    index.save()

    other = _index(tmp_path, dim=128)
    other.load()
    assert other.size == 0


def test_rebuild_from_the_store_reproduces_the_index_exactly(tmp_path):
    config = kb_config(tmp_path, embedding_dim=256)
    store = KnowledgeStore(config)
    for paper in synthetic_papers():
        doc_id = store.add_paper(paper)
        store.add_passages(chunk_sections(paper.sections, doc_id, config.knowledge))

    index = HybridIndex(config.paths.kb_index, HashingEmbedder(dim=256), config.knowledge)
    index.upsert(store.list_passages())
    index.save()
    ids_before = list(index._ids)
    vectors_before = np.array(index._vectors, copy=True)
    hits_before = index.vector_search("quantization eight bits edge inference", 5)

    fresh = HybridIndex(config.paths.kb_index, HashingEmbedder(dim=256), config.knowledge)
    count = fresh.rebuild(store)

    assert count == len(ids_before)
    assert list(fresh._ids) == ids_before
    assert np.array_equal(fresh._vectors, vectors_before)
    assert fresh.vector_search("quantization eight bits edge inference", 5) == hits_before
    store.close()


def test_rebuild_drops_state_the_store_never_had(tmp_path):
    # Files are truth: a rebuild must reconstruct the index from the store alone,
    # so anything only the in-memory index knew about has to disappear.
    config = kb_config(tmp_path, embedding_dim=256)
    store = KnowledgeStore(config)
    for paper in synthetic_papers():
        doc_id = store.add_paper(paper)
        store.add_passages(chunk_sections(paper.sections, doc_id, config.knowledge))

    index = HybridIndex(config.paths.kb_index, HashingEmbedder(dim=256), config.knowledge)
    index.upsert(store.list_passages())
    expected = index.vector_search("the curriculum anneals from easy to hard", 5)

    ghost = Passage(
        id="psg_ghost",
        doc_id="doc_ghost",
        text="the curriculum anneals from easy to hard",
        order=0,
    )
    index.upsert([ghost])
    assert index.vector_search("the curriculum anneals from easy to hard", 5)[0][0] == "psg_ghost"

    assert index.rebuild(store) == len(store.list_passages())
    assert index.size == len(store.list_passages())
    assert index.get_record("psg_ghost") is None
    assert "psg_ghost" not in load_index_ids(index.root)
    assert index.vector_search("the curriculum anneals from easy to hard", 5) == expected
    assert all(pid != "psg_ghost" for pid, _ in index.keyword_search("curriculum anneals", 5))
    store.close()


def test_an_empty_index_answers_without_blowing_up(tmp_path):
    index = _index(tmp_path)
    index.load()
    assert index.size == 0
    assert index.vector_search("anything", 5) == []
    assert index.keyword_search("anything", 5) == []
