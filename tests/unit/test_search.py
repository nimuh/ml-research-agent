"""Retrieval: RRF fusion, strategies, filters, citations, reranking."""

from __future__ import annotations

import pytest
from kb_helpers import kb_config, synthetic_papers

from ml_research_agent.knowledge.chunk import chunk_sections
from ml_research_agent.knowledge.embed import get_embedder
from ml_research_agent.knowledge.index import HybridIndex
from ml_research_agent.knowledge.search import (
    Retriever,
    build_retriever,
    reciprocal_rank_fusion,
)
from ml_research_agent.knowledge.store import KnowledgeStore


@pytest.fixture
def retriever(tmp_path):
    config = kb_config(tmp_path)
    store = KnowledgeStore(config)
    index = HybridIndex(config.paths.kb_index, get_embedder(config), config.knowledge)
    for paper in synthetic_papers():
        doc_id = store.add_paper(paper)
        passages = chunk_sections(paper.sections, doc_id, config.knowledge)
        store.add_passages(passages)
        index.upsert(passages)
    index.save()
    yield Retriever(config, store, index)
    store.close()


def test_rrf_rewards_agreement_between_the_two_rankers():
    vector = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    keyword = [("c", 12.0), ("a", 3.0)]
    fused = reciprocal_rank_fusion(vector, keyword, rrf_k=10, alpha=0.5)
    assert [pid for pid, _ in fused] == ["a", "c", "b"]
    assert fused[0][1] > fused[1][1] > fused[2][1]


def test_rrf_alpha_shifts_the_weight_between_rankers():
    vector = [("v", 1.0)]
    keyword = [("k", 1.0)]
    assert reciprocal_rank_fusion(vector, keyword, alpha=1.0)[0][0] == "v"
    assert reciprocal_rank_fusion(vector, keyword, alpha=0.0)[0][0] == "k"


def test_rrf_promotes_what_neither_ranker_put_first():
    # The case fusion exists for: each ranker's favourite is invisible to the
    # other, and the passage both merely liked is the one worth returning.
    vector = [("vector_only", 0.99), ("both", 0.42)]
    keyword = [("keyword_only", 40.0), ("both", 1.1)]
    fused = reciprocal_rank_fusion(vector, keyword, rrf_k=1, alpha=0.5)
    assert [pid for pid, _ in fused] == ["both", "keyword_only", "vector_only"]
    assert fused[0][1] > fused[1][1]


def test_rrf_is_deterministic_on_ties():
    fused = reciprocal_rank_fusion([("b", 1.0), ("a", 1.0)], [("a", 1.0), ("b", 1.0)], rrf_k=60)
    assert [pid for pid, _ in fused] == ["a", "b"]


def test_search_returns_cited_hits(retriever):
    result = retriever.search("sparse attention long range arena", k=5)
    assert result.strategy == "hybrid"
    assert result.total == len(result.hits) > 0
    assert all(hit.citation for hit in result.hits)
    top = result.hits[0]
    assert "Sato" in top.citation
    assert top.passage.section in top.citation
    assert "sparse" in top.passage.text.lower()


def test_every_strategy_agrees_on_the_obvious_answer(retriever):
    for strategy in ("hybrid", "vector", "keyword"):
        result = retriever.search("quantization eight bits edge", k=3, strategy=strategy)
        assert result.strategy == strategy
        assert "quantiz" in result.hits[0].passage.text.lower()


def test_component_scores_are_reported(retriever):
    hits = retriever.search("curriculum learning difficulty", k=3).hits
    assert any(hit.vector_score is not None for hit in hits)
    assert any(hit.keyword_score is not None for hit in hits)


def test_filters_restrict_the_result_set(retriever):
    paper = synthetic_papers()[3]
    doc = retriever.store.document_by_key("paper", paper.key)
    result = retriever.search("retriever passages wikipedia", k=10, filters={"doc_id": doc.id})
    assert result.hits
    assert {hit.passage.doc_id for hit in result.hits} == {doc.id}

    by_section = retriever.search("model tokens", k=10, filters={"section": "Method"})
    assert by_section.hits
    assert {hit.passage.section for hit in by_section.hits} == {"Method"}


def test_k_bounds_the_result_set(retriever):
    assert len(retriever.search("model", k=2).hits) == 2
    assert len(retriever.search("model").hits) <= retriever.retrieval.top_k


def test_rerank_scores_the_head_of_the_list(retriever):
    result = retriever.search("mixture of experts routing", k=5, rerank=True)
    scored = [hit for hit in result.hits if hit.rerank_score is not None]
    assert scored
    assert all(0.0 <= hit.rerank_score <= 1.0 for hit in scored)
    assert scored == sorted(scored, key=lambda h: -(h.rerank_score or 0.0))


def test_a_query_matching_nothing_returns_no_hits(retriever):
    assert retriever.search("zzzz qqqq", k=5, strategy="keyword").hits == []


IRRELEVANT = [
    "zzzz qqqq wwww",
    "asdfgh lkjhgf poiuyt",
    "photosynthesis chlorophyll stomata",
    "medieval cathedral flying buttress",
]


@pytest.mark.parametrize("query", IRRELEVANT)
def test_an_irrelevant_query_returns_nothing_rather_than_the_least_bad_passage(retriever, query):
    # Nearest neighbours always exist; a hit nobody could honestly cite must not
    # reach the Writer.
    assert retriever.search(query, k=5).hits == []
    assert retriever.search(query, k=5, strategy="vector").hits == []


@pytest.mark.parametrize(
    "query",
    [
        "sparse attention long range arena",
        "curriculum difficulty sentence pairs",
        "eight bit quantization edge latency",
        "mixture of experts routing tokens",
    ],
)
def test_a_relevant_query_still_clears_the_floor(retriever, query):
    assert retriever.search(query, k=5).hits


def test_the_floor_is_what_removes_them_not_an_empty_index(retriever):
    # Same query, floor disabled: the passages were there all along, and RRF
    # ranked them. That is what makes this a policy and not a coincidence.
    assert retriever.search("zzzz qqqq wwww", k=5, min_score=0.0).hits


def test_lexical_evidence_survives_a_floor_no_cosine_could_clear(retriever):
    # A query term that literally occurs in a passage is citable evidence even
    # when the embedding says otherwise, so an impossible floor must not drop it.
    result = retriever.search("quantized", k=5, min_score=1.01)
    assert result.hits
    assert all(hit.keyword_score for hit in result.hits)
    assert "quantiz" in result.hits[0].passage.text.lower()


def test_a_raised_floor_narrows_the_result_set_monotonically(retriever):
    wide = {
        h.passage.id for h in retriever.search("attention mask tokens", k=20, min_score=0.0).hits
    }
    narrow = {h.passage.id for h in retriever.search("attention mask tokens", k=20).hits}
    strict = {
        h.passage.id
        for h in retriever.search(
            "attention mask tokens", k=20, min_score=1.01, strategy="vector"
        ).hits
    }
    assert strict <= narrow <= wide
    assert strict == set()


def test_the_keyword_strategy_is_already_floored_by_bm25(retriever):
    # BM25 only returns passages a term occurs in, so the cosine floor is a
    # no-op there; raising it must not silently empty a keyword search.
    assert retriever.search("curriculum", k=5, strategy="keyword", min_score=1.01).hits


def test_cite_reads_like_a_citation(retriever):
    hit = retriever.search("dense retriever wikipedia index generator", k=1).hits[0]
    citation = retriever.cite(hit)
    assert citation.startswith("Turing et al., 2020")
    assert "p." in citation


def test_get_passage_goes_through_the_store(retriever):
    hit = retriever.search("scaling laws experts", k=1).hits[0]
    assert retriever.get_passage(hit.passage.id) == hit.passage
    assert retriever.get_passage("psg_missing") is None


def test_build_retriever_opens_what_is_on_disk(tmp_path):
    config = kb_config(tmp_path)
    store = KnowledgeStore(config)
    index = HybridIndex(config.paths.kb_index, get_embedder(config), config.knowledge)
    paper = synthetic_papers()[0]
    doc_id = store.add_paper(paper)
    passages = chunk_sections(paper.sections, doc_id, config.knowledge)
    store.add_passages(passages)
    index.upsert(passages)
    index.save()
    store.close()

    reopened = build_retriever(config)
    assert reopened.index.size == len(passages)
    assert reopened.search("sparse attention", k=3).hits
    reopened.store.close()
