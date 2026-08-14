"""Retrieval API: hybrid query, reciprocal-rank fusion, optional reranking,
returns passages with citations. The single retrieval entrypoint for agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..config import Config
from ..types import Passage, SearchHit
from .embed import get_embedder
from .index import HybridIndex, tokenize
from .store import KnowledgeStore

Strategy = Literal["hybrid", "vector", "keyword"]

# Cosine floor below which a passage is treated as unrelated to the query.
#
# Dense retrieval has no notion of "no match": nearest neighbours always exist,
# so an off-topic query still returns the least-unrelated passage. Left alone,
# the Writer can cite a passage that scored 0.008 as if it were evidence, which
# is precisely the fabricated-citation failure the traceability rule exists to
# stop. A floor makes "nothing in the KB answers this" expressible.
#
# 0.10 is measured, not guessed: over the test corpus, off-topic queries top out
# at 0.097 (hash collisions in HashingEmbedder) while on-topic queries score
# 0.113 upward. That margin is thin, which is why the floor is only half the
# rule -- see _admissible: a passage that literally contains a query term is
# admitted on that lexical evidence regardless of its cosine. Swapping in an
# embedding backend that carries real paraphrase signal should re-tune this.
DEFAULT_MIN_SCORE = 0.10


class SearchResult(BaseModel):
    """One retrieval call's worth of hits, plus what produced them."""

    hits: list[SearchHit] = Field(default_factory=list)
    query: str = ""
    total: int = 0
    strategy: str = "hybrid"

    def __len__(self) -> int:
        return len(self.hits)

    @property
    def passages(self) -> list[Passage]:
        return [hit.passage for hit in self.hits]

    @property
    def citations(self) -> list[str]:
        return [hit.citation for hit in self.hits]


class Retriever:
    """The single retrieval entrypoint.

    Every agent read of the KB goes through here, which is what makes it
    possible to instrument, cache and evaluate retrieval in one place instead of
    guessing which of eleven agents phrased a query badly.
    """

    def __init__(self, config: Config, store: KnowledgeStore, index: HybridIndex) -> None:
        self.config = config
        self.store = store
        self.index = index
        self.retrieval = config.knowledge.retrieval
        self._citation_cache: dict[str, str] = {}

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        strategy: Strategy = "hybrid",
        rerank: bool = False,
        min_score: float | None = None,
    ) -> SearchResult:
        """Retrieve passages, dropping everything with no evidence behind it.

        ``min_score`` is a floor on the *cosine* score, not on the fused score:
        RRF scores are rank-derived and incomparable between queries, so a
        threshold on them would mean nothing. A passage is returned when it
        clears that floor **or** when a query term literally occurs in it (BM25
        found it). Pass ``min_score=0.0`` to disable the floor and get raw
        nearest neighbours.
        """
        top_k = k or self.retrieval.top_k
        floor = DEFAULT_MIN_SCORE if min_score is None else min_score
        # Fuse from a deeper pool than we return: a passage ranked 30th by BM25
        # and 5th by vector should still be able to win.
        pool = max(top_k * 3, self.retrieval.top_k)
        vector = (
            self.index.vector_search(query, pool, filters=filters) if strategy != "keyword" else []
        )
        keyword = (
            self.index.keyword_search(query, pool, filters=filters) if strategy != "vector" else []
        )

        if strategy == "vector":
            scored = [(pid, score) for pid, score in vector]
        elif strategy == "keyword":
            scored = [(pid, score) for pid, score in keyword]
        else:
            scored = reciprocal_rank_fusion(
                vector,
                keyword,
                rrf_k=self.retrieval.rrf_k,
                alpha=self.retrieval.hybrid_alpha,
            )

        vector_scores = dict(vector)
        keyword_scores = dict(keyword)
        admissible = _admissible(vector, keyword, floor=floor, strategy=strategy)
        # Filtered after ranking, not before: dropping a candidate must not
        # promote the ones below it into ranks they did not earn.
        scored = [(pid, score) for pid, score in scored if pid in admissible]

        hits: list[SearchHit] = []
        for passage_id, score in scored[:top_k]:
            passage = self.get_passage(passage_id)
            if passage is None:
                # Indexed but absent from the store: a drift the health check
                # reports. Skipping is right -- a hit we cannot cite is useless.
                continue
            hit = SearchHit(
                passage=passage,
                score=score,
                vector_score=vector_scores.get(passage_id),
                keyword_score=keyword_scores.get(passage_id),
            )
            hit.citation = self.cite(hit)
            hits.append(hit)

        if rerank:
            hits = self._rerank(query, hits)
        return SearchResult(hits=hits, query=query, total=len(hits), strategy=strategy)

    def get_passage(self, passage_id: str) -> Passage | None:
        return self.store.get_passage(passage_id)

    def cite(self, hit: SearchHit) -> str:
        """Human-readable citation, e.g. ``Vaswani et al., 2017 §3.1 p.4``."""
        label = self._doc_label(hit.passage.doc_id)
        return f"{label} {hit.passage.locator}".strip()

    def _doc_label(self, doc_id: str) -> str:
        cached = self._citation_cache.get(doc_id)
        if cached is not None:
            return cached
        record = self.store.get_document(doc_id)
        label = doc_id
        if record is not None:
            paper = self.store.paper_for_doc(doc_id)
            label = paper.citation_label if paper else (record.title or record.key)
        self._citation_cache[doc_id] = label
        return label

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """Cheap lexical rerank of the head of the list.

        Deliberately not an LLM call: reranking runs on every retrieval, and a
        cross-encoder belongs behind the router where its cost is visible.
        """
        top_n = self.retrieval.rerank_top_n
        terms = set(tokenize(query))
        if not terms:
            return hits
        head, tail = hits[:top_n], hits[top_n:]
        for hit in head:
            tokens = tokenize(hit.passage.text)
            unique = set(tokens)
            coverage = len(terms & unique) / len(terms)
            density = sum(1 for t in tokens if t in terms) / max(len(tokens), 1)
            hit.rerank_score = round(0.8 * coverage + 0.2 * min(density * 10, 1.0), 6)
        head.sort(key=lambda h: (-(h.rerank_score or 0.0), -h.score, h.passage.id))
        return head + tail


def _admissible(
    vector: list[tuple[str, float]],
    keyword: list[tuple[str, float]],
    *,
    floor: float,
    strategy: Strategy,
) -> set[str]:
    """Passage ids with enough evidence behind them to be worth returning.

    Two kinds of evidence count. Semantic: the cosine score clears the floor.
    Lexical: BM25 matched, meaning a query term actually occurs in the passage
    (``keyword_search`` already drops zero scores). Either one justifies a
    citation; neither one means the KB has nothing to say.
    """
    semantic = {pid for pid, score in vector if score >= floor}
    lexical = {pid for pid, _ in keyword}
    if strategy == "vector":
        return semantic
    if strategy == "keyword":
        return lexical
    return semantic | lexical


def reciprocal_rank_fusion(
    vector: list[tuple[str, float]],
    keyword: list[tuple[str, float]],
    *,
    rrf_k: int = 60,
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """Weighted RRF over two ranked lists.

    Rank-based rather than score-based on purpose: cosine similarity and BM25
    are not on comparable scales, and normalizing them invents a calibration
    neither one has.
    """
    fused: dict[str, float] = {}
    for weight, ranking in ((alpha, vector), (1.0 - alpha, keyword)):
        for rank, (passage_id, _score) in enumerate(ranking, start=1):
            fused[passage_id] = fused.get(passage_id, 0.0) + weight / (rrf_k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def build_retriever(config: Config) -> Retriever:
    """Open the store and the index and hand back the entrypoint agents use."""
    store = KnowledgeStore(config)
    index = HybridIndex(config.paths.kb_index, get_embedder(config), config.knowledge)
    index.load()
    return Retriever(config, store, index)


__all__ = [
    "DEFAULT_MIN_SCORE",
    "Retriever",
    "SearchResult",
    "build_retriever",
    "reciprocal_rank_fusion",
]
