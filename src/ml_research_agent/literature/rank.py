"""Candidate ranking: hybrid of semantic similarity to the brief, citation
velocity, recency, venue, and code availability. Cheap filter before LLM screening."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

from ..config import LiteratureConfig
from ..types import Paper, RankedPaper, ResearchBrief
from ._text import cosine, term_frequencies, tokenize

# Weights sum to 1.0 so a score is directly comparable to snowball_min_score.
# Similarity dominates deliberately: the other four are priors about a paper,
# not about whether it has anything to do with this brief.
W_SIMILARITY = 0.50
W_CITATION_VELOCITY = 0.20
W_RECENCY = 0.15
W_VENUE = 0.10
W_CODE = 0.05

WEIGHTS: dict[str, float] = {
    "similarity": W_SIMILARITY,
    "citation_velocity": W_CITATION_VELOCITY,
    "recency": W_RECENCY,
    "venue": W_VENUE,
    "code_availability": W_CODE,
}

# 100 citations per year saturates the velocity component.
CITATION_VELOCITY_CAP = 100.0
# The title states what a paper is about; the abstract hedges. Weight accordingly.
TITLE_REPEAT = 3
UNKNOWN_YEAR_AGE = 3
UNKNOWN_RECENCY = 0.5

_TOP_VENUES = frozenset(
    {"neurips", "nips", "icml", "iclr", "jmlr", "cvpr", "iccv", "eccv", "acl", "emnlp", "naacl"}
)
_STRONG_VENUES = frozenset(
    {
        "aaai",
        "ijcai",
        "kdd",
        "sigir",
        "www",
        "colt",
        "aistats",
        "uai",
        "tmlr",
        "mlsys",
        "colm",
        "coling",
        "siggraph",
        "interspeech",
        "icassp",
        "wacv",
        "bmvc",
        "conll",
        "eacl",
    }
)
_VENUE_SCORES = {
    "conference": 0.7,
    "journal": 0.7,
    "workshop": 0.45,
    "preprint": 0.35,
    "unknown": 0.3,
}


def score_components(
    paper: Paper,
    brief: ResearchBrief,
    config: LiteratureConfig,
    *,
    now_year: int | None = None,
) -> dict[str, float]:
    """The five ranking signals, each normalized to 0..1.

    Every one of them is lexical or metadata-derived: ranking here is a *filter*
    before LLM screening, so it must stay cheap enough to run on 300 candidates
    and dumb enough that nobody mistakes it for a relevance judgment.
    """
    year = now_year if now_year is not None else datetime.now(UTC).year
    return {
        "similarity": _similarity(paper, brief),
        "citation_velocity": _citation_velocity(paper, now_year=year),
        "recency": _recency(paper, config, now_year=year),
        "venue": _venue_score(paper),
        "code_availability": 1.0 if paper.code_urls else 0.0,
    }


def rank_candidates(
    papers: Sequence[Paper],
    brief: ResearchBrief,
    config: LiteratureConfig,
    *,
    now_year: int | None = None,
) -> list[RankedPaper]:
    """Score and order candidates. Stable: ties break on ``paper.key``."""
    ranked: list[RankedPaper] = []
    for paper in papers:
        components = score_components(paper, brief, config, now_year=now_year)
        ranked.append(RankedPaper(paper=paper, score=_combine(components), components=components))
    ranked.sort(key=lambda item: (-item.score, item.paper.key))
    for position, item in enumerate(ranked, start=1):
        item.rank = position
    return ranked


def _combine(components: dict[str, float]) -> float:
    return round(sum(WEIGHTS[name] * value for name, value in components.items()), 6)


def _similarity(paper: Paper, brief: ResearchBrief) -> float:
    query = term_frequencies(tokenize(brief.query_text))
    document_text = " ".join(
        [
            *([paper.title] * TITLE_REPEAT),
            paper.abstract,
            paper.tldr or "",
            " ".join(paper.tags),
        ]
    )
    return cosine(query, term_frequencies(tokenize(document_text)))


def _citation_velocity(paper: Paper, *, now_year: int) -> float:
    if paper.citation_count <= 0:
        return 0.0
    age = UNKNOWN_YEAR_AGE if paper.year is None else max(1, now_year - paper.year + 1)
    velocity = paper.citation_count / age
    return min(1.0, math.log1p(velocity) / math.log1p(CITATION_VELOCITY_CAP))


def _recency(paper: Paper, config: LiteratureConfig, *, now_year: int) -> float:
    """Linear decay across the date window, with a seminal-work exemption.

    The exemption is the whole point of ``seminal_citation_floor``: a survey
    that ranks the field's foundational paper below last month's preprint has
    already failed its exit criterion.
    """
    if config.seminal_citation_floor > 0 and paper.citation_count >= config.seminal_citation_floor:
        return 1.0
    if paper.year is None:
        return UNKNOWN_RECENCY
    age = now_year - paper.year
    if age <= 0:
        return 1.0
    return max(0.0, 1.0 - age / config.date_window_years)


def _venue_score(paper: Paper) -> float:
    if paper.venue is None:
        return _VENUE_SCORES["unknown"]
    tokens = set(tokenize(paper.venue.name, drop_stopwords=False))
    if tokens & _TOP_VENUES:
        return 1.0
    if tokens & _STRONG_VENUES:
        return 0.85
    return _VENUE_SCORES.get(paper.venue.kind, _VENUE_SCORES["unknown"])


__all__ = [
    "CITATION_VELOCITY_CAP",
    "WEIGHTS",
    "W_CITATION_VELOCITY",
    "W_CODE",
    "W_RECENCY",
    "W_SIMILARITY",
    "W_VENUE",
    "rank_candidates",
    "score_components",
]
