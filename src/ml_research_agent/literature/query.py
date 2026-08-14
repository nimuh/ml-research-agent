"""Query planning: brief -> a set of source-specific queries (keyword, semantic,
author, venue, date-window), plus follow-up queries seeded by early results."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field

from ..config import LiteratureConfig
from ..types import MRAModel, Paper, ResearchBrief
from ._text import collapse_whitespace, key_phrase, tokenize

# Caps on the plan itself. Fan-out is bounded before a single request is made:
# a plan of 40 queries against 6 sources is 240 round trips nobody asked for.
MAX_QUERIES = 12
MAX_FOLLOWUP_QUERIES = 6
MAX_SEMANTIC_CHARS = 400

# Recognised venues, used only to *detect* a venue the brief already names --
# this layer never decides that a venue makes a paper relevant.
KNOWN_VENUES: tuple[str, ...] = (
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "IJCAI",
    "ACL",
    "EMNLP",
    "NAACL",
    "COLING",
    "CVPR",
    "ICCV",
    "ECCV",
    "SIGGRAPH",
    "KDD",
    "WWW",
    "SIGIR",
    "COLT",
    "AISTATS",
    "UAI",
    "TMLR",
    "JMLR",
    "MLSys",
    "COLM",
)

_AUTHOR_PREFIX = "author:"
_VENUE_PREFIX = "venue:"


class SourceQuery(MRAModel):
    """One query as issued to one source. Frozen: a plan is a record, not a scratchpad."""

    model_config = ConfigDict(frozen=True)

    text: str
    keywords: list[str] = Field(default_factory=list)
    authors: list[str] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    limit: int = 50
    kind: Literal["keyword", "semantic", "author", "venue"] = "keyword"

    @property
    def cache_key(self) -> str:
        return "|".join(
            [
                self.kind,
                self.text.lower(),
                ",".join(self.keywords),
                ",".join(self.authors),
                ",".join(self.venues),
                str(self.year_from),
                str(self.year_to),
                str(self.limit),
            ]
        )


class QueryPlan(MRAModel):
    """The set of queries a survey will issue, plus why it looks like this."""

    queries: list[SourceQuery] = Field(default_factory=list)
    rationale: str = ""

    def __len__(self) -> int:
        return len(self.queries)


def plan_queries(
    brief: ResearchBrief, config: LiteratureConfig, *, now_year: int | None = None
) -> QueryPlan:
    """Derive a bounded, deterministic query set from the brief.

    No LLM: the brief has already been through the Framer, so query planning is
    a mechanical expansion of what it states. The date window is *soft* -- one
    unwindowed semantic sweep is always issued so seminal older work can still
    surface (the SURVEY exit criterion expects it to).
    """
    year = now_year if now_year is not None else datetime.now(UTC).year
    year_from = year - config.date_window_years
    limit = config.per_source_limit

    builder = _PlanBuilder(limit=limit, year_from=year_from, year_to=year)

    # The unwindowed sweep. Everything else is windowed.
    builder.add(
        collapse_whitespace(brief.query_text)[:MAX_SEMANTIC_CHARS],
        kind="semantic",
        windowed=False,
        keywords=list(brief.search_terms[:8]),
    )
    builder.add(collapse_whitespace(brief.title), kind="keyword")

    plain_terms, authors, venues = _partition_terms([*brief.search_terms, *brief.tags])
    for term in plain_terms[:6]:
        builder.add(term, kind="keyword")
    for author in authors[:3]:
        builder.add(author, kind="author", authors=[author])
    for venue in venues[:3]:
        builder.add(venue, kind="venue", venues=[venue])

    for claim in brief.claims[:3]:
        builder.add(key_phrase(claim.statement), kind="keyword")

    anchor = key_phrase(brief.title, max_terms=3)
    for area in brief.related_areas[:3]:
        builder.add(collapse_whitespace(f"{area} {anchor}"), kind="keyword")

    haystack = " ".join(
        [brief.title, brief.problem_statement, *brief.related_areas, *brief.search_terms]
    )
    for venue in _detect_venues(haystack):
        builder.add(venue, kind="venue", venues=[venue])

    queries, dropped = builder.finish(MAX_QUERIES)
    rationale = (
        f"{len(queries)} queries from brief {brief.id}: one unwindowed semantic sweep "
        f"(seminal work is exempt from the {config.date_window_years}-year window), "
        f"the rest windowed to {year_from}-{year}."
    )
    if dropped:
        rationale += f" {dropped} lower-priority queries dropped at the cap of {MAX_QUERIES}."
    return QueryPlan(queries=queries, rationale=rationale)


def followup_queries(
    plan: QueryPlan, seen: list[Paper], config: LiteratureConfig
) -> list[SourceQuery]:
    """Mine early results for terms, authors and venues the plan did not anticipate.

    This is the cheap half of recall: the vocabulary an area actually uses is
    in its titles, not in the brief.
    """
    if not seen:
        return []

    covered = {token for query in plan.queries for token in tokenize(query.text)}
    covered.update(
        tokenize(" ".join(a for q in plan.queries for a in q.authors), drop_stopwords=False)
    )
    year_from = max((q.year_from for q in plan.queries if q.year_from is not None), default=None)
    year_to = max((q.year_to for q in plan.queries if q.year_to is not None), default=None)
    anchor = _anchor_term(plan)

    builder = _PlanBuilder(limit=config.per_source_limit, year_from=year_from, year_to=year_to)
    for query in plan.queries:  # follow-ups must not repeat what was already issued
        builder.seen.add(_dedup_key(query.kind, query.text))

    title_terms = Counter(
        token for paper in seen for token in set(tokenize(paper.title)) if token not in covered
    )
    for term, count in title_terms.most_common(8):
        if count < 2:
            break
        builder.add(collapse_whitespace(f"{anchor} {term}" if anchor else term), kind="keyword")

    author_counts = Counter(
        author.name for paper in seen for author in paper.authors[:5] if author.name.strip()
    )
    for name, count in author_counts.most_common(3):
        if count < 2:
            break
        builder.add(name, kind="author", authors=[name])

    venue_counts = Counter(
        paper.venue.name for paper in seen if paper.venue and paper.venue.name.strip()
    )
    for name, count in venue_counts.most_common(2):
        if count < 3:
            break
        builder.add(name, kind="venue", venues=[name])

    queries, _ = builder.finish(MAX_FOLLOWUP_QUERIES)
    return queries


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _dedup_key(kind: str, text: str) -> str:
    return f"{kind}:{' '.join(text.lower().split())}"


class _PlanBuilder:
    """Accumulates queries in priority order, de-duplicated, then truncates once."""

    def __init__(self, *, limit: int, year_from: int | None, year_to: int | None) -> None:
        self.limit = limit
        self.year_from = year_from
        self.year_to = year_to
        self.queries: list[SourceQuery] = []
        self.seen: set[str] = set()

    def add(
        self,
        text: str,
        *,
        kind: Literal["keyword", "semantic", "author", "venue"] = "keyword",
        windowed: bool = True,
        keywords: list[str] | None = None,
        authors: list[str] | None = None,
        venues: list[str] | None = None,
    ) -> None:
        text = collapse_whitespace(text)
        if not text:
            return
        key = _dedup_key(kind, text)
        if key in self.seen:
            return
        self.seen.add(key)
        self.queries.append(
            SourceQuery(
                text=text,
                kind=kind,
                keywords=keywords or [],
                authors=authors or [],
                venues=venues or [],
                year_from=self.year_from if windowed else None,
                year_to=self.year_to if windowed else None,
                limit=self.limit,
            )
        )

    def finish(self, cap: int) -> tuple[list[SourceQuery], int]:
        dropped = max(0, len(self.queries) - cap)
        return self.queries[:cap], dropped


def _partition_terms(terms: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split ``author:``/``venue:`` markers out of free-text search terms."""
    plain: list[str] = []
    authors: list[str] = []
    venues: list[str] = []
    for raw in terms:
        term = raw.strip()
        if not term:
            continue
        lowered = term.lower()
        if lowered.startswith(_AUTHOR_PREFIX):
            authors.append(term[len(_AUTHOR_PREFIX) :].strip())
        elif lowered.startswith(_VENUE_PREFIX):
            venues.append(term[len(_VENUE_PREFIX) :].strip())
        else:
            plain.append(term)
    return plain, [a for a in authors if a], [v for v in venues if v]


def _detect_venues(text: str) -> list[str]:
    upper = text.upper()
    return [venue for venue in KNOWN_VENUES if venue.upper() in upper]


def _anchor_term(plan: QueryPlan) -> str:
    """The plan's dominant term, used to keep follow-ups on topic."""
    counts = Counter(token for query in plan.queries for token in tokenize(query.text))
    return counts.most_common(1)[0][0] if counts else ""


__all__ = [
    "KNOWN_VENUES",
    "MAX_FOLLOWUP_QUERIES",
    "MAX_QUERIES",
    "QueryPlan",
    "SourceQuery",
    "followup_queries",
    "plan_queries",
]
