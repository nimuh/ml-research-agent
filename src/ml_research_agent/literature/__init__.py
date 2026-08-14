"""Literature acquisition: query expansion, multi-source retrieval, dedup,
ranking, citation snowballing, and full-text fetch/parse. Returns Paper
objects; it does not decide relevance (that is the Curator's job)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from ..config import Config
from ..types import Paper, RankedPaper, ResearchBrief
from ..utils.concurrency import gather_bounded
from ._log import emit, truncated
from .dedupe import dedupe, find_duplicates, merge_papers, normalize_title, paper_key
from .fetch import ContentFetcher, FetchedDocument
from .parse import extract_references, parse_document, parse_html, parse_latex, parse_pdf
from .query import QueryPlan, SourceQuery, followup_queries, plan_queries
from .rank import rank_candidates, score_components
from .snowball import SnowballStats, snowball
from .sources import BaseSource, Source, available_sources, get_source

# One in-flight request per source, roughly: sources rate-limit per client, so
# more parallelism here buys 429s rather than speed.
MAX_PARALLEL_SEARCHES = 6
# Only the best candidates are worth paying two graph requests each for.
MAX_SNOWBALL_SEEDS = 15


def search_literature(
    brief: ResearchBrief,
    config: Config,
    *,
    sources: Sequence[Source] | None = None,
    logger: Any | None = None,
    include_followups: bool = False,
    now_year: int | None = None,
) -> tuple[list[RankedPaper], dict[str, Any]]:
    """Plan -> fan out -> dedupe -> rank -> snowball -> re-rank -> truncate.

    The Scout's one entry point into this layer. It returns *candidates* with a
    cheap score and the stats behind them; nothing here judges relevance, and a
    single failing source degrades the survey rather than ending it.

    ``include_followups`` adds one extra round of queries mined from the first
    pass -- off by default because it doubles the request count. ``now_year``
    pins the date window and the recency score so a survey is reproducible.
    """
    settings = config.literature
    plan = plan_queries(brief, settings, now_year=now_year)
    active = list(sources) if sources is not None else available_sources(config, logger=logger)
    if not active:
        emit(logger, "error", "literature.no_sources", configured=settings.sources)
        return [], {"queries": len(plan.queries), "sources": {}, "error": "no sources available"}

    per_source, harvested = _fan_out(active, plan.queries, logger=logger)
    if include_followups:
        extra = followup_queries(plan, harvested, settings)
        if extra:
            emit(logger, "info", "literature.followup_queries", count=len(extra))
            more_counts, more_papers = _fan_out(active, extra, logger=logger)
            harvested.extend(more_papers)
            for name, count in more_counts.items():
                per_source[name] = per_source.get(name, 0) + count
            plan = QueryPlan(queries=[*plan.queries, *extra], rationale=plan.rationale)

    candidates = dedupe(harvested)
    emit(
        logger,
        "info",
        "literature.retrieved",
        raw=len(harvested),
        deduped=len(candidates),
        queries=len(plan.queries),
        sources=len(active),
    )

    ranked = rank_candidates(candidates, brief, settings, now_year=now_year)
    seeds = [item.paper for item in ranked[:MAX_SNOWBALL_SEEDS]]
    expanded, stats = snowball(seeds, brief, active, settings, logger=logger, now_year=now_year)

    combined = dedupe([*candidates, *expanded])
    final = rank_candidates(combined, brief, settings, now_year=now_year)
    dropped = max(0, len(final) - settings.max_candidates)
    if dropped:
        truncated(
            logger,
            stage="candidates",
            kept=settings.max_candidates,
            dropped=dropped,
            cap=settings.max_candidates,
        )
        final = final[: settings.max_candidates]

    summary: dict[str, Any] = {
        "queries": len(plan.queries),
        "rationale": plan.rationale,
        "sources": per_source,
        "raw_candidates": len(harvested),
        "after_dedupe": len(candidates),
        "snowball": {
            "rounds": stats.rounds,
            "expanded": stats.expanded,
            "kept": stats.kept,
            "yield_by_round": stats.yield_by_round,
            "truncated": stats.truncated,
            "stopped_early": stats.stopped_early,
        },
        "truncated": dropped,
        "returned": len(final),
    }
    return final, summary


def _fan_out(
    sources: Sequence[Source], queries: Sequence[SourceQuery], *, logger: Any | None
) -> tuple[dict[str, int], list[Paper]]:
    """Run every (source, query) pair with bounded concurrency."""
    jobs = [(source, query) for source in sources for query in queries]
    if not jobs:
        return {}, []

    def _factory(source: Source, query: SourceQuery) -> Callable[[], Awaitable[list[Paper]]]:
        return lambda: asyncio.to_thread(_search_one, source, query, logger)

    async def _run() -> list[list[Paper] | BaseException]:
        return await gather_bounded(
            [_factory(source, query) for source, query in jobs], limit=MAX_PARALLEL_SEARCHES
        )

    try:
        results = asyncio.run(_run())
    except RuntimeError:
        # Already inside an event loop (or no thread support): sequential is
        # slower but must still work -- a survey is not allowed to just fail.
        results = [_search_one(source, query, logger) for source, query in jobs]

    per_source: dict[str, int] = {getattr(s, "name", "unknown"): 0 for s in sources}
    harvested: list[Paper] = []
    for (source, _query), result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException) or not isinstance(result, list):
            continue
        name = getattr(source, "name", "unknown")
        per_source[name] = per_source.get(name, 0) + len(result)
        harvested.extend(result)
    return per_source, harvested


def _search_one(source: Source, query: SourceQuery, logger: Any | None) -> list[Paper]:
    try:
        return list(source.search(query))
    except Exception as exc:  # noqa: BLE001 - one dead source must not end the survey
        emit(
            logger,
            "warning",
            "literature.source_failed",
            source=getattr(source, "name", "unknown"),
            query=query.text[:80],
            kind=query.kind,
            error=str(exc),
        )
        return []


__all__ = [
    "MAX_PARALLEL_SEARCHES",
    "MAX_SNOWBALL_SEEDS",
    "BaseSource",
    "ContentFetcher",
    "FetchedDocument",
    "QueryPlan",
    "SnowballStats",
    "Source",
    "SourceQuery",
    "available_sources",
    "dedupe",
    "extract_references",
    "find_duplicates",
    "followup_queries",
    "get_source",
    "merge_papers",
    "normalize_title",
    "paper_key",
    "parse_document",
    "parse_html",
    "parse_latex",
    "parse_pdf",
    "plan_queries",
    "rank_candidates",
    "score_components",
    "search_literature",
    "snowball",
]
