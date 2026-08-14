"""Citation snowballing: expand backward (references) and forward (citations)
from a seed set, breadth-limited and score-gated, until marginal yield drops."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import LiteratureConfig
from ..types import Paper, ResearchBrief
from ._log import emit, truncated
from .dedupe import dedupe
from .rank import rank_candidates
from .sources import Source

# Below this fraction of expanded-to-kept, another round is paying full price
# for papers the gate will reject anyway.
MIN_MARGINAL_YIELD = 0.15
# How many papers get expanded per round. Two directions x many sources means
# one round of 25 seeds is already ~50 requests.
MAX_FRONTIER = 25


@dataclass
class SnowballStats:
    """What the expansion cost and what it bought."""

    rounds: int = 0
    expanded: int = 0
    kept: int = 0
    yield_by_round: list[float] = field(default_factory=list)
    truncated: int = 0

    @property
    def stopped_early(self) -> bool:
        return bool(self.yield_by_round) and self.yield_by_round[-1] < MIN_MARGINAL_YIELD


def snowball(
    seeds: Sequence[Paper],
    brief: ResearchBrief,
    sources: Sequence[Source],
    config: LiteratureConfig,
    *,
    logger: Any | None = None,
    now_year: int | None = None,
) -> tuple[list[Paper], SnowballStats]:
    """Expand backward and forward from ``seeds`` until it stops paying.

    Three independent brakes, because an unbounded citation graph will happily
    consume a whole budget: depth (``snowball_depth``), breadth (``MAX_FRONTIER``
    per round), and a score gate (``snowball_min_score``). Marginal yield stops
    it earlier than any of them when the frontier goes cold.

    ``now_year`` is threaded into ranking so the gate does not silently depend
    on the wall clock: the same seeds must expand the same way on any day.
    """
    stats = SnowballStats()
    accumulated = dedupe(seeds)
    seen: set[str] = {paper.key for paper in accumulated}

    if not accumulated or not sources or config.snowball_depth <= 0:
        return accumulated, stats

    ordered = rank_candidates(accumulated, brief, config, now_year=now_year)
    frontier = [item.paper for item in ordered][:MAX_FRONTIER]
    for depth in range(config.snowball_depth):
        remaining = config.max_candidates - len(accumulated)
        if not frontier or remaining <= 0:
            if remaining <= 0:
                emit(logger, "info", "literature.snowball_budget_exhausted", round=depth + 1)
            break

        harvested = _expand(frontier, sources, logger=logger)
        stats.rounds += 1
        # Distinct neighbours, not raw rows: every extra source returns the same
        # citation list, and dividing yield by the source count would stop the
        # expansion for a reason that has nothing to do with the frontier.
        distinct = dedupe(harvested)
        stats.expanded += len(distinct)
        if not distinct:
            stats.yield_by_round.append(0.0)
            emit(logger, "info", "literature.snowball_dry_round", round=depth + 1)
            break

        fresh = [paper for paper in distinct if paper.key not in seen]
        ranked = rank_candidates(fresh, brief, config, now_year=now_year)
        gated = [item.paper for item in ranked if item.score >= config.snowball_min_score]

        kept = gated[:remaining]
        dropped = len(gated) - len(kept)
        if dropped:
            stats.truncated += dropped
            truncated(
                logger,
                stage="snowball",
                kept=len(kept),
                dropped=dropped,
                cap=config.max_candidates,
                round=depth + 1,
            )

        accumulated.extend(kept)
        seen.update(paper.key for paper in kept)
        stats.kept += len(kept)

        round_yield = len(kept) / len(distinct)
        stats.yield_by_round.append(round(round_yield, 4))
        emit(
            logger,
            "info",
            "literature.snowball_round",
            round=depth + 1,
            harvested=len(harvested),
            expanded=len(distinct),
            fresh=len(fresh),
            kept=len(kept),
            gate=config.snowball_min_score,
            marginal_yield=round(round_yield, 4),
        )

        if round_yield < MIN_MARGINAL_YIELD:
            emit(
                logger,
                "info",
                "literature.snowball_stopped_early",
                round=depth + 1,
                marginal_yield=round(round_yield, 4),
                threshold=MIN_MARGINAL_YIELD,
            )
            break

        if len(kept) > MAX_FRONTIER:
            stats.truncated += len(kept) - MAX_FRONTIER
            truncated(
                logger,
                stage="snowball_frontier",
                kept=MAX_FRONTIER,
                dropped=len(kept) - MAX_FRONTIER,
                cap=MAX_FRONTIER,
                round=depth + 1,
            )
        frontier = kept[:MAX_FRONTIER]

    return accumulated, stats


def _expand(
    frontier: Sequence[Paper], sources: Sequence[Source], *, logger: Any | None
) -> list[Paper]:
    """Both directions, over every source that supports them.

    A source that raises is dropped for this paper only: losing one edge list
    must degrade the expansion, never end it.
    """
    harvested: list[Paper] = []
    for paper in frontier:
        for source in sources:
            for direction in ("references", "citations"):
                try:
                    neighbours = getattr(source, direction)(paper)
                except Exception as exc:  # noqa: BLE001 - one bad edge list is survivable
                    emit(
                        logger,
                        "warning",
                        "literature.snowball_edge_failed",
                        source=getattr(source, "name", "unknown"),
                        direction=direction,
                        paper=paper.key,
                        error=str(exc),
                    )
                    continue
                harvested.extend(neighbours)
    return harvested


__all__ = ["MAX_FRONTIER", "MIN_MARGINAL_YIELD", "SnowballStats", "snowball"]
