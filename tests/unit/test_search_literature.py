"""The ``search_literature`` pipeline against in-memory sources.

Plan -> fan out -> dedupe -> rank -> snowball -> re-rank -> truncate, checked at
its seams: the stats it reports, the degradation when a source dies, the caps it
enforces and logs, and the determinism the reproducibility contract needs.
No adapter, no transport, no clock: ``now_year`` is pinned on every call.
"""

from __future__ import annotations

from typing import Any

import pytest
from lit_helpers import RecordingLogger, make_brief, off_topic_paper, on_topic_paper

from ml_research_agent.config import Config
from ml_research_agent.literature import search_literature
from ml_research_agent.literature.query import SourceQuery, plan_queries
from ml_research_agent.types import Author, Paper, RankedPaper, Venue

BRIEF = make_brief()
# Ranking, the date window and the snowball gate all read a year. Pin it, or the
# same survey stops being the same survey tomorrow.
NOW = 2030


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeSource:
    """An in-memory source: the same answer for every query, and it records them."""

    def __init__(
        self, name: str, papers: list[Paper], *, neighbours: list[Paper] | None = None
    ) -> None:
        self.name = name
        self.papers = list(papers)
        self.neighbours = list(neighbours or [])
        self.queries: list[SourceQuery] = []

    @property
    def call_count(self) -> int:
        return len(self.queries)

    def search(self, query: SourceQuery) -> list[Paper]:
        self.queries.append(query)
        return list(self.papers)

    def get(self, identifier: str) -> Paper | None:
        return None

    def references(self, paper: Paper) -> list[Paper]:
        return list(self.neighbours)

    def citations(self, paper: Paper) -> list[Paper]:
        return []


class BrokenSource(FakeSource):
    """A source whose search is down. One of those must not end a survey."""

    def search(self, query: SourceQuery) -> list[Paper]:
        self.queries.append(query)
        raise RuntimeError("upstream 503 from the search endpoint")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def lit_config(config: Config) -> Config:
    """tmp_path-scoped, with snowballing off so the accounting stays legible.

    The tests that care about expansion turn it back on explicitly.
    """
    return config.with_overrides(**{"literature.snowball_depth": 0})


def planned_query_count(config: Config) -> int:
    return len(plan_queries(BRIEF, config.literature, now_year=NOW).queries)


def run(config: Config, sources: Any, **kw: Any) -> tuple[list[RankedPaper], dict[str, Any]]:
    return search_literature(BRIEF, config, sources=sources, now_year=NOW, **kw)


def keys(ranked: list[RankedPaper]) -> list[str]:
    return [item.paper.key for item in ranked]


# ---------------------------------------------------------------------------
# the happy path
# ---------------------------------------------------------------------------


def test_two_sources_are_merged_deduped_ranked_and_accounted_for(lit_config: Config) -> None:
    shared = on_topic_paper(1)
    alpha = FakeSource("alpha", [on_topic_paper(0), shared])
    beta = FakeSource("beta", [shared, off_topic_paper(0)])
    queries = planned_query_count(lit_config)

    ranked, stats = run(lit_config, [alpha, beta])

    assert set(keys(ranked)) == {on_topic_paper(0).key, shared.key, off_topic_paper(0).key}
    assert [item.rank for item in ranked] == list(range(1, len(ranked) + 1))
    assert all(a.score >= b.score for a, b in zip(ranked, ranked[1:], strict=False))
    assert ranked[-1].paper.key == off_topic_paper(0).key  # the off-topic one ranks last

    assert stats["queries"] == queries
    assert stats["sources"] == {"alpha": 2 * queries, "beta": 2 * queries}
    assert stats["raw_candidates"] == 4 * queries
    assert stats["after_dedupe"] == 3
    assert stats["truncated"] == 0
    assert stats["returned"] == len(ranked) == 3
    assert set(stats["snowball"]) == {
        "rounds",
        "expanded",
        "kept",
        "yield_by_round",
        "truncated",
        "stopped_early",
    }


def test_the_returned_list_never_repeats_a_paper_key(lit_config: Config) -> None:
    overlap = [on_topic_paper(index) for index in range(3)]
    alpha = FakeSource("alpha", overlap)
    beta = FakeSource("beta", [*overlap, off_topic_paper(0)])

    ranked, _stats = run(lit_config, [alpha, beta])

    assert len(keys(ranked)) == len(set(keys(ranked))) == 4


def test_two_identical_calls_return_the_same_order(lit_config: Config) -> None:
    """Same brief, same sources, same year -> the same survey. Every time."""
    papers = [on_topic_paper(index) for index in range(4)] + [off_topic_paper(0)]

    first, first_stats = run(lit_config, [FakeSource("alpha", papers)])
    second, second_stats = run(lit_config, [FakeSource("alpha", papers)])

    assert keys(first) == keys(second)
    assert [item.score for item in first] == [item.score for item in second]
    assert first_stats == second_stats


# ---------------------------------------------------------------------------
# degradation
# ---------------------------------------------------------------------------


def test_a_dead_source_degrades_the_survey_instead_of_ending_it(lit_config: Config) -> None:
    working = FakeSource("alpha", [on_topic_paper(0)])
    broken = BrokenSource("broken", [on_topic_paper(9)])
    logger = RecordingLogger()
    queries = planned_query_count(lit_config)

    ranked, stats = run(lit_config, [working, broken], logger=logger)

    assert keys(ranked) == [on_topic_paper(0).key]
    assert stats["sources"] == {"alpha": queries, "broken": 0}
    failures = logger.events("literature.source_failed")
    assert len(failures) == queries
    assert {event["source"] for event in failures} == {"broken"}


def test_an_empty_source_list_returns_empty_rather_than_raising(lit_config: Config) -> None:
    logger = RecordingLogger()

    ranked, stats = run(lit_config, [], logger=logger)

    assert ranked == []
    assert stats["error"] == "no sources available"
    assert stats["sources"] == {}
    assert stats["queries"] == planned_query_count(lit_config)
    assert logger.events("literature.no_sources")


# ---------------------------------------------------------------------------
# the candidate cap
# ---------------------------------------------------------------------------


def test_max_candidates_is_enforced_logged_and_keeps_the_best(lit_config: Config) -> None:
    wanted = [on_topic_paper(index) for index in range(4)]
    unwanted = [off_topic_paper(index) for index in range(6)]
    config = lit_config.with_overrides(**{"literature.max_candidates": len(wanted)})
    logger = RecordingLogger()

    ranked, stats = run(config, [FakeSource("alpha", [*wanted, *unwanted])], logger=logger)

    assert len(ranked) == len(wanted)
    assert set(keys(ranked)) == {paper.key for paper in wanted}  # the cut is by score
    assert stats["after_dedupe"] == 10
    assert stats["truncated"] == 6
    assert stats["returned"] == 4

    events = logger.events("literature.truncated")
    assert events, f"expected a literature.truncated event, saw {logger.names}"
    assert events[0]["stage"] == "candidates"
    assert events[0]["kept"] == 4
    assert events[0]["dropped"] == 6
    assert events[0]["cap"] == 4


# ---------------------------------------------------------------------------
# snowballing is wired in
# ---------------------------------------------------------------------------


def test_a_neighbour_reached_by_snowballing_lands_in_the_ranked_list(config: Config) -> None:
    neighbour = on_topic_paper(500)
    source = FakeSource("alpha", [on_topic_paper(0)], neighbours=[neighbour])
    snowballing = config.with_overrides(**{"literature.snowball_depth": 1})

    ranked, stats = run(snowballing, [source])

    assert neighbour.key in keys(ranked)
    assert stats["after_dedupe"] == 1  # the neighbour arrived after the dedupe count
    assert stats["snowball"]["rounds"] == 1
    assert stats["snowball"]["kept"] == 1
    assert stats["returned"] == 2


def test_snowballing_off_reaches_no_neighbours(lit_config: Config) -> None:
    neighbour = on_topic_paper(500)
    source = FakeSource("alpha", [on_topic_paper(0)], neighbours=[neighbour])

    ranked, stats = run(lit_config, [source])

    assert neighbour.key not in keys(ranked)
    assert stats["snowball"]["rounds"] == 0


# ---------------------------------------------------------------------------
# follow-up queries
# ---------------------------------------------------------------------------


def _vocabulary_papers() -> list[Paper]:
    """Papers whose titles carry terms the brief never names.

    Follow-ups are the cheap half of recall: the vocabulary an area uses lives
    in its titles, so there has to be something there for the miner to find.
    """
    author = Author(name="Ada Lovelace")
    venue = Venue(name="ICLR", kind="conference")
    return [
        Paper(
            title="Block sparse attention with rotary landmark routing for long context",
            abstract="Rotary landmark routing keeps the perplexity of dense attention.",
            arxiv_id="2401.90001",
            authors=[author],
            venue=venue,
            citation_count=120,
        ),
        Paper(
            title="Rotary landmark routing beats dense attention at 32k context",
            abstract="A second look at rotary landmark routing for long context training.",
            arxiv_id="2401.90002",
            authors=[author],
            venue=venue,
            citation_count=80,
        ),
    ]


def test_follow_ups_are_off_by_default(lit_config: Config) -> None:
    source = FakeSource("alpha", _vocabulary_papers())
    logger = RecordingLogger()

    _ranked, stats = run(lit_config, [source], logger=logger)

    assert source.call_count == planned_query_count(lit_config)
    assert stats["queries"] == planned_query_count(lit_config)
    assert not logger.events("literature.followup_queries")


def test_follow_ups_issue_strictly_more_queries(lit_config: Config) -> None:
    baseline = FakeSource("alpha", _vocabulary_papers())
    expanded = FakeSource("alpha", _vocabulary_papers())
    logger = RecordingLogger()

    _base_ranked, base_stats = run(lit_config, [baseline])
    ranked, stats = run(lit_config, [expanded], logger=logger, include_followups=True)

    assert expanded.call_count > baseline.call_count
    assert stats["queries"] > base_stats["queries"]
    assert expanded.call_count == stats["queries"]
    assert stats["raw_candidates"] > base_stats["raw_candidates"]
    # More recall, not more papers: the extra queries hit the same two records.
    assert set(keys(ranked)) == {paper.key for paper in _vocabulary_papers()}

    events = logger.events("literature.followup_queries")
    assert events, f"expected a literature.followup_queries event, saw {logger.names}"
    assert events[0]["count"] == stats["queries"] - base_stats["queries"]
