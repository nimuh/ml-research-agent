"""Citation snowballing against in-memory sources: the depth cap, the score
gate, the marginal-yield stop, the candidate cap, and degradation on failure."""

from __future__ import annotations

from typing import Any

from lit_helpers import RecordingLogger, make_brief, off_topic_paper, on_topic_paper

from ml_research_agent.config import LiteratureConfig
from ml_research_agent.literature.rank import rank_candidates
from ml_research_agent.literature.snowball import MAX_FRONTIER, MIN_MARGINAL_YIELD, snowball
from ml_research_agent.types import Paper

BRIEF = make_brief()
GATE = 0.55  # LiteratureConfig.snowball_min_score default
# Every ranking call is pinned to a fixed year: the gate must not depend on
# the day the suite runs.
NOW = 2030


class FakeGraphSource:
    """An in-memory citation graph. Every call mints fresh, unique neighbours.

    ``high``/``low`` control how many of each round's neighbours clear the score
    gate, which is what makes the yield-driven stop testable.
    """

    name = "fake"

    def __init__(self, *, high: int = 2, low: int = 0) -> None:
        self.high = high
        self.low = low
        self.calls: list[tuple[str, str]] = []
        self._minted = 0

    def search(self, query: Any) -> list[Paper]:  # pragma: no cover - unused by snowball
        return []

    def get(self, identifier: str) -> Paper | None:  # pragma: no cover - unused by snowball
        return None

    def references(self, paper: Paper) -> list[Paper]:
        return self._neighbours("references", paper)

    def citations(self, paper: Paper) -> list[Paper]:
        return self._neighbours("citations", paper)

    def _neighbours(self, direction: str, paper: Paper) -> list[Paper]:
        self.calls.append((direction, paper.key))
        batch: list[Paper] = []
        for _ in range(self.high):
            self._minted += 1
            batch.append(on_topic_paper(self._minted))
        for _ in range(self.low):
            self._minted += 1
            batch.append(off_topic_paper(self._minted))
        return batch


class BrokenReferencesSource(FakeGraphSource):
    """Forward edges work, backward edges blow up."""

    name = "broken"

    def references(self, paper: Paper) -> list[Paper]:
        raise RuntimeError("upstream 500 while reading references")


def _score(paper: Paper, config: LiteratureConfig) -> float:
    return rank_candidates([paper], BRIEF, config, now_year=NOW)[0].score


def test_the_fixtures_straddle_the_score_gate() -> None:
    """The gate tests are only meaningful if the fixtures actually straddle it."""
    config = LiteratureConfig()
    assert config.snowball_min_score == GATE
    assert _score(on_topic_paper(1), config) > GATE
    assert _score(off_topic_paper(1), config) < GATE


# ---------------------------------------------------------------------------
# brakes
# ---------------------------------------------------------------------------


def test_expansion_stops_at_the_configured_depth() -> None:
    config = LiteratureConfig(snowball_depth=2, max_candidates=500)
    source = FakeGraphSource(high=2)

    papers, stats = snowball([on_topic_paper(0)], BRIEF, [source], config, now_year=NOW)

    assert stats.rounds == 2
    assert len(stats.yield_by_round) == 2
    assert all(value >= MIN_MARGINAL_YIELD for value in stats.yield_by_round)
    assert len(papers) == 1 + 4 + 16  # seed, then 2 directions x 2 neighbours per frontier paper


def test_depth_zero_expands_nothing() -> None:
    source = FakeGraphSource(high=2)
    seed = on_topic_paper(0)

    papers, stats = snowball(
        [seed], BRIEF, [source], LiteratureConfig(snowball_depth=0), now_year=NOW
    )

    assert stats.rounds == 0
    assert source.calls == []
    assert [p.key for p in papers] == [seed.key]


def test_neighbours_below_the_score_gate_are_dropped() -> None:
    config = LiteratureConfig(snowball_depth=1, max_candidates=500)
    source = FakeGraphSource(high=1, low=3)

    papers, stats = snowball([on_topic_paper(0)], BRIEF, [source], config, now_year=NOW)

    kept_titles = [p.title for p in papers]
    assert stats.kept == 2  # one per direction
    assert not any(title.startswith("Protein folding") for title in kept_titles)
    assert all(_score(paper, config) >= config.snowball_min_score for paper in papers)


def test_a_cold_round_stops_the_expansion_before_the_depth_cap() -> None:
    config = LiteratureConfig(snowball_depth=3, max_candidates=500)
    source = FakeGraphSource(high=1, low=15)
    logger = RecordingLogger()

    _papers, stats = snowball(
        [on_topic_paper(0)], BRIEF, [source], config, logger=logger, now_year=NOW
    )

    assert stats.rounds == 1
    assert len(stats.yield_by_round) == 1
    assert stats.yield_by_round[0] < MIN_MARGINAL_YIELD
    assert stats.stopped_early
    assert logger.events("literature.snowball_stopped_early")


def test_a_dry_round_ends_the_expansion() -> None:
    config = LiteratureConfig(snowball_depth=3)
    source = FakeGraphSource(high=0, low=0)
    logger = RecordingLogger()

    papers, stats = snowball(
        [on_topic_paper(0)], BRIEF, [source], config, logger=logger, now_year=NOW
    )

    assert stats.rounds == 1
    assert stats.kept == 0
    assert len(papers) == 1
    assert logger.events("literature.snowball_dry_round")


# ---------------------------------------------------------------------------
# the candidate cap is enforced and logged
# ---------------------------------------------------------------------------


def test_max_candidates_is_never_exceeded_and_the_truncation_is_logged() -> None:
    config = LiteratureConfig(snowball_depth=2, max_candidates=3)
    source = FakeGraphSource(high=4)
    logger = RecordingLogger()

    papers, stats = snowball(
        [on_topic_paper(0)], BRIEF, [source], config, logger=logger, now_year=NOW
    )

    assert len(papers) <= config.max_candidates
    assert stats.truncated == 6  # 8 harvested past the gate, only 2 slots left

    events = logger.events("literature.truncated")
    assert events, f"expected a literature.truncated event, saw {logger.names}"
    assert events[0]["stage"] == "snowball"
    assert events[0]["dropped"] == 6
    assert events[0]["cap"] == config.max_candidates
    assert ("warning", "literature.truncated") in [(lvl, ev) for lvl, ev, _ in logger.records]
    # the second round never runs: there is no room left for what it would find
    assert stats.rounds == 1
    assert logger.events("literature.snowball_budget_exhausted")


def test_the_frontier_is_capped_and_the_cut_is_logged() -> None:
    """Breadth is a brake of its own: a round may keep more than it expands."""
    config = LiteratureConfig(snowball_depth=1, max_candidates=500)
    source = FakeGraphSource(high=13)  # 2 directions x 13 = 26 keepers, cap is 25
    logger = RecordingLogger()

    _papers, stats = snowball(
        [on_topic_paper(0)], BRIEF, [source], config, logger=logger, now_year=NOW
    )

    assert stats.kept == 26
    assert stats.truncated == 1
    events = logger.events("literature.truncated")
    assert [e["stage"] for e in events] == ["snowball_frontier"]
    assert events[0]["kept"] == MAX_FRONTIER
    assert events[0]["dropped"] == 1


# ---------------------------------------------------------------------------
# degradation
# ---------------------------------------------------------------------------


def test_a_source_that_raises_degrades_the_round_instead_of_propagating() -> None:
    config = LiteratureConfig(snowball_depth=1, max_candidates=500)
    source = BrokenReferencesSource(high=2)
    logger = RecordingLogger()

    papers, stats = snowball(
        [on_topic_paper(0)], BRIEF, [source], config, logger=logger, now_year=NOW
    )

    assert stats.kept == 2  # the working direction still contributed
    assert len(papers) == 3
    failures = logger.events("literature.snowball_edge_failed")
    assert [f["direction"] for f in failures] == ["references"]
    assert failures[0]["source"] == "broken"


def test_no_sources_returns_the_deduped_seeds_untouched() -> None:
    seeds = [on_topic_paper(0), on_topic_paper(0)]
    papers, stats = snowball(seeds, BRIEF, [], LiteratureConfig(), now_year=NOW)

    assert len(papers) == 1
    assert stats.rounds == 0


# ---------------------------------------------------------------------------
# output invariants
# ---------------------------------------------------------------------------


def test_result_contains_the_seeds_and_no_duplicate_keys() -> None:
    config = LiteratureConfig(snowball_depth=2, max_candidates=500)
    seeds = [on_topic_paper(0), on_topic_paper(900)]

    papers, _stats = snowball(seeds, BRIEF, [FakeGraphSource(high=2)], config, now_year=NOW)

    keys = [paper.key for paper in papers]
    assert len(keys) == len(set(keys))
    assert {seed.key for seed in seeds} <= set(keys)
