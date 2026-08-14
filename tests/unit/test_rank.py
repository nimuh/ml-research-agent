"""Cheap candidate ranking: the five components, their monotonicity, the
seminal-work exemption, and deterministic ordering."""

from __future__ import annotations

import pytest
from lit_helpers import make_brief, make_paper

from ml_research_agent.config import LiteratureConfig
from ml_research_agent.literature.rank import WEIGHTS, rank_candidates, score_components
from ml_research_agent.types import Paper, Venue

# Deliberately not the current year: a component that read the wall clock
# instead of ``now_year`` would still pass if these agreed.
NOW = 2031
COMPONENTS = ("similarity", "citation_velocity", "recency", "venue", "code_availability")


def _components(paper: Paper, **cfg: object) -> dict[str, float]:
    return score_components(paper, make_brief(), LiteratureConfig(**cfg), now_year=NOW)


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def test_score_components_returns_all_five_signals_in_unit_range() -> None:
    paper = make_paper(
        title="Block sparse attention for long context transformers",
        abstract="We cut the quadratic cost of attention at 32k context.",
        year=NOW - 1,
        citation_count=120,
        venue=Venue(name="ICLR", kind="conference"),
        code_urls=["https://github.com/example/sparse"],
    )
    components = _components(paper)
    assert set(components) == set(COMPONENTS)
    for name, value in components.items():
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"


def test_weights_sum_to_one_and_score_is_their_weighted_sum() -> None:
    assert set(WEIGHTS) == set(COMPONENTS)
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    paper = make_paper(
        title="Long context attention with block sparsity",
        abstract="Sparse attention for transformers.",
        year=NOW - 2,
        citation_count=80,
        venue=Venue(name="NeurIPS", kind="conference"),
        code_urls=["https://github.com/example/x"],
    )
    ranked = rank_candidates([paper], make_brief(), LiteratureConfig(), now_year=NOW)[0]
    expected = sum(WEIGHTS[name] * value for name, value in ranked.components.items())
    assert ranked.score == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# component math
# ---------------------------------------------------------------------------


def test_similarity_rewards_papers_that_echo_the_brief() -> None:
    on_topic = make_paper(
        title="Block sparse attention for long context transformers",
        abstract="Sparse attention keeps dense perplexity at 32k context length.",
    )
    off_topic = make_paper(
        title="Protein folding from crystallography priors",
        abstract="We predict tertiary structure from amino acid sequence.",
    )
    assert _components(on_topic)["similarity"] > _components(off_topic)["similarity"]
    assert _components(off_topic)["similarity"] < 0.2


def test_citation_velocity_is_monotonic_in_citations_and_decreasing_in_age() -> None:
    velocities = [
        _components(make_paper(title="t", year=NOW - 1, citation_count=count))["citation_velocity"]
        for count in (0, 10, 100, 1000)
    ]
    assert velocities == sorted(velocities)
    assert velocities[0] == 0.0
    assert velocities[0] < velocities[1] < velocities[2]

    by_age = [
        _components(make_paper(title="t", year=year, citation_count=100))["citation_velocity"]
        for year in (NOW, NOW - 3, NOW - 11)
    ]
    assert by_age[0] > by_age[1] > by_age[2]


def test_code_availability_is_one_exactly_when_code_urls_are_present() -> None:
    assert _components(make_paper(title="t"))["code_availability"] == 0.0
    with_code = make_paper(title="t", code_urls=["https://github.com/example/x"])
    assert _components(with_code)["code_availability"] == 1.0


def test_venue_score_tiers_named_venues_above_bare_kinds() -> None:
    def venue_score(venue: Venue | None) -> float:
        return _components(make_paper(title="t", venue=venue))["venue"]

    top = venue_score(Venue(name="ICLR", kind="conference"))
    strong = venue_score(Venue(name="AISTATS", kind="conference"))
    unnamed_conference = venue_score(Venue(name="Regional Symposium on Widgets", kind="conference"))
    workshop = venue_score(Venue(name="Workshop on Widgets", kind="workshop"))
    preprint = venue_score(Venue(name="arXiv", kind="preprint"))
    missing = venue_score(None)

    assert top > strong > unnamed_conference > workshop > preprint > missing


def test_similarity_weights_the_title_above_the_abstract() -> None:
    """``TITLE_REPEAT`` exists so an on-topic title beats an on-topic aside."""
    in_title = make_paper(
        title="Block sparse attention for long context transformers",
        abstract="We fold proteins from crystallography priors on amino acid sequences.",
    )
    in_abstract = make_paper(
        title="Folding proteins from crystallography priors",
        abstract="Block sparse attention for long context transformers is unrelated to this.",
    )
    assert _components(in_title)["similarity"] > _components(in_abstract)["similarity"]


# ---------------------------------------------------------------------------
# the seminal exemption
# ---------------------------------------------------------------------------


def test_seminal_paper_is_exempt_from_the_date_window() -> None:
    config = LiteratureConfig()
    old_and_seminal = make_paper(
        title="Attention is all you need",
        year=NOW - 20,
        citation_count=config.seminal_citation_floor,
    )
    old_and_ignored = make_paper(
        title="Attention is all you need",
        year=NOW - 20,
        citation_count=config.seminal_citation_floor - 1,
    )
    seminal = score_components(old_and_seminal, make_brief(), config, now_year=NOW)
    ignored = score_components(old_and_ignored, make_brief(), config, now_year=NOW)

    assert seminal["recency"] == 1.0
    assert ignored["recency"] < 1.0
    assert ignored["recency"] == 0.0  # 20 years old against a 6-year window


def test_recency_decays_linearly_across_the_window() -> None:
    config = LiteratureConfig(date_window_years=10)
    scores = [
        score_components(make_paper(title="t", year=NOW - age), make_brief(), config, now_year=NOW)[
            "recency"
        ]
        for age in (0, 5, 10, 20)
    ]
    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == 0.0
    assert scores[3] == 0.0  # clamped, never negative


def test_unknown_year_gets_the_neutral_recency_prior() -> None:
    assert 0.0 < _components(make_paper(title="t", year=None))["recency"] < 1.0


def test_now_year_and_not_the_wall_clock_drives_the_dated_components() -> None:
    """Both dated components must move with ``now_year``, or ranking drifts daily."""
    paper = make_paper(title="t", year=2020, citation_count=100)
    config = LiteratureConfig()

    near = score_components(paper, make_brief(), config, now_year=2021)
    far = score_components(paper, make_brief(), config, now_year=2031)

    assert near["recency"] > far["recency"]
    assert near["citation_velocity"] > far["citation_velocity"]
    # and it is a pure function of its arguments
    assert near == score_components(paper, make_brief(), config, now_year=2021)


# ---------------------------------------------------------------------------
# ordering
# ---------------------------------------------------------------------------


def test_rank_candidates_sorts_descending_and_numbers_from_one() -> None:
    papers = [
        make_paper(title="Protein folding from crystallography priors", year=NOW - 2),
        make_paper(
            title="Block sparse attention for long context transformers",
            abstract="Sparse attention at 32k context.",
            year=NOW - 1,
            citation_count=400,
            code_urls=["https://github.com/example/x"],
        ),
        make_paper(title="A survey of database indexing", year=NOW - 2),
    ]
    ranked = rank_candidates(papers, make_brief(), LiteratureConfig(), now_year=NOW)

    assert [item.rank for item in ranked] == [1, 2, 3]
    assert [item.score for item in ranked] == sorted((i.score for i in ranked), reverse=True)
    assert ranked[0].paper.title.startswith("Block sparse attention")


def test_ties_break_deterministically_on_paper_key() -> None:
    def twin(arxiv_id: str) -> Paper:
        return make_paper(
            title="Block sparse attention for long context transformers",
            abstract="Identical signals; only the identifier differs.",
            year=NOW - 2,
            citation_count=50,
            venue=Venue(name="ICLR", kind="conference"),
            code_urls=["https://github.com/example/x"],
            arxiv_id=arxiv_id,
        )

    first, second = twin("2401.00001"), twin("2401.00002")
    forward = rank_candidates([first, second], make_brief(), LiteratureConfig(), now_year=NOW)
    backward = rank_candidates([second, first], make_brief(), LiteratureConfig(), now_year=NOW)

    assert forward[0].score == pytest.approx(forward[1].score)
    assert [item.paper.key for item in forward] == ["arxiv:2401.00001", "arxiv:2401.00002"]
    assert [item.paper.key for item in backward] == [item.paper.key for item in forward]


def test_ranking_is_empty_for_no_candidates() -> None:
    assert rank_candidates([], make_brief(), LiteratureConfig(), now_year=NOW) == []
