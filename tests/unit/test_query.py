"""Query planning: what the brief expands into, the date window and its
seminal escape hatch, the caps, and follow-ups mined from early results."""

from __future__ import annotations

import pytest
from lit_helpers import make_brief, make_paper
from pydantic import ValidationError

from ml_research_agent.config import LiteratureConfig
from ml_research_agent.literature.query import (
    MAX_FOLLOWUP_QUERIES,
    MAX_QUERIES,
    QueryPlan,
    SourceQuery,
    followup_queries,
    plan_queries,
)
from ml_research_agent.types import Author, FalsifiableClaim, Paper, Venue

# Deliberately not the current year: a planner that read the wall clock instead
# of ``now_year`` would still pass the window assertions if these agreed.
NOW = 2030


def _plan(**brief_kw: object) -> QueryPlan:
    return plan_queries(make_brief(**brief_kw), LiteratureConfig(), now_year=NOW)


def _texts(plan: QueryPlan) -> list[str]:
    return [q.text for q in plan.queries]


# ---------------------------------------------------------------------------
# what the plan is derived from
# ---------------------------------------------------------------------------


def test_plan_draws_on_title_search_terms_claims_and_related_areas() -> None:
    plan = _plan(related_areas=["efficient transformers"])
    texts = _texts(plan)

    assert "Sparse attention for long context transformers" in texts  # the title
    assert "block sparse attention" in texts  # a search term
    assert any("efficient transformers" in text for text in texts)  # a related area
    # the claim contributes its content-bearing head, not the whole sentence
    keyword_texts = [q.text for q in plan.queries if q.kind == "keyword"]
    assert "block sparse attention matches dense perplexity" in keyword_texts
    # only the unwindowed semantic sweep carries the brief verbatim
    assert not any("32k" in text for text in keyword_texts)


def test_every_query_is_a_frozen_source_query() -> None:
    plan = _plan()
    assert plan.queries
    for query in plan.queries:
        assert isinstance(query, SourceQuery)
    with pytest.raises(ValidationError):
        plan.queries[0].text = "mutated"
    with pytest.raises(ValidationError):
        plan.queries[0].year_from = 1900


def test_plan_rationale_is_recorded() -> None:
    plan = _plan()
    assert str(NOW) in plan.rationale
    assert len(plan) == len(plan.queries)


# ---------------------------------------------------------------------------
# the date window and its escape hatch
# ---------------------------------------------------------------------------


def test_windowed_queries_carry_the_configured_window() -> None:
    config = LiteratureConfig()
    plan = plan_queries(make_brief(), config, now_year=NOW)
    windowed = [q for q in plan.queries if q.year_from is not None or q.year_to is not None]

    assert windowed
    for query in windowed:
        assert query.year_from == NOW - config.date_window_years
        assert query.year_to == NOW


def test_exactly_one_unwindowed_semantic_sweep_is_the_seminal_escape_hatch() -> None:
    plan = _plan()
    unwindowed = [q for q in plan.queries if q.year_from is None and q.year_to is None]

    assert len(unwindowed) == 1
    assert unwindowed[0].kind == "semantic"
    assert [q for q in plan.queries if q.kind == "semantic"] == unwindowed


def test_window_follows_the_configured_number_of_years() -> None:
    plan = plan_queries(make_brief(), LiteratureConfig(date_window_years=2), now_year=NOW)
    windowed = [q for q in plan.queries if q.year_from is not None]
    assert {q.year_from for q in windowed} == {NOW - 2}


def test_queries_carry_the_per_source_limit() -> None:
    plan = plan_queries(make_brief(), LiteratureConfig(per_source_limit=7), now_year=NOW)
    assert {q.limit for q in plan.queries} == {7}


# ---------------------------------------------------------------------------
# markers and venue detection
# ---------------------------------------------------------------------------


def test_author_and_venue_markers_become_typed_queries_with_the_marker_stripped() -> None:
    plan = _plan(search_terms=["author:Rewon Child", "venue:NeurIPS", "sparse attention"])

    authors = [q for q in plan.queries if q.kind == "author"]
    venues = [q for q in plan.queries if q.kind == "venue"]

    assert [q.text for q in authors] == ["Rewon Child"]
    assert authors[0].authors == ["Rewon Child"]
    assert "NeurIPS" in [q.text for q in venues]
    assert any(q.venues == ["NeurIPS"] for q in venues)
    # the marker never leaks into a keyword query
    assert not any(text.startswith(("author:", "venue:")) for text in _texts(plan))


def test_a_venue_named_in_the_brief_text_produces_a_venue_query() -> None:
    plan = _plan(
        problem_statement="Long context training is too expensive to reproduce at ICLR scale.",
    )
    venue_queries = [q for q in plan.queries if q.kind == "venue"]
    assert [q.text for q in venue_queries] == ["ICLR"]
    assert venue_queries[0].venues == ["ICLR"]


# ---------------------------------------------------------------------------
# caps and de-duplication
# ---------------------------------------------------------------------------


def test_plan_never_exceeds_the_cap_and_holds_no_duplicates() -> None:
    plan = _plan(
        search_terms=[f"search term number {i}" for i in range(20)],
        related_areas=[f"related area {i}" for i in range(10)],
        claims=[
            FalsifiableClaim(statement=f"Claim number {i} about sparsity", falsifier=f"not {i}")
            for i in range(6)
        ],
    )
    assert len(plan.queries) == MAX_QUERIES
    keys = [(q.kind, q.text.lower()) for q in plan.queries]
    assert len(set(keys)) == len(keys)
    assert "dropped" in plan.rationale


def test_repeated_terms_collapse_to_one_query() -> None:
    plan = _plan(search_terms=["block sparse attention", "Block Sparse Attention"])
    keyword_texts = [q.text.lower() for q in plan.queries if q.kind == "keyword"]
    assert keyword_texts.count("block sparse attention") == 1
    assert len(set(keyword_texts)) == len(keyword_texts)


# ---------------------------------------------------------------------------
# follow-ups
# ---------------------------------------------------------------------------


def _seen_papers() -> list[Paper]:
    """Three results sharing a vocabulary term, an author, and a venue."""
    return [
        make_paper(
            title="Longformer sliding window attention",
            authors=[Author(name="Iz Beltagy")],
            venue=Venue(name="EMNLP"),
            arxiv_id="2004.05150",
        ),
        make_paper(
            title="Sliding window routing for retrieval",
            authors=[Author(name="Iz Beltagy")],
            venue=Venue(name="EMNLP"),
            arxiv_id="2101.00001",
        ),
        make_paper(
            title="Window based memory compression",
            authors=[Author(name="Iz Beltagy")],
            venue=Venue(name="EMNLP"),
            arxiv_id="2102.00002",
        ),
    ]


def test_followups_mine_a_repeated_term_and_a_repeated_author() -> None:
    config = LiteratureConfig()
    plan = plan_queries(make_brief(), config, now_year=NOW)
    followups = followup_queries(plan, _seen_papers(), config)

    assert followups
    assert any("window" in q.text for q in followups if q.kind == "keyword")
    authors = [q for q in followups if q.kind == "author"]
    assert [q.text for q in authors] == ["Iz Beltagy"]
    assert authors[0].authors == ["Iz Beltagy"]
    assert len(followups) <= MAX_FOLLOWUP_QUERIES


def test_a_venue_carrying_three_results_is_mined_but_two_are_not() -> None:
    config = LiteratureConfig()
    plan = plan_queries(make_brief(), config, now_year=NOW)
    seen = _seen_papers()

    three = [q for q in followup_queries(plan, seen, config) if q.kind == "venue"]
    two = [q for q in followup_queries(plan, seen[:2], config) if q.kind == "venue"]

    assert [q.text for q in three] == ["EMNLP"]
    assert three[0].venues == ["EMNLP"]
    assert two == []


def test_followups_inherit_the_plans_window_and_per_source_limit() -> None:
    config = LiteratureConfig(per_source_limit=9)
    plan = plan_queries(make_brief(), config, now_year=NOW)
    followups = followup_queries(plan, _seen_papers(), config)

    assert followups
    for query in followups:
        assert query.limit == 9
        assert query.year_from == NOW - config.date_window_years
        assert query.year_to == NOW


def test_followups_never_repeat_a_query_already_in_the_plan() -> None:
    config = LiteratureConfig()
    plan = plan_queries(make_brief(), config, now_year=NOW)
    followups = followup_queries(plan, _seen_papers(), config)

    planned = {(q.kind, " ".join(q.text.lower().split())) for q in plan.queries}
    mined = {(q.kind, " ".join(q.text.lower().split())) for q in followups}
    assert planned.isdisjoint(mined)
    assert len(mined) == len(followups)


def test_followups_of_an_empty_result_set_are_empty() -> None:
    config = LiteratureConfig()
    plan = plan_queries(make_brief(), config, now_year=NOW)
    assert followup_queries(plan, [], config) == []


def test_a_term_seen_only_once_is_not_worth_a_follow_up() -> None:
    config = LiteratureConfig()
    plan = plan_queries(make_brief(), config, now_year=NOW)
    once = [make_paper(title="Kolmogorov arnold networks", arxiv_id="2404.19756")]
    followups = followup_queries(plan, once, config)
    assert not any("kolmogorov" in q.text.lower() for q in followups)
