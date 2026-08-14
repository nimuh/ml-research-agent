"""Source adapters replayed against hand-authored cassettes.

No network, ever: every adapter is constructed with an injected transport, and
the player raises rather than inventing a reply, so an accidental live call is
a hard failure instead of a slow test.
"""

from __future__ import annotations

import pytest
from cassettes import CassettePlayer, ExplodingTransport

from ml_research_agent.config import Config
from ml_research_agent.errors import SourceError
from ml_research_agent.literature.query import SourceQuery
from ml_research_agent.literature.sources import available_sources
from ml_research_agent.literature.sources.arxiv import API_URL, ArxivSource
from ml_research_agent.literature.sources.semantic_scholar import (
    BASE_URL,
    SemanticScholarSource,
)
from ml_research_agent.types import Paper

SPARSE_S2_ID = "21da617a0f79aabf94272107184606cefe90ab75"


@pytest.fixture(autouse=True)
def _forbid_a_real_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt and braces: constructing a real client fails loudly.

    Every adapter here is handed a transport, so nothing should ever reach
    ``HttpClient._ensure_client``. If one day something does, this turns a
    silent live request into a test failure.
    """
    import httpx

    def _forbidden(*_args: object, **_kw: object) -> object:
        raise AssertionError("a real httpx.Client was constructed by a source adapter")

    monkeypatch.setattr(httpx, "Client", _forbidden)


@pytest.fixture
def lit_config(config: Config) -> Config:
    """The tmp_path-scoped Config, with the polite rate limiter turned off."""
    return config.with_overrides(**{"literature.rate_limit_per_second": 1000.0})


@pytest.fixture
def arxiv_player() -> CassettePlayer:
    return CassettePlayer.load("arxiv")


@pytest.fixture
def s2_player() -> CassettePlayer:
    return CassettePlayer.load("semantic_scholar")


def _arxiv_query() -> SourceQuery:
    return SourceQuery(
        text="sparse attention", kind="keyword", year_from=2019, year_to=2026, limit=5
    )


def _s2_query() -> SourceQuery:
    return SourceQuery(
        text="sparse attention", kind="keyword", year_from=2019, year_to=2026, limit=3
    )


# ---------------------------------------------------------------------------
# the player itself
# ---------------------------------------------------------------------------


def test_an_unrecorded_request_is_a_loud_failure(arxiv_player: CassettePlayer) -> None:
    with pytest.raises(AssertionError, match="no cassette entry"):
        arxiv_player.request("GET", "https://example.org/not-recorded")


def test_a_recorded_url_with_the_wrong_params_is_still_a_loud_failure(
    arxiv_player: CassettePlayer,
) -> None:
    """Matching on the url alone would let a mis-built query replay someone else's answer."""
    with pytest.raises(AssertionError, match="no cassette entry"):
        arxiv_player.request("GET", API_URL, params={"id_list": "9999.99999", "max_results": 1})


def test_a_non_retryable_status_raises_and_is_not_retried(lit_config: Config) -> None:
    player = CassettePlayer([{"name": "gone", "url": API_URL, "status": 404, "body": ""}])

    with pytest.raises(SourceError) as caught:
        ArxivSource(lit_config, http=player).search(_arxiv_query())

    assert caught.value.retryable is False
    assert player.request_count == 1  # a 404 is not worth three round trips


def test_a_malformed_json_body_raises_rather_than_being_read_as_empty(lit_config: Config) -> None:
    player = CassettePlayer(
        [
            {
                "name": "garbage",
                "url": f"{BASE_URL}/paper/search",
                "body": "<html>rate limited</html>",
            }
        ]
    )

    with pytest.raises(SourceError, match="malformed JSON"):
        SemanticScholarSource(lit_config, http=player).search(_s2_query())


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


def test_arxiv_search_maps_atom_entries_onto_papers(
    lit_config: Config, arxiv_player: CassettePlayer
) -> None:
    source = ArxivSource(lit_config, http=arxiv_player)
    papers = source.search(_arxiv_query())

    assert [p.arxiv_id for p in papers] == ["1904.10509", "2004.05150", "2007.14062"]
    assert "arxiv_search_sparse_attention" in arxiv_player.played

    sparse = papers[0]
    assert sparse.title == "Generating Long Sequences with Sparse Transformers"
    assert [a.name for a in sparse.authors] == [
        "Rewon Child",
        "Scott Gray",
        "Alec Radford",
        "Ilya Sutskever",
    ]
    assert sparse.year == 2019
    assert "quadratically with the sequence length" in sparse.abstract
    assert sparse.doi == "10.48550/arXiv.1904.10509"
    assert sparse.url == "https://arxiv.org/abs/1904.10509"
    assert sparse.pdf_url == "http://arxiv.org/pdf/1904.10509v1"
    assert sparse.tags == ["cs.LG", "stat.ML"]  # categories become tags
    assert sparse.venue is not None
    assert sparse.venue.name == "Proceedings of ICLR 2020"  # journal_ref becomes the venue
    assert sparse.venue.kind == "journal"
    assert sparse.sources == ["arxiv"]
    assert sparse.provenance and sparse.provenance[0].source == "arxiv:1904.10509"


def test_arxiv_strips_the_version_but_keeps_it_as_an_external_id(
    lit_config: Config, arxiv_player: CassettePlayer
) -> None:
    papers = ArxivSource(lit_config, http=arxiv_player).search(_arxiv_query())
    longformer = papers[1]

    assert longformer.arxiv_id == "2004.05150"
    assert longformer.external_ids["arxiv"] == "2004.05150"
    assert longformer.external_ids["arxiv_version"] == "2004.05150v2"
    assert longformer.key == "arxiv:2004.05150"


def test_arxiv_falls_back_to_the_preprint_venue_without_a_journal_ref(
    lit_config: Config, arxiv_player: CassettePlayer
) -> None:
    papers = ArxivSource(lit_config, http=arxiv_player).search(_arxiv_query())
    longformer = papers[1]

    assert longformer.venue is not None
    assert longformer.venue.name == "arXiv"
    assert longformer.venue.kind == "preprint"
    assert longformer.doi is None


def test_arxiv_get_by_identifier(lit_config: Config, arxiv_player: CassettePlayer) -> None:
    source = ArxivSource(lit_config, http=arxiv_player)

    paper = source.get("arXiv:1904.10509v1")

    assert paper is not None
    assert paper.arxiv_id == "1904.10509"
    assert paper.title == "Generating Long Sequences with Sparse Transformers"
    assert "arxiv_get_1904_10509" in arxiv_player.played


_EMPTY_FEED = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


def _recorded_arxiv_request(config: Config, query: SourceQuery) -> dict[str, str]:
    """Issue one search against a permissive cassette and return what was sent."""
    player = CassettePlayer([{"name": "any", "url": API_URL, "body": _EMPTY_FEED}])
    assert ArxivSource(config, http=player).search(query) == []
    return player.requests[0][2]


def test_arxiv_scopes_the_query_by_kind_and_windows_it_by_date(lit_config: Config) -> None:
    keyword = _recorded_arxiv_request(
        lit_config,
        SourceQuery(text="sparse attention", keywords=["long context"], year_from=2019, limit=5),
    )
    assert 'all:"sparse attention"' in keyword["search_query"]
    assert 'all:"long context"' in keyword["search_query"]  # keywords widen with OR
    assert "submittedDate:[20190101000000 TO " in keyword["search_query"]
    assert keyword["sortBy"] == "relevance"

    author = _recorded_arxiv_request(
        lit_config, SourceQuery(text="Rewon Child", kind="author", authors=["Rewon Child"])
    )
    assert author["search_query"] == 'au:"Rewon Child"'  # unwindowed query, no date clause
    assert author["sortBy"] == "submittedDate"

    venue = _recorded_arxiv_request(
        lit_config, SourceQuery(text="NeurIPS", kind="venue", venues=["NeurIPS"])
    )
    assert venue["search_query"] == 'all:"NeurIPS"'


def test_arxiv_refuses_to_issue_an_empty_query(lit_config: Config) -> None:
    with pytest.raises(SourceError, match="empty query"):
        ArxivSource(lit_config, http=ExplodingTransport()).search(SourceQuery(text="   "))


def test_arxiv_results_are_served_from_the_disk_cache_the_second_time(
    lit_config: Config, arxiv_player: CassettePlayer
) -> None:
    first = ArxivSource(lit_config, http=arxiv_player).search(_arxiv_query())
    assert arxiv_player.request_count == 1

    second = ArxivSource(lit_config, http=arxiv_player).search(_arxiv_query())

    # a *new* adapter over the same config: the cache is on disk, not in memory
    assert arxiv_player.request_count == 1
    assert [p.key for p in second] == [p.key for p in first]


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


def test_semantic_scholar_search_maps_graph_fields(
    lit_config: Config, s2_player: CassettePlayer
) -> None:
    papers = SemanticScholarSource(lit_config, http=s2_player).search(_s2_query())

    assert [p.title for p in papers] == [
        "Generating Long Sequences with Sparse Transformers",
        "Longformer: The Long-Document Transformer",
    ]

    sparse = papers[0]
    assert sparse.s2_id == SPARSE_S2_ID
    assert sparse.arxiv_id == "1904.10509"  # from externalIds.ArXiv
    assert sparse.doi == "10.48550/arXiv.1904.10509"
    assert sparse.external_ids["CorpusId"] == "129945531"
    assert sparse.citation_count == 1521
    assert sparse.reference_count == 41
    assert sparse.influential_citation_count == 213
    assert sparse.tldr is not None and sparse.tldr.startswith("Sparse factorizations")
    assert sparse.pdf_url == "https://arxiv.org/pdf/1904.10509.pdf"
    assert [a.author_id for a in sparse.authors][0] == "2058620"
    assert sparse.sources == ["semantic_scholar"]

    longformer = papers[1]
    assert longformer.arxiv_id == "2004.05150"  # the version suffix is normalized away
    assert longformer.tldr is None
    assert longformer.pdf_url is None
    assert longformer.venue is not None and longformer.venue.kind == "conference"


def test_semantic_scholar_references_and_citations_map_neighbours(
    lit_config: Config, s2_player: CassettePlayer
) -> None:
    source = SemanticScholarSource(lit_config, http=s2_player)
    seed = Paper(title="Generating Long Sequences with Sparse Transformers", s2_id=SPARSE_S2_ID)

    references = source.references(seed)
    citations = source.citations(seed)

    assert [p.title for p in references] == ["Attention is All you Need"]
    assert references[0].arxiv_id == "1706.03762"
    assert references[0].citation_count == 92117

    assert [p.title for p in citations] == [
        "Big Bird: Transformers for Longer Sequences",
        "Longformer: The Long-Document Transformer",
    ]
    assert set(s2_player.played) == {
        "s2_references_sparse_transformers",
        "s2_citations_sparse_transformers",
    }


def test_semantic_scholar_edges_are_cached_on_disk(
    lit_config: Config, s2_player: CassettePlayer
) -> None:
    seed = Paper(title="Generating Long Sequences with Sparse Transformers", s2_id=SPARSE_S2_ID)

    SemanticScholarSource(lit_config, http=s2_player).references(seed)
    assert s2_player.request_count == 1

    again = SemanticScholarSource(lit_config, http=s2_player).references(seed)

    assert s2_player.request_count == 1
    assert [p.title for p in again] == ["Attention is All you Need"]


def test_semantic_scholar_identifier_prefers_the_strongest_id() -> None:
    identifier = SemanticScholarSource.identifier_for
    assert identifier(Paper(title="t", s2_id="abc", arxiv_id="2401.1")) == "abc"
    assert identifier(Paper(title="t", arxiv_id="2401.00001v3")) == "arXiv:2401.00001"
    assert identifier(Paper(title="t", doi="10.1/x")) == "DOI:10.1/x"
    assert identifier(Paper(title="t")) is None


# ---------------------------------------------------------------------------
# registry + offline
# ---------------------------------------------------------------------------


def test_available_sources_skips_one_that_cannot_be_constructed(
    lit_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kw: object) -> ArxivSource:
        raise RuntimeError("missing credential")

    monkeypatch.setattr("ml_research_agent.literature.sources.arxiv.ArxivSource", boom)
    config = lit_config.with_overrides(**{"literature.sources": ["arxiv", "semantic_scholar"]})

    sources = available_sources(config, http=ExplodingTransport())

    assert [s.name for s in sources] == ["semantic_scholar"]


def test_available_sources_returns_every_healthy_source_in_config_order(
    lit_config: Config,
) -> None:
    config = lit_config.with_overrides(**{"literature.sources": ["semantic_scholar", "arxiv"]})

    sources = available_sources(config, http=ExplodingTransport())

    assert [s.name for s in sources] == ["semantic_scholar", "arxiv"]


def test_offline_mode_with_an_empty_cache_returns_nothing_and_never_calls_out(
    lit_config: Config,
) -> None:
    offline = lit_config.with_overrides(offline=True)
    transport = ExplodingTransport()

    assert ArxivSource(offline, http=transport).search(_arxiv_query()) == []
    assert SemanticScholarSource(offline, http=transport).search(_s2_query()) == []
    assert ArxivSource(offline, http=transport).get("1904.10509") is None
    assert transport.calls == 0


def test_offline_mode_still_serves_a_warm_cache(
    lit_config: Config, arxiv_player: CassettePlayer
) -> None:
    warm = ArxivSource(lit_config, http=arxiv_player).search(_arxiv_query())
    assert warm

    offline = lit_config.with_overrides(offline=True)
    replayed = ArxivSource(offline, http=ExplodingTransport()).search(_arxiv_query())

    assert [p.key for p in replayed] == [p.key for p in warm]
