"""Cross-source dedup: id cascade, title normalization, fuzzy author-year match,
and the merge that must never lose a source link."""

from __future__ import annotations

from difflib import SequenceMatcher

from lit_helpers import make_paper

from ml_research_agent.literature.dedupe import (
    dedupe,
    find_duplicates,
    merge_papers,
    normalize_arxiv_id,
    normalize_doi,
    normalize_title,
)
from ml_research_agent.types import Author, PaperSection, Provenance, Venue

# ---------------------------------------------------------------------------
# id-based matching
# ---------------------------------------------------------------------------


def test_arxiv_version_suffix_and_prefix_are_the_same_paper() -> None:
    papers = [
        make_paper(title="Sparse transformers", arxiv_id="2401.00001v2"),
        make_paper(title="Sparse Transformers (v3 preprint)", arxiv_id="arXiv:2401.00001"),
    ]
    assert normalize_arxiv_id("2401.00001v2") == normalize_arxiv_id("arXiv:2401.00001")
    merged = dedupe(papers)
    assert len(merged) == 1
    assert merged[0].arxiv_id == "2401.00001"


def test_doi_case_and_url_prefix_differences_dedupe() -> None:
    papers = [
        make_paper(title="A study of masks", doi="10.1000/ABC.123"),
        make_paper(title="Wholly different wording here", doi="https://doi.org/10.1000/abc.123"),
    ]
    assert normalize_doi("10.1000/ABC.123") == normalize_doi("https://doi.org/10.1000/abc.123")
    assert len(dedupe(papers)) == 1


def test_s2_id_match_dedupes() -> None:
    papers = [
        make_paper(title="Left hand title", s2_id="649def34f8be52c8b66281af98ae884c09aef38b"),
        make_paper(title="Right hand title", s2_id="649def34f8be52c8b66281af98ae884c09aef38b"),
    ]
    assert len(dedupe(papers)) == 1


def test_external_id_namespace_matches_a_strong_id() -> None:
    papers = [
        make_paper(title="Some paper", arxiv_id="2401.00009"),
        make_paper(title="Another rendering", external_ids={"arxiv": "2401.00009"}),
    ]
    assert len(dedupe(papers)) == 1


# ---------------------------------------------------------------------------
# title normalization
# ---------------------------------------------------------------------------


def test_normalize_title_folds_case_punctuation_accents_and_whitespace() -> None:
    assert normalize_title("Attention Is All You Need!") == "attention is all you need"
    assert normalize_title("Attention  is\tall\nyou need") == "attention is all you need"
    assert normalize_title("Schölkopf's Kernel Trick") == "scholkopf s kernel trick"
    assert normalize_title("Long-Context: A Re-Evaluation") == "long context a re evaluation"


def test_exact_normalized_title_dedupes_without_any_shared_id() -> None:
    papers = [
        make_paper(title="Attention Is All You Need"),
        make_paper(title="  attention is all you need!  "),
    ]
    assert not (papers[0].arxiv_id or papers[0].doi or papers[0].s2_id)
    assert len(dedupe(papers)) == 1


# ---------------------------------------------------------------------------
# fuzzy author-year matching
# ---------------------------------------------------------------------------


def test_near_identical_title_same_author_adjacent_years_is_one_record() -> None:
    left = make_paper(
        title="Block sparse attention for long context transformers",
        authors=[Author(name="Rewon Child")],
        year=2020,
    )
    right = make_paper(
        title="Block sparse attention for long context transformer",
        authors=[Author(name="Rewon Child"), Author(name="Scott Gray")],
        year=2021,
    )
    # Not the exact-title path: the fuzzy stage is what has to fire here.
    assert normalize_title(left.title) != normalize_title(right.title)
    assert len(dedupe([left, right])) == 1


def test_different_titles_by_the_same_author_stay_separate() -> None:
    papers = [
        make_paper(
            title="Block sparse attention for long context transformers",
            authors=[Author(name="Rewon Child")],
            year=2020,
        ),
        make_paper(
            title="Generative pretraining from pixels",
            authors=[Author(name="Rewon Child")],
            year=2020,
        ),
    ]
    assert len(dedupe(papers)) == 2


def test_a_near_identical_title_by_different_authors_is_not_fuzzy_matched() -> None:
    """The fuzzy stage is author-gated: title similarity alone must not merge."""
    left = make_paper(
        title="Block sparse attention for long context transformers",
        authors=[Author(name="Rewon Child")],
        year=2020,
    )
    right = make_paper(
        title="Block sparse attention for long context transformer",
        authors=[Author(name="Iz Beltagy")],
        year=2020,
    )
    assert normalize_title(left.title) != normalize_title(right.title)
    assert len(dedupe([left, right])) == 2


def test_a_shared_last_author_is_enough_to_bucket_for_the_fuzzy_stage() -> None:
    """Author order varies by source, so the last name is bucketed too."""
    left = make_paper(
        title="Block sparse attention for long context transformers",
        authors=[Author(name="Rewon Child"), Author(name="Ilya Sutskever")],
        year=2020,
    )
    right = make_paper(
        title="Block sparse attention for long context transformer",
        authors=[Author(name="Scott Gray"), Author(name="Ilya Sutskever")],
        year=2020,
    )
    assert len(dedupe([left, right])) == 1


def test_titles_too_short_to_compare_only_match_exactly() -> None:
    """``Bandits``/``Bandit`` clear the 0.92 ratio; the length floor is what saves them."""
    left = make_paper(title="Bandits", authors=[Author(name="Rewon Child")], year=2020)
    right = make_paper(title="Bandit", authors=[Author(name="Rewon Child")], year=2020)
    assert SequenceMatcher(None, "bandits", "bandit").ratio() >= 0.92
    assert len(dedupe([left, right])) == 2


def test_fuzzy_threshold_is_honoured() -> None:
    papers = [
        make_paper(
            title="Scaling laws for sparse attention in transformers",
            authors=[Author(name="Rewon Child")],
            year=2020,
        ),
        make_paper(
            title="Scaling laws of sparse attention for transformers",
            authors=[Author(name="Rewon Child")],
            year=2020,
        ),
    ]
    assert len(dedupe(papers)) == 2  # ~0.918, under the 0.92 default
    assert len(dedupe(papers, fuzzy_threshold=0.85)) == 1


def test_years_more_than_one_apart_are_not_fuzzy_matched() -> None:
    papers = [
        make_paper(
            title="Block sparse attention for long context transformers",
            authors=[Author(name="Rewon Child")],
            year=2019,
        ),
        make_paper(
            title="Block sparse attention for long context transformer",
            authors=[Author(name="Rewon Child")],
            year=2024,
        ),
    ]
    assert len(dedupe(papers)) == 2


# ---------------------------------------------------------------------------
# merging
# ---------------------------------------------------------------------------


def test_merge_papers_keeps_the_richest_record() -> None:
    incumbent = make_paper(
        title="Sparse attention",
        abstract="Short abstract.",
        authors=[Author(name="Rewon Child")],
        year=2020,
        arxiv_id="1904.10509",
        citation_count=10,
        reference_count=5,
        influential_citation_count=1,
        sources=["arxiv"],
        code_urls=["https://github.com/openai/sparse_attention"],
        external_ids={"arxiv": "1904.10509"},
        tags=["cs.LG"],
        venue=Venue(name="arXiv", kind="preprint"),
        provenance=[Provenance(source="arxiv:1904.10509")],
    )
    other = make_paper(
        title="Sparse attention with a longer, more informative title",
        abstract="A considerably longer abstract that says much more about the method.",
        authors=[Author(name="Rewon Child"), Author(name="Scott Gray")],
        year=2021,
        doi="10.1000/xyz",
        s2_id="s2-abc",
        citation_count=4200,
        reference_count=40,
        influential_citation_count=300,
        tldr="Sparse factorizations of the attention matrix.",
        sources=["semantic_scholar"],
        code_urls=["https://github.com/mirror/sparse"],
        external_ids={"CorpusId": "12345"},
        tags=["efficient-transformers"],
        venue=Venue(name="ICLR", kind="conference"),
        url="https://example.org/paper",
        pdf_url="https://example.org/paper.pdf",
        provenance=[Provenance(source="s2:s2-abc")],
    )

    merged = merge_papers(incumbent, other)

    assert merged.id == incumbent.id  # the incumbent's id survives
    assert merged.abstract == other.abstract  # longer abstract wins
    assert merged.title == other.title
    assert len(merged.authors) == 2  # more authors wins
    assert merged.year == 2020  # first appearance is the honest date
    assert merged.citation_count == 4200
    assert merged.reference_count == 40
    assert merged.influential_citation_count == 300
    assert merged.tldr == other.tldr
    assert merged.arxiv_id == "1904.10509"
    assert merged.doi == "10.1000/xyz"
    assert merged.s2_id == "s2-abc"
    assert merged.url == other.url and merged.pdf_url == other.pdf_url
    assert set(merged.sources) == {"arxiv", "semantic_scholar"}
    assert set(merged.code_urls) == set(incumbent.code_urls) | set(other.code_urls)
    assert merged.external_ids == {"arxiv": "1904.10509", "CorpusId": "12345"}
    assert set(merged.tags) == {"cs.LG", "efficient-transformers"}
    assert merged.venue is not None and merged.venue.name == "ICLR"
    assert len(merged.provenance) == 2


def test_merge_never_drops_a_source_link() -> None:
    a = make_paper(title="P", arxiv_id="2401.00021", sources=["arxiv"], code_urls=["https://a"])
    b = make_paper(
        title="P", arxiv_id="2401.00021", sources=["openreview"], code_urls=["https://b"]
    )
    c = make_paper(title="P", arxiv_id="2401.00021", sources=["paperswithcode"])

    merged = dedupe([a, b, c])
    assert len(merged) == 1
    assert merged[0].sources == ["arxiv", "openreview", "paperswithcode"]
    assert merged[0].code_urls == ["https://a", "https://b"]


# ---------------------------------------------------------------------------
# find_duplicates + ordering
# ---------------------------------------------------------------------------


def test_find_duplicates_returns_pairs_without_merging() -> None:
    papers = [
        make_paper(title="First", arxiv_id="2401.00030"),
        make_paper(title="Unrelated work on graph neural networks", doi="10.1/other"),
        make_paper(title="First again", arxiv_id="2401.00030"),
    ]
    before = [p.model_dump() for p in papers]
    pairs = find_duplicates(papers)
    assert pairs == [(0, 2)]
    # nothing was merged: the inputs are untouched, field for field
    assert [p.model_dump() for p in papers] == before
    assert len(papers) == 3


def test_find_duplicates_collapses_a_three_way_chain_onto_one_representative() -> None:
    papers = [
        make_paper(title="Chain head", arxiv_id="2401.00040"),
        make_paper(title="Chain middle", arxiv_id="2401.00040", doi="10.1000/chain"),
        make_paper(title="Chain tail", doi="10.1000/CHAIN"),
    ]
    assert find_duplicates(papers) == [(0, 1), (0, 2)]
    assert len(dedupe(papers)) == 1


def test_dedupe_preserves_first_seen_order() -> None:
    papers = [
        make_paper(title="Alpha", arxiv_id="2401.00051"),
        make_paper(title="Beta", arxiv_id="2401.00052"),
        make_paper(title="Alpha duplicate", arxiv_id="2401.00051"),
        make_paper(title="Gamma", arxiv_id="2401.00053"),
    ]
    merged = dedupe(papers)
    assert [p.arxiv_id for p in merged] == ["2401.00051", "2401.00052", "2401.00053"]


def test_merging_keeps_the_parsed_full_text_rather_than_the_abstract_only_record():
    """The richest-record rule has to cover parsed sections, not just metadata.

    A merge that dropped them would leave the paper abstract-only in the KB, and
    nothing downstream would notice: the Curator would simply write a thinner
    note, and every citation for that paper would quietly lose its locators.
    """
    parsed = make_paper(title="Attention Is All You Need", arxiv_id="1706.03762")
    parsed = parsed.model_copy(
        update={
            "sections": [
                PaperSection(title="Introduction", text="intro body", page_start=1, order=0),
                PaperSection(title="Method", text="method body", page_start=3, order=1),
            ],
            "full_text_hash": "abc123",
            "raw_path": "workspace/kb/raw/ab/abc123.pdf",
        }
    )
    abstract_only = make_paper(title="Attention Is All You Need", arxiv_id="1706.03762")
    assert abstract_only.sections == []

    for merged in (merge_papers(parsed, abstract_only), merge_papers(abstract_only, parsed)):
        assert [s.title for s in merged.sections] == ["Introduction", "Method"], (
            "the merge must keep the parsed full text whichever side it arrived on"
        )
        assert merged.full_text_hash == "abc123"
        assert merged.raw_path is not None
        assert merged.has_full_text
