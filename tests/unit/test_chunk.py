"""Chunking: overlap, section boundaries, locators, deterministic ids."""

from __future__ import annotations

from ml_research_agent.config import KnowledgeConfig
from ml_research_agent.knowledge.chunk import chunk_sections, chunk_text, estimate_tokens
from ml_research_agent.types import PaperSection

SMALL = KnowledgeConfig(chunk_tokens=64, chunk_overlap=16)


def _paragraphs(marker: str, n: int) -> str:
    return "\n\n".join(
        f"{marker} paragraph number {i} with enough words to cost tokens." for i in range(n)
    )


def test_estimate_tokens_is_about_four_chars_per_token():
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 401) == 101
    assert estimate_tokens("x" * 100) < estimate_tokens("x" * 200)


def test_chunk_text_respects_the_token_budget():
    passages = chunk_text(_paragraphs("alpha", 12), "doc_1", SMALL)
    assert len(passages) > 1
    # Every atom here fits the window, so no chunk has an excuse to overrun it.
    assert max(p.token_count for p in passages) <= SMALL.chunk_tokens
    assert [p.order for p in passages] == list(range(len(passages)))


def test_chunks_overlap_at_the_seam():
    passages = chunk_text(_paragraphs("beta", 12), "doc_1", SMALL)
    tail = passages[0].text.split("\n\n")[-1]
    assert tail in passages[1].text


def test_zero_overlap_shares_nothing():
    config = KnowledgeConfig(chunk_tokens=64, chunk_overlap=0)
    passages = chunk_text(_paragraphs("gamma", 12), "doc_1", config)
    seen: set[str] = set()
    for passage in passages:
        atoms = set(passage.text.split("\n\n"))
        assert not (atoms & seen)
        seen |= atoms


def test_chunks_never_cross_a_section_boundary():
    sections = [
        PaperSection(title="Method", text=_paragraphs("methodword", 10), order=0, page_start=3),
        PaperSection(title="Results", text=_paragraphs("resultword", 10), order=1, page_start=7),
    ]
    passages = chunk_sections(sections, "doc_1", SMALL)
    assert len(passages) > 2
    for passage in passages:
        has_method = "methodword" in passage.text
        has_result = "resultword" in passage.text
        assert has_method != has_result
        assert passage.section == ("Method" if has_method else "Results")
        assert passage.page == (3 if has_method else 7)


def test_locators_and_hashes_are_populated():
    sections = [PaperSection(title="Intro", text="A short intro.", order=0, page_start=1)]
    (passage,) = chunk_sections(sections, "doc_1", SMALL)
    assert passage.section == "Intro"
    assert passage.page == 1
    assert passage.token_count > 0
    assert passage.content_hash
    assert passage.locator == "Intro p.1"
    assert passage.to_provenance().locator == "Intro p.1"


def test_ids_are_deterministic_so_re_ingest_upserts():
    sections = [PaperSection(title="Method", text=_paragraphs("delta", 8), order=0)]
    first = chunk_sections(sections, "doc_1", SMALL)
    second = chunk_sections(sections, "doc_1", SMALL)
    assert [p.id for p in first] == [p.id for p in second]
    other_doc = chunk_sections(sections, "doc_2", SMALL)
    assert {p.id for p in first}.isdisjoint({p.id for p in other_doc})


def test_a_caption_stays_with_the_text_it_describes():
    text = (
        "We evaluate the model on three benchmarks and report the mean.\n\n"
        "Table 2: accuracy per benchmark.\n\n"
        "The table shows that the treatment wins on two of the three."
    )
    passages = chunk_text(text, "doc_1", KnowledgeConfig(chunk_tokens=64, chunk_overlap=0))
    with_caption = [p for p in passages if "Table 2" in p.text]
    assert len(with_caption) == 1
    assert "shows that the treatment" in with_caption[0].text
    assert with_caption[0].metadata["has_caption"] is True


def test_an_oversized_paragraph_is_split_on_word_boundaries():
    long_paragraph = " ".join(f"word{i}" for i in range(400))
    passages = chunk_text(long_paragraph, "doc_1", SMALL)
    assert len(passages) > 1
    assert max(p.token_count for p in passages) <= SMALL.chunk_tokens
    rejoined = {w for p in passages for w in p.text.replace("\n\n", " ").split()}
    assert rejoined == {f"word{i}" for i in range(400)}


def test_empty_text_produces_no_passages():
    assert chunk_text("   \n\n  ", "doc_1", SMALL) == []
    assert chunk_sections([PaperSection(title="Empty", text="")], "doc_1", SMALL) == []
