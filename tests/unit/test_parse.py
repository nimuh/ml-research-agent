"""Document parsing: PDF, LaTeX, HTML, reference splitting, and byte-level
dispatch. Every input is synthesized in-memory -- nothing is fetched."""

from __future__ import annotations

import io
import tarfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from ml_research_agent.literature.fetch import FetchedDocument
from ml_research_agent.literature.parse import (
    extract_references,
    parse_document,
    parse_html,
    parse_latex,
    parse_pdf,
)
from ml_research_agent.types import PaperSection

# ---------------------------------------------------------------------------
# fixtures built in memory
# ---------------------------------------------------------------------------


def _synthetic_pdf() -> bytes:
    """A two-page paper: title, numbered headings, a caption, a bibliography.

    ``pymupdf`` is skipped from *inside* the PDF helpers rather than at module
    import: a module-level skip would silently take the LaTeX, HTML and
    reference tests down with it.
    """
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    first = doc.new_page()
    first.insert_text((72, 80), "Scaling Laws for Sparse Attention", fontsize=18, fontname="hebo")
    first.insert_text((72, 110), "Jane Doe, John Roe", fontsize=10)
    first.insert_text((72, 150), "1 Introduction", fontsize=12, fontname="hebo")
    first.insert_text(
        (72, 170), "Sparse attention reduces the quadratic cost of self attention.", fontsize=10
    )
    first.insert_text(
        (72, 185), "We study how the trade off scales with sequence length.", fontsize=10
    )
    first.insert_text(
        (72, 220), "Figure 1: Loss versus sequence length for three sparsity levels.", fontsize=9
    )
    first.insert_text((72, 250), "The figure caption is followed by more prose.", fontsize=10)

    second = doc.new_page()
    second.insert_text((72, 80), "2 Method", fontsize=12, fontname="hebo")
    second.insert_text(
        (72, 100), "We train a decoder only transformer with block sparse masks.", fontsize=10
    )
    second.insert_text((72, 140), "REFERENCES", fontsize=12, fontname="hebo")
    second.insert_text(
        (72, 160), "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.", fontsize=9
    )
    data: bytes = doc.tobytes()
    doc.close()
    return data


MAIN_TEX = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\newcommand{\ours}{SparseNet}
\title{Scaling Laws for Sparse Attention}
\author{Jane Doe}
\begin{document}
\maketitle
\begin{abstract}
We study how sparse attention trades accuracy for compute at long context lengths.
\end{abstract}
\section{Introduction}
Dense attention is quadratic \cite{vaswani2017}, which motivates sparsity \citep{child2019}.
\subsection{Scope}
We restrict attention to decoder only models.
\begin{figure}
\includegraphics{loss.png}
\caption{Loss versus sequence length for three sparsity levels.}
\end{figure}
\section{Method}
We train with block sparse masks.
\begin{table}
\caption{Validation perplexity by sparsity level.}
\end{table}
\end{document}
"""

RESULTS_TEX = r"""
\section{Results}
Sparsity of one half matches the dense baseline \cite{child2019}.
"""

HTML_PAGE = b"""<html><head><style>p {color: red}</style></head><body>
<h1>Scaling Laws for Sparse Attention</h1>
<p>Dense attention is quadratic in sequence length.</p>
<h2>Method</h2>
<p>We train block sparse transformers.</p>
<figure><img src="loss.png"><figcaption>Figure 2: Loss versus sequence length.</figcaption></figure>
<p>The trend continues past 32k tokens.</p>
<script>var tracker = 1;</script>
</body></html>"""


def _tar_gz(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in files.items():
            payload = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _titles(sections: Sequence[PaperSection]) -> list[str]:
    return [section.title for section in sections]


def _find(sections: Sequence[PaperSection], title: str) -> PaperSection:
    matches = [section for section in sections if section.title == title]
    assert matches, f"no section titled {title!r} in {_titles(sections)}"
    return matches[0]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def test_parse_pdf_detects_sections_pages_and_captions() -> None:
    sections = parse_pdf(_synthetic_pdf())
    titles = _titles(sections)

    assert "1 Introduction" in titles
    assert "2 Method" in titles
    assert "REFERENCES" in titles

    intro = _find(sections, "1 Introduction")
    assert "quadratic cost" in intro.text
    assert "block sparse masks" not in intro.text  # body landed under the right heading
    assert intro.page_start == 1

    method = _find(sections, "2 Method")
    assert "block sparse masks" in method.text
    assert method.page_start == 2  # page tracking survives the page break

    references = _find(sections, "REFERENCES")
    assert "Vaswani" in references.text


def test_parse_pdf_gives_a_caption_its_own_section_and_returns_to_the_enclosing_one() -> None:
    sections = parse_pdf(_synthetic_pdf())

    caption = _find(sections, "Figure 1")
    assert caption.level == 3
    assert caption.text.startswith("Figure 1:")
    assert "trade off scales" not in caption.text

    # the prose after the caption is filed back under the enclosing heading
    intro_sections = [s for s in sections if s.title == "1 Introduction"]
    assert len(intro_sections) == 2
    assert "followed by more prose" in intro_sections[1].text


def test_parse_pdf_orders_sections_monotonically() -> None:
    sections = parse_pdf(_synthetic_pdf())
    orders = [section.order for section in sections]
    assert orders == sorted(orders)
    assert orders == list(range(len(sections)))
    assert all(section.page_start is not None for section in sections)


def test_parse_pdf_of_unreadable_bytes_raises_a_source_error() -> None:
    from ml_research_agent.errors import SourceError

    pytest.importorskip("pymupdf")  # otherwise this passes for the wrong reason
    with pytest.raises(SourceError, match="unreadable pdf"):
        parse_pdf(b"not a pdf at all")


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------


def test_parse_latex_bundle_reads_abstract_sections_levels_and_captions() -> None:
    bundle = _tar_gz({"paper/main.tex": MAIN_TEX, "paper/sections/results.tex": RESULTS_TEX})
    sections = parse_latex(bundle)
    titles = _titles(sections)

    assert "Abstract" in titles
    assert "Introduction" in titles and "Method" in titles

    abstract = _find(sections, "Abstract")
    assert "trades accuracy for compute" in abstract.text
    # the preamble is package configuration, not prose
    assert "usepackage" not in abstract.text
    assert "amsmath" not in abstract.text
    assert "documentclass" not in abstract.text

    assert _find(sections, "Introduction").level == 1
    assert _find(sections, "Scope").level == 2

    figure = _find(sections, "Figure 1")
    assert figure.level == 3
    assert figure.text == "Loss versus sequence length for three sparsity levels."
    table = _find(sections, "Table 1")
    assert table.text == "Validation perplexity by sparsity level."

    assert [s.order for s in sections] == list(range(len(sections)))


def test_parse_latex_files_prose_under_the_heading_it_actually_follows() -> None:
    """Each section's body must be exactly its own -- no bleed across a boundary.

    A citation is only as good as this: if §Introduction carries text that
    belongs to §Scope, a note citing "§Introduction p.3" quotes the wrong
    section, and nothing downstream can tell. Asserting titles and levels alone
    would not notice, so each body is checked positively *and* negatively.
    """
    sections = parse_latex(MAIN_TEX.encode("utf-8"))
    bodies = {name: _find(sections, name).text for name in ("Introduction", "Scope", "Method")}

    assert "Dense attention is quadratic" in bodies["Introduction"]
    assert bodies["Scope"] == "We restrict attention to decoder only models."
    assert bodies["Method"] == "We train with block sparse masks."

    # no section may contain another's prose, nor a heading that follows it
    assert "decoder only" not in bodies["Introduction"]
    assert "block sparse masks" not in bodies["Introduction"]
    assert "block sparse masks" not in bodies["Scope"]
    assert "Dense attention" not in bodies["Method"]
    for name, body in bodies.items():
        assert "Scope" not in body or name == "Scope"
        assert "\\section" not in body and "\\subsection" not in body

    # and the float caption is not duplicated into the prose around it
    assert "Loss versus sequence length" not in bodies["Introduction"]
    assert "Loss versus sequence length" not in bodies["Scope"]


def test_parse_latex_bundle_expands_one_level_of_input() -> None:
    with_input = MAIN_TEX.replace(r"\end{document}", "\\input{sections/results}\n\\end{document}")
    bundle = _tar_gz({"paper/main.tex": with_input, "paper/sections/results.tex": RESULTS_TEX})
    sections = parse_latex(bundle)

    assert "Results" in _titles(sections)
    assert "matches the dense baseline" in _find(sections, "Results").text


def test_parse_latex_strips_citations_from_prose() -> None:
    sections = parse_latex(MAIN_TEX.encode("utf-8"))
    introduction = _find(sections, "Introduction")

    assert "Dense attention is quadratic" in introduction.text
    assert "cite" not in introduction.text
    assert "vaswani2017" not in introduction.text
    assert "child2019" not in introduction.text


def test_parse_latex_accepts_a_raw_tex_byte_string() -> None:
    sections = parse_latex(MAIN_TEX.encode("utf-8"))
    titles = _titles(sections)
    assert {"Abstract", "Introduction", "Scope", "Method", "Figure 1", "Table 1"} <= set(titles)


def test_parse_latex_drops_comments_and_unresolvable_inputs() -> None:
    noisy = rb"""
\documentclass{article}
\begin{document}
\section{Introduction}
Dense attention is quadratic. % TODO: cite the original paper here
\input{sections/nowhere}
More prose follows the missing input.
\end{document}
"""
    introduction = _find(parse_latex(noisy), "Introduction")

    assert "Dense attention is quadratic." in introduction.text
    assert "More prose follows" in introduction.text
    assert "TODO" not in introduction.text
    assert "sections/nowhere" not in introduction.text


def test_parse_latex_of_empty_input_is_empty() -> None:
    assert parse_latex(b"   ") == []


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def test_parse_html_maps_heading_levels_and_isolates_figcaptions() -> None:
    pytest.importorskip("bs4")
    sections = parse_html(HTML_PAGE)

    title = _find(sections, "Scaling Laws for Sparse Attention")
    assert title.level == 1
    assert "quadratic in sequence length" in title.text

    method_sections = [s for s in sections if s.title == "Method"]
    assert method_sections[0].level == 2
    assert "block sparse transformers" in method_sections[0].text

    caption = _find(sections, "Figure 2")
    assert caption.level == 3
    assert caption.text == "Figure 2: Loss versus sequence length."

    # prose after the caption returns to the enclosing section, at its level
    assert len(method_sections) == 2
    assert "32k tokens" in method_sections[1].text
    assert method_sections[1].level == 2

    body = " ".join(s.text for s in sections)
    assert "tracker" not in body and "color: red" not in body


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


def _reference_sections(text: str) -> list[PaperSection]:
    return [
        PaperSection(title="Introduction", text="Dense attention is quadratic.", order=0),
        PaperSection(title="References", text=text, order=1),
    ]


def test_extract_references_splits_bracketed_entries_and_drops_stubs() -> None:
    body = (
        "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
        "[2] Child et al. Generating long sequences with sparse transformers. arXiv 2019.\n"
        "[3] ibid.\n"
        "[4] Beltagy et al. Longformer: the long document transformer. arXiv 2020.\n"
    )
    entries = extract_references(_reference_sections(body))

    assert len(entries) == 3  # "ibid." is below the minimum entry length
    assert entries[0].startswith("Vaswani et al.")
    assert entries[-1].startswith("Beltagy et al.")
    assert all(not entry.startswith("[") for entry in entries)


def test_extract_references_handles_numbered_lines() -> None:
    body = (
        "1. Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
        "2. Child et al. Generating long sequences with sparse transformers. 2019.\n"
        "3. Beltagy et al. Longformer: the long document transformer. 2020.\n"
    )
    entries = extract_references(_reference_sections(body))

    assert len(entries) == 3
    assert entries[0].startswith("Vaswani")


def test_extract_references_falls_back_to_one_entry_per_line() -> None:
    body = (
        "Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
        "Child et al. Generating long sequences with sparse transformers. 2019.\n"
        "ibid.\n"
    )
    entries = extract_references(_reference_sections(body))

    assert entries == [
        "Vaswani et al. Attention is all you need. NeurIPS 2017.",
        "Child et al. Generating long sequences with sparse transformers. 2019.",
    ]


def test_extract_references_ignores_everything_outside_the_bibliography() -> None:
    sections = [
        PaperSection(
            title="Introduction",
            text="[1] This bracketed line is prose, not a bibliography entry at all.\n"
            "[2] Neither is this one, and both are long enough to survive the filter.\n"
            "[3] Nor this third decoy line, which is also comfortably long enough.",
            order=0,
        )
    ]
    assert extract_references(sections) == []


def test_extract_references_without_a_bibliography_is_empty() -> None:
    assert extract_references([PaperSection(title="Introduction", text="Body text.")]) == []
    assert extract_references([]) == []


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def _document(content: bytes, content_type: str) -> FetchedDocument:
    return FetchedDocument(
        url="https://example.org/paper",
        content=content,
        content_type=content_type,
        path=Path("unused"),
        content_hash="deadbeef",
        from_cache=False,
    )


def test_parse_document_dispatches_on_the_bytes_not_the_declared_type() -> None:
    # a PDF mislabelled as HTML is still parsed as a PDF
    pdf_sections = parse_document(_document(_synthetic_pdf(), "text/html"))
    assert "1 Introduction" in _titles(pdf_sections)

    # a gzipped LaTeX bundle mislabelled as HTML is still parsed as LaTeX
    bundle = _tar_gz({"paper/main.tex": MAIN_TEX})
    tex_sections = parse_document(_document(bundle, "text/html"))
    assert "Abstract" in _titles(tex_sections)


def test_parse_document_falls_back_to_html() -> None:
    pytest.importorskip("bs4")
    sections = parse_document(_document(HTML_PAGE, "text/html"))
    assert "Scaling Laws for Sparse Attention" in _titles(sections)


def test_fetched_document_kind_follows_the_resolved_content_type() -> None:
    assert _document(b"%PDF-1.7", "application/pdf").kind == "pdf"
    assert _document(b"\x1f\x8b", "application/gzip").kind == "tex"
    assert _document(b"\\documentclass{article}", "application/x-tex").kind == "tex"
    assert _document(b"<html></html>", "text/html").kind == "html"
