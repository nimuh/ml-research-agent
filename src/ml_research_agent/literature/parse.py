"""PDF/LaTeX -> structured text: sections, tables, figure captions, references.
Section-aware chunking matters more than raw text quality downstream."""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..errors import SourceError
from ..types import PaperSection
from .fetch import FetchedDocument

# Section names that are headings whatever the typography says.
KNOWN_SECTIONS: frozenset[str] = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "related work",
        "preliminaries",
        "method",
        "methods",
        "methodology",
        "approach",
        "model",
        "architecture",
        "experiments",
        "experimental setup",
        "setup",
        "results",
        "evaluation",
        "analysis",
        "ablation",
        "ablations",
        "ablation study",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "future work",
        "acknowledgments",
        "acknowledgements",
        "references",
        "bibliography",
        "appendix",
        "supplementary material",
        "broader impact",
        "ethics statement",
        "reproducibility statement",
    }
)

_NUMBERED_HEADING = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Z][^.]{2,80})$")
_APPENDIX_HEADING = re.compile(r"^(appendix\s+[A-Z0-9]+)\b[.:]?\s*(?P<title>.*)$", re.IGNORECASE)
_CAPTION = re.compile(
    r"^(?P<label>(figure|fig\.?|table|algorithm)\s*\d+)\s*[.:]?\s*", re.IGNORECASE
)
_REFERENCE_MARKER = re.compile(r"\[\s*\d{1,3}\s*\]")
_NUMBERED_REFERENCE = re.compile(r"^\s*\d{1,3}[.)]\s+")
_MAX_HEADING_CHARS = 90
_MIN_REFERENCE_CHARS = 24

_LATEX_SECTION = re.compile(
    r"\\(?P<cmd>section|subsection|subsubsection|paragraph)\*?\s*\{(?P<title>[^}]*)\}"
)
_LATEX_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
_LATEX_COMMENT = re.compile(r"(?<!\\)%.*")
_LATEX_CAPTION = re.compile(r"\\caption\s*\{", re.MULTILINE)
_LATEX_ENV = re.compile(r"\\begin\{(figure|table)\*?\}")
_LATEX_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_LATEX_ABSTRACT = re.compile(r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", re.DOTALL)
_LATEX_DOCUMENT = re.compile(r"\\begin\{document\}(?P<body>.*?)(?:\\end\{document\}|\Z)", re.DOTALL)
_LATEX_FLOAT = re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", re.DOTALL)
_LATEX_TITLE = re.compile(r"\\title\s*\{(?P<title>.*?)\}", re.DOTALL)
_LATEX_STRIP = [
    (re.compile(r"\\(?:cite[a-z]*|ref|eqref|label|citep|citet)\s*\{[^}]*\}"), ""),
    (re.compile(r"\\(?:textbf|textit|emph|texttt|mathrm|mathbf)\s*\{([^}]*)\}"), r"\1"),
    (re.compile(r"\\(?:footnote)\s*\{[^}]*\}"), ""),
    (re.compile(r"\\[a-zA-Z]+\*?\s*(\[[^\]]*\])?"), " "),
    (re.compile(r"[{}]"), ""),
    (re.compile(r"[ \t]+"), " "),
]


def parse_pdf(data: bytes, *, max_pages: int = 60) -> list[PaperSection]:
    """Section-aware PDF text extraction.

    Headings are detected from typography (font size, weight) *and* from shape
    (numbering, known names, ALL CAPS): either signal alone misfires on the
    layouts ML papers actually use.
    """
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SourceError("pdf parsing needs pymupdf", retryable=False) from exc

    try:
        document: Any = pymupdf.open(stream=data, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001 - pymupdf raises bare RuntimeError
        raise SourceError(f"unreadable pdf: {exc}", retryable=False) from exc

    with document:
        lines = _pdf_lines(document, max_pages=max_pages)
    if not lines:
        return []

    body_size = _body_font_size(lines)
    builder = _SectionBuilder(default_title="Front Matter")
    for line in lines:
        text = line["text"]
        caption = _CAPTION.match(text)
        if caption:
            # Captions are their own sections, but the surrounding prose keeps
            # its title -- a locator that says "Front Matter p.4" is useless.
            enclosing, enclosing_level = builder.current_title, builder.current_level
            builder.open(caption.group("label").title(), level=3, page=line["page"])
            builder.add(text, page=line["page"])
            builder.open(enclosing, level=enclosing_level, page=line["page"])
            continue
        heading = _heading_of(text, size=line["size"], body_size=body_size, bold=line["bold"])
        if heading is not None:
            title, level = heading
            builder.open(title, level=level, page=line["page"])
            continue
        builder.add(text, page=line["page"])
    return builder.finish()


def parse_latex(data: bytes) -> list[PaperSection]:
    """Parse an arXiv source bundle (``.tar.gz``) or a raw ``.tex`` file.

    LaTeX is the better input by a wide margin: the section structure is stated
    rather than inferred, so locators survive intact.
    """
    source = _latex_source(data)
    if not source.strip():
        return []

    builder = _SectionBuilder(default_title="Front Matter")
    title_match = _LATEX_TITLE.search(source)
    if title_match:
        builder.add(_clean_latex(title_match.group("title")), page=None)

    # The preamble is package configuration, not prose; floats are captured
    # separately as captions, so leaving them inline would duplicate them.
    document = _LATEX_DOCUMENT.search(source)
    body = document.group("body") if document else source
    abstract = _LATEX_ABSTRACT.search(body)
    if abstract:
        builder.open("Abstract", level=1, page=None)
        builder.add(_clean_latex(abstract.group("body")), page=None)

    # Any \input left here had no file to resolve against; dropping the whole
    # command keeps its path out of the prose.
    body = _LATEX_INPUT.sub(" ", _LATEX_FLOAT.sub(" ", _LATEX_ABSTRACT.sub(" ", body)))
    cursor = 0
    for match in _LATEX_SECTION.finditer(body):
        builder.add(_clean_latex(body[cursor : match.start()]), page=None)
        builder.open(
            _clean_latex(match.group("title")) or "Section",
            level=_LATEX_LEVELS.get(match.group("cmd"), 1),
            page=None,
        )
        cursor = match.end()
    builder.add(_clean_latex(body[cursor:]), page=None)

    sections = builder.finish()
    sections.extend(_latex_captions(source, start_order=len(sections)))
    return sections


def parse_html(data: bytes) -> list[PaperSection]:
    """Parse an HTML paper page (ar5iv, ACL Anthology, a lab blog post)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SourceError("html parsing needs beautifulsoup4", retryable=False) from exc

    soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    builder = _SectionBuilder(default_title="Front Matter")
    wanted = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "blockquote", "figcaption"]
    for node in soup.find_all(wanted):
        text = " ".join(node.get_text(" ", strip=True).split())
        if not text:
            continue
        name = node.name
        if name.startswith("h") and name[1:].isdigit():
            builder.open(text[:_MAX_HEADING_CHARS], level=int(name[1]), page=None)
        elif name == "figcaption":
            enclosing, enclosing_level = builder.current_title, builder.current_level
            label = _CAPTION.match(text)
            builder.open(label.group("label").title() if label else "Caption", level=3, page=None)
            builder.add(text, page=None)
            builder.open(enclosing, level=enclosing_level, page=None)
        else:
            builder.add(text, page=None)
    return builder.finish()


def parse_document(doc: FetchedDocument) -> list[PaperSection]:
    """Dispatch on what the bytes actually are, not on what the URL claimed."""
    if doc.content.startswith(b"%PDF") or doc.content_type == "application/pdf":
        return parse_pdf(doc.content)
    if doc.content.startswith(b"\x1f\x8b") or doc.content_type in (
        "application/gzip",
        "application/x-tex",
    ):
        return parse_latex(doc.content)
    return parse_html(doc.content)


def extract_references(sections: Sequence[PaperSection]) -> list[str]:
    """Split the bibliography into individual entries.

    Reference strings are what the snowball falls back on when a source has no
    citation graph, so a rough split beats no split.
    """
    body = "\n".join(
        section.text
        for section in sections
        if section.title.strip().lower().startswith(("references", "bibliography"))
    )
    if not body.strip():
        return []

    if len(_REFERENCE_MARKER.findall(body)) >= 3:
        parts = _REFERENCE_MARKER.split(body)
    elif sum(1 for line in body.splitlines() if _NUMBERED_REFERENCE.match(line)) >= 3:
        parts = _split_on_numbered_lines(body)
    else:
        parts = body.split("\n")

    entries = []
    for part in parts:
        entry = " ".join(part.split())
        if len(entry) >= _MIN_REFERENCE_CHARS:
            entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


class _SectionBuilder:
    """Accumulates lines into sections, one open section at a time."""

    def __init__(self, *, default_title: str) -> None:
        self.default_title = default_title
        self.sections: list[PaperSection] = []
        self._title = default_title
        self._level = 1
        self._lines: list[str] = []
        self._page_start: int | None = None
        self._page_end: int | None = None

    @property
    def current_title(self) -> str:
        return self._title

    @property
    def current_level(self) -> int:
        return self._level

    def open(self, title: str, *, level: int, page: int | None) -> None:
        self.close()
        self._title = " ".join(title.split()) or self.default_title
        self._level = level
        self._page_start = page
        self._page_end = page

    def add(self, text: str, *, page: int | None) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        self._lines.append(cleaned)
        if page is not None:
            self._page_start = page if self._page_start is None else self._page_start
            self._page_end = page

    def close(self) -> None:
        # A heading with no body still records where that section starts, so it
        # is only dropped when it is unnamed or a continuation that got no text.
        if not self._lines and (
            self._title == self.default_title
            or any(section.title == self._title for section in self.sections)
        ):
            self._reset()
            return
        self.sections.append(
            PaperSection(
                title=self._title,
                text="\n".join(self._lines),
                level=self._level,
                page_start=self._page_start,
                page_end=self._page_end,
                order=len(self.sections),
            )
        )
        self._reset()

    def _reset(self) -> None:
        self._lines = []
        self._title = self.default_title
        self._level = 1
        self._page_start = None
        self._page_end = None

    def finish(self) -> list[PaperSection]:
        self.close()
        return self.sections


def _pdf_lines(document: Any, *, max_pages: int) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for page_number, page in enumerate(document, start=1):
        if page_number > max_pages:
            break
        blocks = page.get_text("dict").get("blocks", [])
        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = " ".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                sizes = [
                    float(span.get("size", 0.0)) for span in spans if span.get("text", "").strip()
                ]
                flags = [int(span.get("flags", 0)) for span in spans]
                lines.append(
                    {
                        "text": " ".join(text.split()),
                        "page": page_number,
                        "size": max(sizes) if sizes else 0.0,
                        # bit 4 of the pymupdf span flags is "bold".
                        "bold": any(flag & 2**4 for flag in flags),
                    }
                )
    return lines


def _body_font_size(lines: Sequence[dict[str, Any]]) -> float:
    """Modal font size weighted by characters: the size the prose is set in."""
    counts: Counter[float] = Counter()
    for line in lines:
        counts[round(float(line["size"]), 1)] += len(line["text"])
    return counts.most_common(1)[0][0] if counts else 10.0


def _heading_of(text: str, *, size: float, body_size: float, bold: bool) -> tuple[str, int] | None:
    if not text or len(text) > _MAX_HEADING_CHARS or text.endswith((",", ";")):
        return None
    lowered = text.strip().rstrip(".:").lower()

    numbered = _NUMBERED_HEADING.match(text)
    if numbered:
        return text, min(4, numbered.group("number").count(".") + 1)
    appendix = _APPENDIX_HEADING.match(text)
    if appendix:
        return text, 1
    if lowered in KNOWN_SECTIONS:
        return text, 1
    if text.isupper() and 3 <= len(text) <= 60 and any(c.isalpha() for c in text):
        return text, 1
    # Typography alone only counts when the line is short enough to be a title.
    if len(text) <= 60 and (size >= body_size * 1.15 or (bold and size >= body_size)):
        words = text.split()
        if 1 <= len(words) <= 10 and text[0].isupper():
            return text, 2
    return None


def _latex_source(data: bytes) -> str:
    """arXiv e-print bundles are tarballs, single gzipped files, or raw tex."""
    if data.startswith(b"\x1f\x8b"):
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
                return _main_tex(_tex_members(archive))
        except tarfile.TarError:
            return gzip.decompress(data).decode("utf-8", errors="replace")
    return data.decode("utf-8", errors="replace")


def _tex_members(archive: tarfile.TarFile) -> dict[str, str]:
    files: dict[str, str] = {}
    for member in archive.getmembers():
        if not member.isfile() or not member.name.lower().endswith(".tex"):
            continue
        handle = archive.extractfile(member)
        if handle is None:
            continue
        files[member.name] = handle.read().decode("utf-8", errors="replace")
    return files


def _main_tex(files: dict[str, str]) -> str:
    if not files:
        return ""
    main_name = next(
        (name for name, text in files.items() if "\\documentclass" in text),
        max(files, key=lambda name: len(files[name])),
    )
    source = files[main_name]

    # One level of \input expansion: enough for the usual paper.tex + sections/.
    def _expand(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        for candidate in (target, f"{target}.tex"):
            for name, text in files.items():
                if name == candidate or name.endswith(f"/{candidate}"):
                    return text
        return ""

    return _LATEX_INPUT.sub(_expand, source)


def _clean_latex(text: str) -> str:
    cleaned = _LATEX_COMMENT.sub("", text)
    for pattern, replacement in _LATEX_STRIP:
        cleaned = pattern.sub(replacement, cleaned)
    lines = [" ".join(line.split()) for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line)


def _latex_captions(source: str, *, start_order: int) -> list[PaperSection]:
    """Captions are their own sections: a figure locator must stay citable."""
    counters = {"figure": 0, "table": 0}
    environments = [(m.start(), m.group(1)) for m in _LATEX_ENV.finditer(source)]
    sections: list[PaperSection] = []
    for match in _LATEX_CAPTION.finditer(source):
        body = _balanced_braces(source, match.end() - 1)
        if not body:
            continue
        kind = next(
            (name for position, name in reversed(environments) if position < match.start()),
            "figure",
        )
        counters[kind] += 1
        sections.append(
            PaperSection(
                title=f"{kind.title()} {counters[kind]}",
                text=_clean_latex(body),
                level=3,
                order=start_order + len(sections),
            )
        )
    return sections


def _balanced_braces(text: str, start: int) -> str:
    """Read ``{...}`` starting at ``start``, honouring nesting."""
    if start >= len(text) or text[start] != "{":
        return ""
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    return ""


def _split_on_numbered_lines(body: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        if _NUMBERED_REFERENCE.match(line) and current:
            entries.append(" ".join(current))
            current = []
        current.append(_NUMBERED_REFERENCE.sub("", line))
    if current:
        entries.append(" ".join(current))
    return entries


__all__ = [
    "KNOWN_SECTIONS",
    "extract_references",
    "parse_document",
    "parse_html",
    "parse_latex",
    "parse_pdf",
]
