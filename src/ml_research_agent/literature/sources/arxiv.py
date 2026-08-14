"""arXiv adapter: search API + PDF/LaTeX source fetch."""

from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree

from ...errors import SourceError
from ...types import Author, Paper, Provenance, Venue
from ..dedupe import normalize_arxiv_id
from ..query import SourceQuery
from . import BaseSource

API_URL = "http://export.arxiv.org/api/query"
ABS_URL = "https://arxiv.org/abs/"
PDF_URL = "https://arxiv.org/pdf/"
# LaTeX source beats the PDF for section-aware parsing, which is why fetch.py
# asks for this first.
SOURCE_URL = "https://arxiv.org/e-print/"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_ID_IN_URL = re.compile(r"abs/(?P<id>[^/?#]+)")
_UNSAFE = re.compile(r'["\\]')
_MAX_RESULTS = 200


class ArxivSource(BaseSource):
    """arXiv's Atom API.

    Uses the HTTP API directly rather than the ``arxiv`` package: the package
    owns its own transport, and an adapter whose transport cannot be injected
    cannot be tested against a cassette.
    """

    name = "arxiv"

    def search(self, query: SourceQuery) -> list[Paper]:
        params = {
            "search_query": self._build_query(query),
            "start": 0,
            "max_results": min(query.limit, _MAX_RESULTS),
            "sortBy": "relevance" if query.kind != "author" else "submittedDate",
            "sortOrder": "descending",
        }
        body = self.fetch_text(API_URL, params=params)
        if body is None:
            return []
        return self._parse_feed(body)

    def get(self, identifier: str) -> Paper | None:
        arxiv_id = normalize_arxiv_id(identifier)
        if not arxiv_id:
            return None
        body = self.fetch_text(API_URL, params={"id_list": arxiv_id, "max_results": 1})
        if body is None:
            return None
        papers = self._parse_feed(body)
        return papers[0] if papers else None

    # -- internals ---------------------------------------------------------

    def _build_query(self, query: SourceQuery) -> str:
        clauses: list[str] = []
        text = _UNSAFE.sub(" ", query.text).strip()
        if query.kind == "author":
            names = query.authors or ([text] if text else [])
            clauses.extend(f'au:"{name}"' for name in names)
        elif query.kind == "venue":
            # arXiv has no venue field; the journal ref is the closest thing and
            # it is only populated after publication, so this stays a text match.
            names = query.venues or ([text] if text else [])
            clauses.extend(f'all:"{name}"' for name in names)
        elif text:
            clauses.append(f'all:"{text}"')
        for keyword in query.keywords[:4]:
            cleaned = _UNSAFE.sub(" ", keyword).strip()
            if cleaned and cleaned.lower() not in text.lower():
                clauses.append(f'all:"{cleaned}"')
        if not clauses:
            raise SourceError("arxiv: empty query", retryable=False)
        expression = " OR ".join(clauses) if len(clauses) > 1 else clauses[0]
        window = self._date_clause(query)
        return f"({expression}) AND {window}" if window else expression

    def _date_clause(self, query: SourceQuery) -> str:
        if query.year_from is None and query.year_to is None:
            return ""
        start = f"{query.year_from}0101000000" if query.year_from else "190001010000"
        end = f"{query.year_to}1231235959" if query.year_to else "299912312359"
        return f"submittedDate:[{start} TO {end}]"

    def _parse_feed(self, body: str) -> list[Paper]:
        try:
            root = ElementTree.fromstring(body)
        except ElementTree.ParseError as exc:
            raise SourceError("arxiv: malformed Atom feed", retryable=False) from exc
        papers = []
        for entry in root.findall(f"{_ATOM}entry"):
            paper = self._parse_entry(entry)
            if paper is not None:
                papers.append(paper)
        return papers

    def _parse_entry(self, entry: Any) -> Paper | None:
        raw_id = _text(entry.find(f"{_ATOM}id"))
        match = _ID_IN_URL.search(raw_id or "")
        versioned = match.group("id") if match else (raw_id or "")
        arxiv_id = normalize_arxiv_id(versioned)
        title = " ".join((_text(entry.find(f"{_ATOM}title")) or "").split())
        if not arxiv_id or not title:
            return None

        published = _text(entry.find(f"{_ATOM}published")) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        journal_ref = _text(entry.find(f"{_ARXIV}journal_ref"))
        categories = [
            c.attrib["term"] for c in entry.findall(f"{_ATOM}category") if c.attrib.get("term")
        ]

        pdf_url = f"{PDF_URL}{arxiv_id}.pdf"
        for link in entry.findall(f"{_ATOM}link"):
            if link.attrib.get("title") == "pdf" and link.attrib.get("href"):
                pdf_url = link.attrib["href"]

        external_ids = {"arxiv": arxiv_id}
        if versioned and versioned != arxiv_id:
            external_ids["arxiv_version"] = versioned

        return Paper(
            title=title,
            abstract=" ".join((_text(entry.find(f"{_ATOM}summary")) or "").split()),
            authors=[
                Author(name=name)
                for name in (_text(a.find(f"{_ATOM}name")) for a in entry.findall(f"{_ATOM}author"))
                if name
            ],
            year=year,
            venue=(
                Venue(name=journal_ref, year=year, kind="journal")
                if journal_ref
                else Venue(name="arXiv", year=year, kind="preprint")
            ),
            arxiv_id=arxiv_id,
            doi=_text(entry.find(f"{_ARXIV}doi")),
            external_ids=external_ids,
            url=f"{ABS_URL}{arxiv_id}",
            pdf_url=pdf_url,
            tags=categories,
            sources=[self.name],
            provenance=[
                Provenance(source=f"arxiv:{arxiv_id}", locator="atom entry", confidence=1.0)
            ],
        )


def _text(node: Any) -> str | None:
    if node is None or node.text is None:
        return None
    value = node.text.strip()
    return value or None


__all__ = ["API_URL", "PDF_URL", "SOURCE_URL", "ArxivSource"]
