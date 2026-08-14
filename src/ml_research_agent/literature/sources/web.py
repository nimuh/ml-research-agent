"""Generic web search + fetch for blogs, docs and technical reports."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from ...errors import SourceError
from ...types import Author, Paper, Provenance, Venue
from ..dedupe import normalize_arxiv_id
from ..query import SourceQuery
from . import BaseSource

SEARCH_URL = "https://html.duckduckgo.com/html/"

_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(?P<id>\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
_OPENREVIEW_URL = re.compile(r"openreview\.net/forum\?id=(?P<id>[\w-]+)", re.IGNORECASE)
_DOI_URL = re.compile(r"doi\.org/(?P<doi>10\.\d{4,9}/\S+)", re.IGNORECASE)
_MAX_RESULTS = 25


class WebSource(BaseSource):
    """Best-effort fallback.

    Scraping a search page is fragile by nature, so every failure returns empty
    with a log line. Nothing downstream may assume this source produced anything.
    """

    name = "web"

    def search(self, query: SourceQuery) -> list[Paper]:
        try:
            body = self.fetch_text(SEARCH_URL, params={"q": query.text})
        except SourceError as exc:
            self.log("warning", "literature.source_degraded", error=str(exc))
            return []
        if not body:
            return []

        papers: list[Paper] = []
        for url in _result_urls(body)[:_MAX_RESULTS]:
            paper = _stub_from_url(url, source=self.name)
            if paper is not None:
                papers.append(paper)
        if not papers:
            self.log("info", "literature.web_no_identifiable_papers", query=query.text)
        return papers

    def get(self, identifier: str) -> Paper | None:
        """Fetch a page and read its citation meta tags, if it has any."""
        try:
            body = self.fetch_text(identifier)
        except SourceError as exc:
            self.log("warning", "literature.page_unavailable", url=identifier, error=str(exc))
            return None
        if not body:
            return None
        meta = _citation_meta(body)
        title = meta.get("citation_title") or _html_title(body)
        if not title:
            return _stub_from_url(identifier, source=self.name)
        year_raw = (meta.get("citation_publication_date") or meta.get("citation_date") or "")[:4]
        journal = meta.get("citation_journal_title")
        return Paper(
            title=" ".join(title.split()),
            abstract=" ".join((meta.get("description") or "").split()),
            authors=[
                Author(name=name)
                for name in meta.get("citation_authors", "").split(";")
                if name.strip()
            ],
            year=int(year_raw) if year_raw.isdigit() else None,
            venue=Venue(name=journal, kind="journal") if journal else None,
            arxiv_id=normalize_arxiv_id(meta.get("citation_arxiv_id")),
            doi=meta.get("citation_doi"),
            url=identifier,
            pdf_url=meta.get("citation_pdf_url"),
            sources=[self.name],
            provenance=[Provenance(source=identifier, locator="citation meta tags")],
        )


def _soup(body: str) -> Any:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SourceError("web source needs beautifulsoup4", retryable=False) from exc
    return BeautifulSoup(body, "html.parser")


def _result_urls(body: str) -> list[str]:
    urls: list[str] = []
    for anchor in _soup(body).find_all("a", href=True):
        href = str(anchor["href"])
        if href.startswith("//duckduckgo.com/l/") or "uddg=" in href:
            target = parse_qs(urlparse(href).query).get("uddg")
            href = target[0] if target else ""
        if href.startswith("http") and href not in urls:
            urls.append(href)
    return urls


def _citation_meta(body: str) -> dict[str, str]:
    """Highwire ``citation_*`` tags: how publishers self-describe a paper."""
    meta: dict[str, str] = {}
    authors: list[str] = []
    for tag in _soup(body).find_all("meta"):
        name = str(tag.get("name") or tag.get("property") or "").lower()
        content = str(tag.get("content") or "").strip()
        if not name or not content:
            continue
        if name == "citation_author":
            authors.append(content)
        elif name not in meta:
            meta[name] = content
    if authors:
        meta["citation_authors"] = "; ".join(authors)
    return meta


def _html_title(body: str) -> str | None:
    node = _soup(body).find("title")
    return node.get_text(strip=True) if node else None


def _stub_from_url(url: str, *, source: str) -> Paper | None:
    """Only URLs carrying a real paper id become candidates."""
    arxiv = _ARXIV_URL.search(url)
    if arxiv:
        arxiv_id = normalize_arxiv_id(arxiv.group("id"))
        return Paper(
            title=f"arXiv:{arxiv_id}",
            arxiv_id=arxiv_id,
            url=url,
            sources=[source],
            tags=["web-stub"],
            provenance=[Provenance(source=url, locator="web search result")],
        )
    openreview = _OPENREVIEW_URL.search(url)
    if openreview:
        return Paper(
            title=f"openreview:{openreview.group('id')}",
            openreview_id=openreview.group("id"),
            url=url,
            sources=[source],
            tags=["web-stub"],
            provenance=[Provenance(source=url, locator="web search result")],
        )
    doi = _DOI_URL.search(url)
    if doi:
        return Paper(
            title=f"doi:{doi.group('doi')}",
            doi=doi.group("doi"),
            url=url,
            sources=[source],
            tags=["web-stub"],
            provenance=[Provenance(source=url, locator="web search result")],
        )
    return None


__all__ = ["SEARCH_URL", "WebSource"]
