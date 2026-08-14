"""Papers-with-Code adapter: paper -> repo links, benchmark leaderboards, SOTA."""

from __future__ import annotations

from typing import Any

from ...errors import SourceError
from ...types import Author, Paper, Provenance, Venue
from ..dedupe import normalize_arxiv_id
from ..query import SourceQuery
from . import BaseSource

BASE_URL = "https://paperswithcode.com/api/v1"

# Repo links are one request per paper. The code layer does the real repo work
# later; this is just enough to know a paper is *testable*.
MAX_REPO_LOOKUPS = 20
_MAX_ITEMS_PER_PAGE = 100


class PapersWithCodeSource(BaseSource):
    """Paper -> repository links.

    The public API has a history of being unavailable; every failure here is
    logged and returns empty rather than taking a survey down with it.
    """

    name = "paperswithcode"

    def search(self, query: SourceQuery) -> list[Paper]:
        try:
            payload = self.fetch_json(
                f"{BASE_URL}/papers/",
                params={
                    "q": query.text,
                    "items_per_page": min(query.limit, _MAX_ITEMS_PER_PAGE),
                    "page": 1,
                },
            )
        except SourceError as exc:
            self.log("warning", "literature.source_degraded", error=str(exc))
            return []
        if not isinstance(payload, dict):
            return []

        papers = []
        for row in payload.get("results") or []:
            if not isinstance(row, dict):
                continue
            paper = self._to_paper(row)
            if paper is not None and self._within_window(paper, query):
                papers.append(paper)
        self._attach_repos(papers[:MAX_REPO_LOOKUPS])
        if len(papers) > MAX_REPO_LOOKUPS:
            self.log(
                "info",
                "literature.repo_lookup_capped",
                papers=len(papers),
                looked_up=MAX_REPO_LOOKUPS,
            )
        return papers

    def get(self, identifier: str) -> Paper | None:
        try:
            payload = self.fetch_json(f"{BASE_URL}/papers/{identifier}/")
        except SourceError as exc:
            self.log("warning", "literature.source_degraded", error=str(exc))
            return None
        if not isinstance(payload, dict):
            return None
        paper = self._to_paper(payload)
        if paper is not None:
            self._attach_repos([paper])
        return paper

    # -- internals ---------------------------------------------------------

    def _within_window(self, paper: Paper, query: SourceQuery) -> bool:
        if paper.year is None:
            return True
        if query.year_from is not None and paper.year < query.year_from:
            return False
        return not (query.year_to is not None and paper.year > query.year_to)

    def _attach_repos(self, papers: list[Paper]) -> None:
        for paper in papers:
            slug = paper.external_ids.get("paperswithcode")
            if not slug:
                continue
            try:
                payload = self.fetch_json(f"{BASE_URL}/papers/{slug}/repositories/")
            except SourceError as exc:
                self.log("warning", "literature.repos_unavailable", slug=slug, error=str(exc))
                continue
            if not isinstance(payload, dict):
                continue
            urls = [
                str(row["url"])
                for row in (payload.get("results") or [])
                if isinstance(row, dict) and row.get("url")
            ]
            official = [
                str(row["url"])
                for row in (payload.get("results") or [])
                if isinstance(row, dict) and row.get("url") and row.get("is_official")
            ]
            # Official implementations first: fidelity beats popularity.
            ordered = list(dict.fromkeys([*official, *urls]))
            if ordered:
                paper.code_urls = list(dict.fromkeys([*paper.code_urls, *ordered]))

    def _to_paper(self, data: dict[str, Any]) -> Paper | None:
        title = str(data.get("title") or "").strip()
        if not title:
            return None
        published = str(data.get("published") or "")
        year = int(published[:4]) if published[:4].isdigit() else None
        proceeding = data.get("proceeding") or data.get("conference")
        slug = str(data.get("id") or "")
        arxiv_id = normalize_arxiv_id(data.get("arxiv_id"))

        external_ids = {"paperswithcode": slug} if slug else {}
        if arxiv_id:
            external_ids["arxiv"] = arxiv_id

        return Paper(
            title=" ".join(title.split()),
            abstract=" ".join(str(data.get("abstract") or "").split()),
            authors=[Author(name=str(a)) for a in (data.get("authors") or []) if str(a).strip()],
            year=year,
            venue=(
                Venue(name=str(proceeding), year=year, kind="conference")
                if proceeding
                else Venue(name="arXiv", year=year, kind="preprint")
            ),
            arxiv_id=arxiv_id,
            doi=data.get("doi"),
            external_ids=external_ids,
            url=data.get("url_abs"),
            pdf_url=data.get("url_pdf"),
            sources=[self.name],
            provenance=[
                Provenance(source=f"paperswithcode:{slug or title[:40]}", locator="api/v1")
            ],
        )


__all__ = ["BASE_URL", "MAX_REPO_LOOKUPS", "PapersWithCodeSource"]
