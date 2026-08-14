"""GitHub adapter: repo search, metadata, README/file fetch, license, activity."""

from __future__ import annotations

import os
import re
from typing import Any

from ...config import Config
from ...errors import SourceError
from ...types import Paper, Provenance
from ..dedupe import normalize_arxiv_id
from ..query import SourceQuery
from . import BaseSource

BASE_URL = "https://api.github.com"
TOKEN_ENV = "GITHUB_TOKEN"
_MAX_PER_PAGE = 100

# A repo becomes a Paper only when it names one. Everything else is a repo the
# code/ layer may want later, not a literature candidate.
_ARXIV_REF = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arxiv[:\s]\s*)(?P<id>\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})",
    re.IGNORECASE,
)


class GitHubSource(BaseSource):
    """Repo search, used mainly to attach code to papers other sources found.

    Honours ``GITHUB_TOKEN`` when present: unauthenticated search is limited to
    10 requests/minute, which a survey will hit.
    """

    name = "github"

    def __init__(self, config: Config, **kw: Any) -> None:
        super().__init__(config, **kw)
        self.token = os.environ.get(TOKEN_ENV) or None

    def search(self, query: SourceQuery) -> list[Paper]:
        try:
            payload = self.fetch_json(
                f"{BASE_URL}/search/repositories",
                params={
                    "q": self._build_query(query),
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(query.limit, _MAX_PER_PAGE),
                },
                headers=self._headers(),
            )
        except SourceError as exc:
            self.log("warning", "literature.source_degraded", error=str(exc))
            return []
        if not isinstance(payload, dict):
            return []

        papers: list[Paper] = []
        skipped = 0
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            paper = self._to_paper(item)
            if paper is None:
                skipped += 1
            else:
                papers.append(paper)
        if skipped:
            self.log("info", "literature.repos_without_paper", skipped=skipped)
        return papers

    def get(self, identifier: str) -> Paper | None:
        """``identifier`` is ``owner/repo``."""
        data = self.repo_metadata(identifier)
        return self._to_paper(data) if data else None

    def repo_metadata(self, full_name: str) -> dict[str, Any] | None:
        """Raw repo record. Fetching is not running -- nothing here executes code."""
        try:
            payload = self.fetch_json(f"{BASE_URL}/repos/{full_name}", headers=self._headers())
        except SourceError as exc:
            self.log("warning", "literature.repo_unavailable", repo=full_name, error=str(exc))
            return None
        return payload if isinstance(payload, dict) else None

    # -- internals ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _build_query(self, query: SourceQuery) -> str:
        terms = [query.text, *query.keywords[:3]]
        expression = " ".join(t for t in terms if t).strip()
        if query.year_from:
            expression += f" pushed:>={query.year_from}-01-01"
        return expression or "machine learning"

    def _to_paper(self, item: dict[str, Any]) -> Paper | None:
        haystack = " ".join(
            str(item.get(field) or "")
            for field in ("description", "homepage", "html_url", "full_name")
        )
        match = _ARXIV_REF.search(haystack)
        arxiv_id = normalize_arxiv_id(match.group("id")) if match else None
        if not arxiv_id:
            return None

        full_name = str(item.get("full_name") or item.get("name") or "").strip()
        html_url = str(item.get("html_url") or "")
        return Paper(
            # A short stub title on purpose: dedup merges this into the real
            # record from arXiv/S2, and the longer, truer title must win.
            title=full_name or arxiv_id,
            abstract=str(item.get("description") or "").strip(),
            arxiv_id=arxiv_id,
            external_ids={"github": full_name} if full_name else {},
            url=item.get("homepage") or html_url or None,
            code_urls=[html_url] if html_url else [],
            tags=["code-stub", *(str(t) for t in (item.get("topics") or []))],
            sources=[self.name],
            provenance=[Provenance(source=f"github:{full_name}", locator="search/repositories")],
        )


__all__ = ["BASE_URL", "TOKEN_ENV", "GitHubSource"]
