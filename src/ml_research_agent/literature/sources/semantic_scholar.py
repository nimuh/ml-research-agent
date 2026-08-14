"""Semantic Scholar adapter: metadata, abstracts, citations/references, TLDRs,
embeddings-based recommendations."""

from __future__ import annotations

import os
from typing import Any, Literal

from ...config import Config
from ...types import Author, Paper, Provenance, Venue
from ..dedupe import normalize_arxiv_id
from ..query import SourceQuery
from . import BaseSource

BASE_URL = "https://api.semanticscholar.org/graph/v1"
API_KEY_ENV = "S2_API_KEY"

PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,publicationTypes,publicationDate,"
    "citationCount,referenceCount,influentialCitationCount,tldr,authors,openAccessPdf,url"
)
# Graph edges are the expensive call, so ask for less per neighbour.
EDGE_FIELDS = "paperId,externalIds,title,abstract,year,venue,citationCount,authors,url"
MAX_EDGE_LIMIT = 100

_JOURNAL_TYPES = {"JournalArticle", "Review"}
_CONFERENCE_TYPES = {"Conference"}
_PREPRINT_HOSTS = ("arxiv", "biorxiv", "medrxiv", "ssrn", "researchsquare", "preprint")

VenueKind = Literal["conference", "journal", "workshop", "preprint", "unknown"]


class SemanticScholarSource(BaseSource):
    """The snowball workhorse: the only source here with both edge directions.

    Works without a key at a low rate limit; ``S2_API_KEY`` raises that ceiling.
    """

    name = "semantic_scholar"

    def __init__(self, config: Config, **kw: Any) -> None:
        super().__init__(config, **kw)
        self.api_key = os.environ.get(API_KEY_ENV) or None

    def search(self, query: SourceQuery) -> list[Paper]:
        params: dict[str, Any] = {
            "query": query.text,
            "limit": min(query.limit, MAX_EDGE_LIMIT),
            "fields": PAPER_FIELDS,
        }
        if query.kind == "venue" and query.venues:
            params["venue"] = ",".join(query.venues)
        if query.year_from or query.year_to:
            params["year"] = f"{query.year_from or ''}-{query.year_to or ''}"
        payload = self.fetch_json(
            f"{BASE_URL}/paper/search", params=params, headers=self._headers()
        )
        return self._papers_from(payload, key="data")

    def get(self, identifier: str) -> Paper | None:
        payload = self.fetch_json(
            f"{BASE_URL}/paper/{identifier}",
            params={"fields": PAPER_FIELDS},
            headers=self._headers(),
        )
        if not isinstance(payload, dict):
            return None
        return self._to_paper(payload)

    def references(self, paper: Paper) -> list[Paper]:
        """Backward snowball: what this paper cites."""
        return self._edges(paper, "references", "citedPaper")

    def citations(self, paper: Paper) -> list[Paper]:
        """Forward snowball: what cites this paper."""
        return self._edges(paper, "citations", "citingPaper")

    # -- internals ---------------------------------------------------------

    def _edges(self, paper: Paper, endpoint: str, node_key: str) -> list[Paper]:
        identifier = self.identifier_for(paper)
        if identifier is None:
            return []
        payload = self.fetch_json(
            f"{BASE_URL}/paper/{identifier}/{endpoint}",
            params={"fields": EDGE_FIELDS, "limit": MAX_EDGE_LIMIT},
            headers=self._headers(),
        )
        if not isinstance(payload, dict):
            return []
        papers = []
        for row in payload.get("data") or []:
            node = row.get(node_key) if isinstance(row, dict) else None
            if isinstance(node, dict):
                mapped = self._to_paper(node)
                if mapped is not None:
                    papers.append(mapped)
        return papers

    @staticmethod
    def identifier_for(paper: Paper) -> str | None:
        """S2 accepts several id namespaces; use the strongest one we hold."""
        if paper.s2_id:
            return paper.s2_id
        arxiv_id = normalize_arxiv_id(paper.arxiv_id)
        if arxiv_id:
            return f"arXiv:{arxiv_id}"
        if paper.doi:
            return f"DOI:{paper.doi}"
        corpus_id = paper.external_ids.get("CorpusId") or paper.external_ids.get("corpusid")
        return f"CorpusId:{corpus_id}" if corpus_id else None

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def _papers_from(self, payload: Any, *, key: str) -> list[Paper]:
        if not isinstance(payload, dict):
            return []
        rows = payload.get(key) or []
        papers = []
        for row in rows:
            if isinstance(row, dict):
                mapped = self._to_paper(row)
                if mapped is not None:
                    papers.append(mapped)
        return papers

    def _to_paper(self, data: dict[str, Any]) -> Paper | None:
        title = (data.get("title") or "").strip()
        if not title:
            return None
        external = {k: str(v) for k, v in (data.get("externalIds") or {}).items() if v is not None}
        arxiv_id = normalize_arxiv_id(external.get("ArXiv"))
        s2_id = data.get("paperId")
        tldr = data.get("tldr")
        open_access = data.get("openAccessPdf") or {}
        venue_name = (data.get("venue") or "").strip()
        publication_types = data.get("publicationTypes") or []

        return Paper(
            title=title,
            abstract=(data.get("abstract") or "").strip(),
            authors=[
                Author(name=a["name"], author_id=a.get("authorId"))
                for a in (data.get("authors") or [])
                if isinstance(a, dict) and a.get("name")
            ],
            year=data.get("year") if isinstance(data.get("year"), int) else None,
            venue=(
                Venue(
                    name=venue_name,
                    year=data.get("year") if isinstance(data.get("year"), int) else None,
                    kind=_venue_kind(venue_name, publication_types),
                )
                if venue_name
                else None
            ),
            arxiv_id=arxiv_id,
            doi=external.get("DOI"),
            s2_id=s2_id,
            external_ids=external,
            url=data.get("url"),
            pdf_url=open_access.get("url") if isinstance(open_access, dict) else None,
            citation_count=int(data.get("citationCount") or 0),
            reference_count=int(data.get("referenceCount") or 0),
            influential_citation_count=int(data.get("influentialCitationCount") or 0),
            tldr=(tldr or {}).get("text") if isinstance(tldr, dict) else None,
            sources=[self.name],
            provenance=[
                Provenance(
                    source=f"s2:{s2_id}" if s2_id else f"s2:{title[:40]}", locator="graph/v1"
                )
            ],
        )


def _venue_kind(name: str, publication_types: list[Any]) -> VenueKind:
    lowered = name.lower()
    # S2 labels arXiv records as JournalArticle. Taking that at face value would
    # score a preprint like a journal paper in rank.py.
    if any(host in lowered for host in _PREPRINT_HOSTS):
        return "preprint"
    if "workshop" in lowered:
        return "workshop"
    types = {str(t) for t in publication_types}
    if types & _CONFERENCE_TYPES:
        return "conference"
    if types & _JOURNAL_TYPES:
        return "journal"
    return "unknown"


__all__ = ["API_KEY_ENV", "BASE_URL", "PAPER_FIELDS", "SemanticScholarSource"]
