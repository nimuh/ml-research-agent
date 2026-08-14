"""OpenReview adapter: submissions, reviews and rebuttals (useful signal on
what reviewers considered weak or unresolved)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from statistics import fmean
from typing import Any, Literal

from ...errors import SourceError
from ...types import Author, Paper, Provenance, Venue
from ..query import SourceQuery
from . import BaseSource

BASE_URL = "https://api2.openreview.net"
FORUM_URL = "https://openreview.net/forum?id="
PDF_BASE = "https://openreview.net"

# Reviews are one extra request per paper, so only the head of the result list
# gets them. Reviewer criticism is real signal, but not at any price.
MAX_REVIEW_LOOKUPS = 10
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_RATING_RE = re.compile(r"^(\d+(?:\.\d+)?)")
_DECISIONS = ("oral", "spotlight", "poster", "accept", "reject", "withdrawn", "desk")


class OpenReviewSource(BaseSource):
    """OpenReview API v2.

    Venue strings there carry the decision ("ICLR 2024 poster"), which is the
    cheapest honest quality signal in this layer -- captured as a tag, never as
    a relevance judgment.
    """

    name = "openreview"

    def search(self, query: SourceQuery) -> list[Paper]:
        term = " ".join(query.venues) if query.kind == "venue" and query.venues else query.text
        payload = self.fetch_json(
            f"{BASE_URL}/notes/search",
            params={"term": term, "limit": min(query.limit, 100), "type": "terms"},
        )
        if not isinstance(payload, dict):
            return []
        papers = [
            paper
            for note in (payload.get("notes") or [])
            if isinstance(note, dict) and (paper := self._to_paper(note)) is not None
        ]
        papers = [p for p in papers if self._within_window(p, query)]
        self._attach_reviews(papers[:MAX_REVIEW_LOOKUPS])
        return papers

    def get(self, identifier: str) -> Paper | None:
        payload = self.fetch_json(f"{BASE_URL}/notes", params={"id": identifier})
        if not isinstance(payload, dict):
            return None
        notes = payload.get("notes") or []
        return self._to_paper(notes[0]) if notes and isinstance(notes[0], dict) else None

    # -- internals ---------------------------------------------------------

    def _within_window(self, paper: Paper, query: SourceQuery) -> bool:
        if paper.year is None:
            return True
        if query.year_from is not None and paper.year < query.year_from:
            return False
        return not (query.year_to is not None and paper.year > query.year_to)

    def _attach_reviews(self, papers: list[Paper]) -> None:
        """Fold rating statistics into tags. Failures here degrade, never abort."""
        for paper in papers:
            forum = paper.external_ids.get("openreview_forum")
            if not forum:
                continue
            try:
                payload = self.fetch_json(f"{BASE_URL}/notes", params={"forum": forum})
            except SourceError as exc:
                self.log("warning", "literature.reviews_unavailable", forum=forum, error=str(exc))
                continue
            if not isinstance(payload, dict):
                continue
            ratings = [
                value
                for note in (payload.get("notes") or [])
                if isinstance(note, dict)
                for value in (_rating(note.get("content") or {}),)
                if value is not None
            ]
            if not ratings:
                continue
            paper.tags = [
                *paper.tags,
                f"reviews:{len(ratings)}",
                f"mean_rating:{round(fmean(ratings), 2)}",
            ]

    def _to_paper(self, note: dict[str, Any]) -> Paper | None:
        content = note.get("content") or {}
        title = str(_value(content, "title") or "").strip()
        if not title:
            return None
        note_id = str(note.get("id") or "")
        forum = str(note.get("forum") or note_id)
        venue_name = str(_value(content, "venue") or _value(content, "venueid") or "OpenReview")
        year = _year_of(venue_name, note)
        pdf = _value(content, "pdf")
        pdf_url = f"{PDF_BASE}{pdf}" if isinstance(pdf, str) and pdf.startswith("/") else pdf

        authors = _value(content, "authors") or []
        keywords = _value(content, "keywords") or []

        return Paper(
            title=" ".join(title.split()),
            abstract=" ".join(str(_value(content, "abstract") or "").split()),
            authors=[Author(name=str(name)) for name in authors if str(name).strip()],
            year=year,
            venue=Venue(name=venue_name, year=year, kind=_venue_kind(venue_name)),
            openreview_id=note_id or None,
            external_ids={"openreview": note_id, "openreview_forum": forum},
            url=f"{FORUM_URL}{forum}",
            pdf_url=pdf_url if isinstance(pdf_url, str) else None,
            tags=[*(str(k) for k in keywords if str(k).strip()), *_decision_tags(venue_name)],
            sources=[self.name],
            provenance=[Provenance(source=f"openreview:{note_id}", locator="notes/search")],
        )


def _value(content: dict[str, Any], key: str) -> Any:
    """API v2 wraps every field as ``{"value": ...}``; v1 did not."""
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _rating(content: dict[str, Any]) -> float | None:
    for key in ("rating", "recommendation", "overall_rating"):
        raw = _value(content, key)
        if raw is None:
            continue
        match = _RATING_RE.match(str(raw).strip())
        if match:
            return float(match.group(1))
    return None


def _year_of(venue_name: str, note: dict[str, Any]) -> int | None:
    match = _YEAR_RE.search(venue_name)
    if match:
        return int(match.group(0))
    for key in ("pdate", "cdate", "tcdate"):
        stamp = note.get(key)
        if isinstance(stamp, int | float) and stamp > 0:
            return datetime.fromtimestamp(stamp / 1000, tz=UTC).year
    return None


def _decision_tags(venue_name: str) -> list[str]:
    lowered = venue_name.lower()
    return [f"decision:{word}" for word in _DECISIONS if word in lowered]


def _venue_kind(venue_name: str) -> Literal["conference", "workshop", "preprint"]:
    lowered = venue_name.lower()
    if "workshop" in lowered:
        return "workshop"
    if "submitted" in lowered or "withdrawn" in lowered:
        return "preprint"
    return "conference"


__all__ = ["BASE_URL", "MAX_REVIEW_LOOKUPS", "OpenReviewSource"]
