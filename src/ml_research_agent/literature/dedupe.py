"""Cross-source dedup: DOI/arXiv id, normalized title, author-year fuzzy match;
merges records and keeps the richest metadata plus all source links."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher

from ..types import Paper, Venue
from ._text import strip_accents

_PUNCT_RE = re.compile(r"[^a-z0-9]+")
_ARXIV_VERSION_RE = re.compile(r"v\d+$")
_ARXIV_PREFIX_RE = re.compile(r"^(arxiv[:/]|abs/)", re.IGNORECASE)

# Titles shorter than this are too generic for a fuzzy comparison to mean
# anything ("Attention" vs "Attention?"), so they only match exactly.
_MIN_FUZZY_TITLE_CHARS = 12


def paper_key(paper: Paper) -> str:
    """The dedup key. Delegates to :attr:`Paper.key` so citation and dedup agree."""
    return paper.key


def normalize_title(title: str) -> str:
    """Fold case, accents and punctuation so cross-source titles compare equal."""
    folded = _PUNCT_RE.sub(" ", strip_accents(title).lower())
    return " ".join(folded.split())


def normalize_arxiv_id(value: str | None) -> str | None:
    """``arXiv:2401.00001v3`` -> ``2401.00001``: versions are not distinct papers."""
    if not value:
        return None
    cleaned = _ARXIV_PREFIX_RE.sub("", value.strip()).strip("/")
    return _ARXIV_VERSION_RE.sub("", cleaned).lower() or None


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
    return cleaned or None


def merge_papers(a: Paper, b: Paper) -> Paper:
    """Merge two records of the same paper into the richer one.

    ``a`` is the incumbent: it keeps its id so anything already referring to it
    stays valid. Every source link, id and code url from both survives -- losing
    a source link would make the merged record less traceable than its parts.
    """
    updates = {
        "title": _richer_text(a.title, b.title),
        "abstract": _richer_text(a.abstract, b.abstract),
        "authors": a.authors if len(a.authors) >= len(b.authors) else b.authors,
        # First appearance wins: a preprint year is the honest date for recency.
        "year": min(y for y in (a.year, b.year) if y is not None) if (a.year or b.year) else None,
        "venue": _richer_venue(a, b),
        "arxiv_id": normalize_arxiv_id(a.arxiv_id) or normalize_arxiv_id(b.arxiv_id),
        "doi": normalize_doi(a.doi) or normalize_doi(b.doi),
        "s2_id": a.s2_id or b.s2_id,
        "openreview_id": a.openreview_id or b.openreview_id,
        "external_ids": {**b.external_ids, **a.external_ids},
        "url": a.url or b.url,
        "pdf_url": a.pdf_url or b.pdf_url,
        "code_urls": _union(a.code_urls, b.code_urls),
        "citation_count": max(a.citation_count, b.citation_count),
        "reference_count": max(a.reference_count, b.reference_count),
        "influential_citation_count": max(
            a.influential_citation_count, b.influential_citation_count
        ),
        "tldr": a.tldr or b.tldr,
        "sources": _union(a.sources, b.sources),
        "references": _union(a.references, b.references),
        "citations": _union(a.citations, b.citations),
        "tags": _union(a.tags, b.tags),
        "provenance": [*a.provenance, *b.provenance],
        "sections": a.sections if len(a.sections) >= len(b.sections) else b.sections,
        "full_text_hash": a.full_text_hash or b.full_text_hash,
        "raw_path": a.raw_path or b.raw_path,
    }
    return a.model_copy(update=updates)


def dedupe(papers: Iterable[Paper], *, fuzzy_threshold: float = 0.92) -> list[Paper]:
    """Collapse duplicates across sources, preserving first-seen order."""
    matcher = _Matcher(fuzzy_threshold=fuzzy_threshold)
    merged: list[Paper] = []
    for paper in papers:
        index = matcher.match(paper)
        if index is None:
            matcher.register(paper, len(merged))
            merged.append(paper)
        else:
            merged[index] = merge_papers(merged[index], paper)
            # Re-register: the merged record may now carry ids the incumbent lacked.
            matcher.register(merged[index], index)
    return merged


def find_duplicates(
    papers: Sequence[Paper], *, fuzzy_threshold: float = 0.92
) -> list[tuple[int, int]]:
    """Duplicate index pairs ``(representative, duplicate)``, without merging."""
    matcher = _Matcher(fuzzy_threshold=fuzzy_threshold)
    pairs: list[tuple[int, int]] = []
    representative: dict[int, int] = {}
    for position, paper in enumerate(papers):
        index = matcher.match(paper)
        if index is None:
            matcher.register(paper, position)
            representative[position] = position
        else:
            root = representative.get(index, index)
            pairs.append((root, position))
            representative[position] = root
            matcher.register(paper, root)
    return pairs


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


class _Matcher:
    """The dedup cascade: strong id -> normalized title -> author-year fuzzy.

    Each stage is cheaper and stricter than the next, so the expensive string
    comparison only ever runs against a handful of same-author, same-era papers.
    """

    def __init__(self, *, fuzzy_threshold: float) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.by_id: dict[str, int] = {}
        self.by_title: dict[str, int] = {}
        self.by_author: dict[str, list[int]] = {}
        self.titles: dict[int, str] = {}
        self.years: dict[int, int | None] = {}
        self.ids: dict[int, dict[str, str]] = {}

    def match(self, paper: Paper) -> int | None:
        for token in _id_tokens(paper):
            hit = self.by_id.get(token)
            if hit is not None:
                return hit

        ids = _id_map(paper)
        title = normalize_title(paper.title)
        if title:
            hit = self.by_title.get(title)
            if hit is not None and not self._ids_conflict(hit, ids):
                return hit

        if len(title) < _MIN_FUZZY_TITLE_CHARS:
            return None
        for bucket in _author_buckets(paper):
            for index in self.by_author.get(bucket, []):
                if not _years_compatible(paper.year, self.years.get(index)):
                    continue
                if self._ids_conflict(index, ids):
                    continue
                other = self.titles.get(index, "")
                if len(other) < _MIN_FUZZY_TITLE_CHARS:
                    continue
                if SequenceMatcher(None, title, other).ratio() >= self.fuzzy_threshold:
                    return index
        return None

    def _ids_conflict(self, index: int, ids: dict[str, str]) -> bool:
        """Same namespace, different value: two papers, whatever the title says.

        Preprint-and-published pairs disagree *across* namespaces (one has the
        arXiv id, the other the DOI), so only a within-namespace clash blocks a
        title match.
        """
        known = self.ids.get(index, {})
        return any(
            namespace in known and known[namespace] != value for namespace, value in ids.items()
        )

    def register(self, paper: Paper, index: int) -> None:
        for token in _id_tokens(paper):
            self.by_id.setdefault(token, index)
        self.ids.setdefault(index, {}).update(_id_map(paper))
        title = normalize_title(paper.title)
        if title:
            self.by_title.setdefault(title, index)
            self.titles[index] = title
        self.years[index] = paper.year if paper.year is not None else self.years.get(index)
        for bucket in _author_buckets(paper):
            slot = self.by_author.setdefault(bucket, [])
            if index not in slot:
                slot.append(index)


def _id_tokens(paper: Paper) -> list[str]:
    tokens: list[str] = []
    arxiv = normalize_arxiv_id(paper.arxiv_id)
    if arxiv:
        tokens.append(f"arxiv:{arxiv}")
    doi = normalize_doi(paper.doi)
    if doi:
        tokens.append(f"doi:{doi}")
    if paper.s2_id:
        tokens.append(f"s2:{paper.s2_id.strip().lower()}")
    if paper.openreview_id:
        tokens.append(f"openreview:{paper.openreview_id.strip().lower()}")
    for name, value in paper.external_ids.items():
        if name.lower() in {"arxiv", "doi", "corpusid", "openreview"} and value:
            tokens.append(f"{name.lower()}:{str(value).strip().lower()}")
    return tokens


def _id_map(paper: Paper) -> dict[str, str]:
    """Strong ids by namespace, for detecting a clash rather than a match."""
    ids: dict[str, str] = {}
    for token in _id_tokens(paper):
        namespace, _, value = token.partition(":")
        ids.setdefault(namespace, value)
    return ids


def _author_buckets(paper: Paper) -> list[str]:
    """Last name of the first (and last) author -- author order varies by source."""
    buckets = []
    for author in (paper.authors[:1] + paper.authors[-1:]) if paper.authors else []:
        parts = normalize_title(author.name).split()
        if parts:
            buckets.append(parts[-1])
    return list(dict.fromkeys(buckets))


def _years_compatible(left: int | None, right: int | None) -> bool:
    # Preprint and proceedings versions are routinely a year apart.
    if left is None or right is None:
        return True
    return abs(left - right) <= 1


def _richer_text(a: str, b: str) -> str:
    if not a.strip():
        return b
    if not b.strip():
        return a
    return a if len(a) >= len(b) else b


def _richer_venue(a: Paper, b: Paper) -> Venue | None:
    weak = {"preprint", "unknown"}
    if a.venue and (a.venue.kind not in weak or not b.venue):
        return a.venue
    if b.venue and b.venue.kind not in weak:
        return b.venue
    return a.venue or b.venue


def _union(a: Sequence[str], b: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*a, *b]))


__all__ = [
    "dedupe",
    "find_duplicates",
    "merge_papers",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_title",
    "paper_key",
]
