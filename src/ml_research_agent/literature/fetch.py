"""Content fetching with an on-disk cache: PDFs, LaTeX sources, HTML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..config import Config
from ..errors import SourceError
from ..types import Paper
from ..utils.cache import BlobCache, DiskCache
from ..utils.concurrency import retry_sync
from ._log import emit
from .dedupe import normalize_arxiv_id
from .sources import HttpClient, TokenBucket

Kind = Literal["pdf", "tex", "html", "auto"]

ARXIV_SOURCE_URL = "https://arxiv.org/e-print/"
ARXIV_PDF_URL = "https://arxiv.org/pdf/"

# A 200MB "paper" is a mirror of a dataset, not a paper.
MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
_RETRY_ATTEMPTS = 3
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_SUFFIX_BY_TYPE = {
    "application/pdf": ".pdf",
    "application/gzip": ".tar.gz",
    "application/x-tex": ".tex",
    "text/html": ".html",
}


@dataclass(frozen=True)
class FetchedDocument:
    """One fetched blob, addressed by the hash of its content."""

    url: str
    content: bytes
    content_type: str
    path: Path
    content_hash: str
    from_cache: bool

    @property
    def kind(self) -> Kind:
        if self.content_type == "application/pdf":
            return "pdf"
        if self.content_type in ("application/gzip", "application/x-tex"):
            return "tex"
        return "html"


class ContentFetcher:
    """Content-addressed fetching into ``kb_raw``.

    The digest *is* the identity: fetching the same bytes twice occupies one
    path and ingests once, which is what makes the KB idempotent. Failures
    degrade to ``None`` -- one unavailable PDF must not end a survey.
    """

    def __init__(
        self, config: Config, *, http: Any | None = None, logger: Any | None = None
    ) -> None:
        self.config = config
        self.settings = config.literature
        self.logger = logger
        self.http: Any = (
            http
            if http is not None
            else HttpClient(
                timeout=self.settings.request_timeout_seconds, user_agent=self.settings.user_agent
            )
        )
        self.blobs = BlobCache(config.paths.kb_raw)
        # url -> digest, so a re-fetch is a disk read rather than a request.
        self.index = DiskCache(config.paths.cache / "literature", namespace="fetch")
        self.limiter = TokenBucket(self.settings.rate_limit_per_second)

    def fetch(self, url: str, *, kind: Kind = "auto") -> FetchedDocument | None:
        cached = self._from_cache(url)
        if cached is not None:
            return cached
        if self.config.offline:
            self._log("warning", "literature.offline_fetch_skipped", url=url)
            return None

        try:
            content, content_type = self._download(url)
        except SourceError as exc:
            self._log("warning", "literature.fetch_failed", url=url, error=str(exc))
            return None
        if content is None:
            return None

        resolved = _resolve_content_type(content, content_type, kind=kind, url=url)
        suffix = _SUFFIX_BY_TYPE.get(resolved, "")
        digest, path = self.blobs.put(content, suffix=suffix)
        self.index.set(url, {"digest": digest, "content_type": resolved, "suffix": suffix})
        self._log("info", "literature.fetched", url=url, bytes=len(content), content_type=resolved)
        return FetchedDocument(
            url=url,
            content=content,
            content_type=resolved,
            path=path,
            content_hash=digest,
            from_cache=False,
        )

    def fetch_paper(self, paper: Paper) -> FetchedDocument | None:
        """Prefer arXiv LaTeX source over the PDF.

        LaTeX keeps the section structure the traceability commitment needs;
        a PDF makes us re-infer it from font sizes.
        """
        for url, kind in self._candidate_urls(paper):
            document = self.fetch(url, kind=kind)
            if document is not None:
                return document
        self._log("warning", "literature.no_full_text", paper=paper.key)
        return None

    def attach(self, paper: Paper, document: FetchedDocument) -> Paper:
        """Return a *copy* of ``paper`` pointing at the fetched blob.

        Deviation from the sketched interface: ``fetch_paper`` returns the
        document (as specified), so the paper update is a separate, explicit
        step rather than a hidden mutation of the caller's object.
        """
        return paper.model_copy(
            update={"raw_path": str(document.path), "full_text_hash": document.content_hash}
        )

    def close(self) -> None:
        closer = getattr(self.http, "close", None)
        if callable(closer):
            closer()

    # -- internals ---------------------------------------------------------

    def _candidate_urls(self, paper: Paper) -> list[tuple[str, Kind]]:
        candidates: list[tuple[str, Kind]] = []
        arxiv_id = normalize_arxiv_id(paper.arxiv_id)
        if arxiv_id:
            candidates.append((f"{ARXIV_SOURCE_URL}{arxiv_id}", "tex"))
            candidates.append((f"{ARXIV_PDF_URL}{arxiv_id}.pdf", "pdf"))
        if paper.pdf_url:
            candidates.append((paper.pdf_url, "pdf"))
        if paper.url:
            candidates.append((paper.url, "auto"))
        unique: dict[str, Kind] = {}
        for url, kind in candidates:
            unique.setdefault(url, kind)
        return list(unique.items())

    def _from_cache(self, url: str) -> FetchedDocument | None:
        record = self.index.get(url)
        if not isinstance(record, dict):
            return None
        digest = str(record.get("digest") or "")
        suffix = str(record.get("suffix") or "")
        content = self.blobs.get(digest, suffix=suffix) if digest else None
        if content is None:
            return None
        return FetchedDocument(
            url=url,
            content=content,
            content_type=str(record.get("content_type") or "application/octet-stream"),
            path=self.blobs.path_for(digest, suffix=suffix),
            content_hash=digest,
            from_cache=True,
        )

    def _download(self, url: str) -> tuple[bytes | None, str]:
        def _once() -> tuple[bytes, str]:
            self.limiter.acquire()
            reply = self.http.request("GET", url)
            status = int(getattr(reply, "status_code", 0))
            if not 200 <= status < 300:
                raise SourceError(
                    f"fetch failed: HTTP {status}",
                    retryable=status in _RETRYABLE_STATUS,
                    url=url,
                    status=status,
                )
            headers = getattr(reply, "headers", {}) or {}
            declared = str(headers.get("content-type") or headers.get("Content-Type") or "").split(
                ";"
            )[0]
            return bytes(reply.content), declared.strip()

        content, declared = retry_sync(
            _once,
            attempts=_RETRY_ATTEMPTS,
            retry_on=(SourceError,),
            on_retry=lambda attempt, exc: self._log(
                "warning", "literature.fetch_retry", url=url, attempt=attempt, error=str(exc)
            ),
        )
        if len(content) > MAX_DOCUMENT_BYTES:
            self._log(
                "warning",
                "literature.document_too_large",
                url=url,
                bytes=len(content),
                cap=MAX_DOCUMENT_BYTES,
            )
            return None, declared
        return content, declared

    def _log(self, level: str, event: str, /, **fields: Any) -> None:
        emit(self.logger, level, event, **fields)


def _resolve_content_type(content: bytes, declared: str, *, kind: Kind, url: str) -> str:
    """Trust the bytes over the header: arXiv serves tarballs as octet-stream."""
    if content.startswith(b"%PDF"):
        return "application/pdf"
    if content.startswith(b"\x1f\x8b"):
        return "application/gzip"
    if declared in _SUFFIX_BY_TYPE:
        return declared
    lowered = url.lower()
    if kind == "pdf" or lowered.endswith(".pdf"):
        return "application/pdf"
    if kind == "tex" or lowered.endswith((".tex", ".tar.gz")):
        return "application/x-tex"
    return "text/html"


__all__ = ["MAX_DOCUMENT_BYTES", "ContentFetcher", "FetchedDocument"]
