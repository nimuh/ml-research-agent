"""ContentFetcher against a scripted transport: content addressing, the on-disk
index, content-type sniffing, arXiv preference, offline, retries and the size cap.

Every fetcher here is handed a fake transport, and an autouse fixture turns a
real ``httpx.Client`` into a hard failure -- a unit test that reaches the
network is worse than one that fails.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from lit_helpers import RecordingLogger, make_paper

from ml_research_agent.config import Config
from ml_research_agent.literature import fetch as fetch_module
from ml_research_agent.literature.fetch import (
    ARXIV_PDF_URL,
    ARXIV_SOURCE_URL,
    ContentFetcher,
    FetchedDocument,
    Kind,
)
from ml_research_agent.literature.sources import HttpReply
from ml_research_agent.utils import concurrency
from ml_research_agent.utils.hashing import hash_bytes

ARXIV_ID = "2401.00001"
EPRINT_URL = f"{ARXIV_SOURCE_URL}{ARXIV_ID}"
PDF_URL = f"{ARXIV_PDF_URL}{ARXIV_ID}.pdf"

PDF_BYTES = b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\n"
GZIP_BYTES = b"\x1f\x8b\x08\x00tarball-of-latex-sources"
HTML_BYTES = b"<html><body><h1>Sparse attention</h1></body></html>"

URL = "https://example.org/document"
OCTET = "application/octet-stream"
PDF_TYPE = "application/pdf"
GZIP_TYPE = "application/gzip"
TEX_TYPE = "application/x-tex"
HTML_TYPE = "text/html"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def reply(
    body: bytes = b"", *, status: int = 200, content_type: str | None = None, url: str = ""
) -> HttpReply:
    headers = {"content-type": content_type} if content_type is not None else {}
    return HttpReply(status_code=status, content=body, headers=headers, url=url)


class ScriptedTransport:
    """A per-URL scripted transport that counts what it was asked for.

    An unscripted url answers 404, so "the fetcher asked for something nobody
    recorded" surfaces as a missing document rather than as a live request.
    """

    def __init__(self, routes: Mapping[str, HttpReply] | None = None) -> None:
        self.routes = dict(routes or {})
        self.requests: list[tuple[str, str]] = []

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpReply:
        self.requests.append((method.upper(), url))
        return self.routes.get(url, reply(status=404, url=url))

    @property
    def urls(self) -> list[str]:
        return [url for _method, url in self.requests]

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _forbid_a_real_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing a real client is a test failure, not a slow test."""
    import httpx

    def _forbidden(*_args: object, **_kw: object) -> object:
        raise AssertionError("a real httpx.Client was constructed by ContentFetcher")

    monkeypatch.setattr(httpx, "Client", _forbidden)


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries are instant here: the backoff schedule is not what is under test."""
    monkeypatch.setattr(concurrency, "backoff_delays", lambda attempts, **_kw: [0.0] * attempts)


@pytest.fixture
def fetch_config(config: Config) -> Config:
    """The tmp_path-scoped Config with the polite rate limiter turned off."""
    return config.with_overrides(**{"literature.rate_limit_per_second": 1000.0})


def make_fetcher(
    config: Config, transport: ScriptedTransport, logger: RecordingLogger | None = None
) -> ContentFetcher:
    return ContentFetcher(config, http=transport, logger=logger)


# ---------------------------------------------------------------------------
# content addressing
# ---------------------------------------------------------------------------


def test_a_fetch_writes_a_blob_named_after_the_hash_of_its_bytes(fetch_config: Config) -> None:
    transport = ScriptedTransport({"https://example.org/paper": reply(HTML_BYTES)})

    document = make_fetcher(fetch_config, transport).fetch("https://example.org/paper")

    assert document is not None
    assert document.content == HTML_BYTES
    assert document.content_hash == hash_bytes(HTML_BYTES)
    assert document.from_cache is False
    assert document.path.exists()
    assert document.path.read_bytes() == HTML_BYTES
    assert fetch_config.paths.kb_raw in document.path.parents
    assert document.content_hash in document.path.name


def test_two_urls_serving_identical_bytes_share_one_blob(fetch_config: Config) -> None:
    """Addressing is by content, not by url: the same PDF twice is one file."""
    transport = ScriptedTransport(
        {
            "https://mirror-a.example/paper.pdf": reply(PDF_BYTES),
            "https://mirror-b.example/paper.pdf": reply(PDF_BYTES),
        }
    )
    fetcher = make_fetcher(fetch_config, transport)

    first = fetcher.fetch("https://mirror-a.example/paper.pdf")
    second = fetcher.fetch("https://mirror-b.example/paper.pdf")

    assert first is not None and second is not None
    assert first.content_hash == second.content_hash
    assert first.path == second.path
    assert transport.request_count == 2  # two urls, but one blob on disk
    assert len(list(fetch_config.paths.kb_raw.rglob("*.pdf"))) == 1


# ---------------------------------------------------------------------------
# idempotence
# ---------------------------------------------------------------------------


def test_refetching_the_same_url_costs_no_request(fetch_config: Config) -> None:
    transport = ScriptedTransport({"https://example.org/paper.pdf": reply(PDF_BYTES)})
    fetcher = make_fetcher(fetch_config, transport)

    first = fetcher.fetch("https://example.org/paper.pdf")
    second = fetcher.fetch("https://example.org/paper.pdf")

    assert first is not None and second is not None
    assert transport.request_count == 1
    assert second.from_cache is True
    assert second.content_hash == first.content_hash
    assert second.path == first.path
    assert second.content == first.content


def test_a_fresh_fetcher_over_the_same_config_still_hits_the_cache(fetch_config: Config) -> None:
    """The url index lives on disk, so a new process inherits the cache."""
    warm = ScriptedTransport({"https://example.org/paper.pdf": reply(PDF_BYTES)})
    first = make_fetcher(fetch_config, warm).fetch("https://example.org/paper.pdf")

    cold = ScriptedTransport()
    second = make_fetcher(fetch_config, cold).fetch("https://example.org/paper.pdf")

    assert first is not None and second is not None
    assert cold.request_count == 0
    assert second.from_cache is True
    assert (second.content_hash, second.path) == (first.content_hash, first.path)


# ---------------------------------------------------------------------------
# content-type resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "declared", "kind", "url", "expected_type", "expected_kind"),
    [
        pytest.param(PDF_BYTES, HTML_TYPE, "auto", URL, PDF_TYPE, "pdf", id="pdf-magic-wins"),
        pytest.param(GZIP_BYTES, OCTET, "tex", EPRINT_URL, GZIP_TYPE, "tex", id="gzip-magic-wins"),
        pytest.param(b"unknown", OCTET, "pdf", URL, PDF_TYPE, "pdf", id="kind-pdf"),
        pytest.param(b"unknown", OCTET, "auto", f"{URL}.pdf", PDF_TYPE, "pdf", id="url-suffix-pdf"),
        pytest.param(b"unknown", OCTET, "tex", URL, TEX_TYPE, "tex", id="kind-tex"),
        pytest.param(b"unknown", "", "auto", URL, HTML_TYPE, "html", id="fallback-html"),
        pytest.param(HTML_BYTES, HTML_TYPE, "auto", URL, HTML_TYPE, "html", id="declared-html"),
    ],
)
def test_the_bytes_outrank_the_declared_content_type(
    fetch_config: Config,
    body: bytes,
    declared: str,
    kind: Kind,
    url: str,
    expected_type: str,
    expected_kind: str,
) -> None:
    transport = ScriptedTransport({url: reply(body, content_type=declared)})

    document = make_fetcher(fetch_config, transport).fetch(url, kind=kind)

    assert document is not None
    assert document.content_type == expected_type
    assert document.kind == expected_kind


# ---------------------------------------------------------------------------
# fetch_paper: candidate order and degradation
# ---------------------------------------------------------------------------


def test_fetch_paper_asks_for_the_arxiv_latex_source_first(fetch_config: Config) -> None:
    """LaTeX keeps the section structure; a PDF makes us re-infer it."""
    transport = ScriptedTransport(
        {EPRINT_URL: reply(GZIP_BYTES, content_type="application/x-eprint")}
    )
    paper = make_paper(title="Sparse attention", arxiv_id=ARXIV_ID, pdf_url=PDF_URL)

    document = make_fetcher(fetch_config, transport).fetch_paper(paper)

    assert document is not None
    assert transport.urls == [EPRINT_URL]
    assert document.url == EPRINT_URL
    assert document.kind == "tex"


def test_fetch_paper_falls_back_to_the_pdf_when_the_source_is_missing(fetch_config: Config) -> None:
    transport = ScriptedTransport({PDF_URL: reply(PDF_BYTES, content_type="application/pdf")})
    paper = make_paper(title="Sparse attention", arxiv_id=ARXIV_ID)

    document = make_fetcher(fetch_config, transport).fetch_paper(paper)

    assert document is not None
    assert transport.urls == [EPRINT_URL, PDF_URL]  # e-print tried first, 404, then the pdf
    assert document.url == PDF_URL
    assert document.kind == "pdf"


def test_fetch_paper_returns_none_and_logs_when_every_candidate_fails(
    fetch_config: Config,
) -> None:
    transport = ScriptedTransport()
    logger = RecordingLogger()
    paper = make_paper(
        title="Sparse attention",
        arxiv_id=ARXIV_ID,
        pdf_url="https://example.org/paper.pdf",
        url="https://example.org/abs",
    )

    document = make_fetcher(fetch_config, transport, logger).fetch_paper(paper)

    assert document is None
    events = logger.events("literature.no_full_text")
    assert events, f"expected a literature.no_full_text event, saw {logger.names}"
    assert events[0]["paper"] == paper.key
    assert set(transport.urls) == {
        EPRINT_URL,
        PDF_URL,
        "https://example.org/paper.pdf",
        "https://example.org/abs",
    }


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------


def test_attach_returns_a_copy_and_leaves_the_caller_s_paper_alone(fetch_config: Config) -> None:
    transport = ScriptedTransport({"https://example.org/paper.pdf": reply(PDF_BYTES)})
    fetcher = make_fetcher(fetch_config, transport)
    paper = make_paper(title="Sparse attention", arxiv_id=ARXIV_ID)
    document = fetcher.fetch("https://example.org/paper.pdf")
    assert document is not None

    attached = fetcher.attach(paper, document)

    assert attached is not paper
    assert attached.raw_path == str(document.path)
    assert attached.full_text_hash == document.content_hash
    assert paper.raw_path is None
    assert paper.full_text_hash is None
    assert attached.key == paper.key


# ---------------------------------------------------------------------------
# offline
# ---------------------------------------------------------------------------


def test_offline_with_a_cold_cache_returns_nothing_and_touches_no_transport(
    fetch_config: Config,
) -> None:
    offline = fetch_config.with_overrides(offline=True)
    transport = ScriptedTransport({"https://example.org/paper.pdf": reply(PDF_BYTES)})
    logger = RecordingLogger()

    document = make_fetcher(offline, transport, logger).fetch("https://example.org/paper.pdf")

    assert document is None
    assert transport.request_count == 0
    assert logger.events("literature.offline_fetch_skipped")


def test_offline_still_serves_a_warm_cache(fetch_config: Config) -> None:
    warm = ScriptedTransport({"https://example.org/paper.pdf": reply(PDF_BYTES)})
    online = make_fetcher(fetch_config, warm).fetch("https://example.org/paper.pdf")

    offline = fetch_config.with_overrides(offline=True)
    cold = ScriptedTransport()
    document = make_fetcher(offline, cold).fetch("https://example.org/paper.pdf")

    assert online is not None and document is not None
    assert cold.request_count == 0
    assert document.from_cache is True
    assert document.content_hash == online.content_hash


# ---------------------------------------------------------------------------
# HTTP failure
# ---------------------------------------------------------------------------


def test_a_retryable_status_is_retried_and_then_degrades_to_none(fetch_config: Config) -> None:
    transport = ScriptedTransport({"https://example.org/paper.pdf": reply(status=500)})
    logger = RecordingLogger()

    document = make_fetcher(fetch_config, transport, logger).fetch("https://example.org/paper.pdf")

    assert document is None  # a dead PDF must not end a survey
    assert transport.request_count == fetch_module._RETRY_ATTEMPTS
    assert len(logger.events("literature.fetch_retry")) == fetch_module._RETRY_ATTEMPTS - 1
    assert logger.events("literature.fetch_failed")


def test_a_404_is_not_worth_three_round_trips(fetch_config: Config) -> None:
    transport = ScriptedTransport()
    logger = RecordingLogger()

    document = make_fetcher(fetch_config, transport, logger).fetch("https://example.org/gone.pdf")

    assert document is None
    assert transport.request_count == 1
    assert logger.events("literature.fetch_failed")
    assert not logger.events("literature.fetch_retry")


def test_nothing_is_written_to_disk_when_a_fetch_fails(fetch_config: Config) -> None:
    transport = ScriptedTransport()

    assert make_fetcher(fetch_config, transport).fetch("https://example.org/gone.pdf") is None
    assert list(fetch_config.paths.kb_raw.rglob("*")) == []


# ---------------------------------------------------------------------------
# the size cap
# ---------------------------------------------------------------------------


def test_an_oversized_body_is_refused_and_logged(
    fetch_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200MB "paper" is a mirror of a dataset. The cap is shrunk, not the body."""
    monkeypatch.setattr(fetch_module, "MAX_DOCUMENT_BYTES", 8)
    body = b"x" * 64
    transport = ScriptedTransport({"https://example.org/huge.pdf": reply(body)})
    logger = RecordingLogger()

    document = make_fetcher(fetch_config, transport, logger).fetch("https://example.org/huge.pdf")

    assert document is None
    events = logger.events("literature.document_too_large")
    assert events, f"expected a literature.document_too_large event, saw {logger.names}"
    assert events[0]["bytes"] == len(body)
    assert events[0]["cap"] == 8
    assert list(fetch_config.paths.kb_raw.rglob("*")) == []


# ---------------------------------------------------------------------------
# the document itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content_type", "expected"),
    [
        ("application/pdf", "pdf"),
        ("application/gzip", "tex"),
        ("application/x-tex", "tex"),
        ("text/html", "html"),
        ("application/octet-stream", "html"),
    ],
)
def test_kind_is_derived_from_the_resolved_content_type(content_type: str, expected: str) -> None:
    document = FetchedDocument(
        url="https://example.org/x",
        content=b"",
        content_type=content_type,
        path=Path("/dev/null"),
        content_hash="deadbeef",
        from_cache=False,
    )

    assert document.kind == expected
