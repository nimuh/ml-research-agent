"""Source adapters. Each implements the `Source` protocol: `search(query) ->
list[Paper]`, `get(id) -> Paper`, with rate limiting, caching and retries."""

from __future__ import annotations

import importlib
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ...config import Config
from ...errors import SourceError
from ...types import Paper
from ...utils.cache import DiskCache
from ...utils.concurrency import retry_sync
from .._log import emit
from ..query import SourceQuery

# Registry entries are (module, class) so a missing optional dependency in one
# adapter cannot break importing the package.
_SOURCE_MODULES: dict[str, tuple[str, str]] = {
    "arxiv": (".arxiv", "ArxivSource"),
    "semantic_scholar": (".semantic_scholar", "SemanticScholarSource"),
    "openreview": (".openreview", "OpenReviewSource"),
    "paperswithcode": (".paperswithcode", "PapersWithCodeSource"),
    "github": (".github_src", "GitHubSource"),
    "web": (".web", "WebSource"),
}

_RETRY_ATTEMPTS = 3
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HttpReply:
    """A minimal response object, so a cassette can stand in for httpx."""

    status_code: int
    content: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)
    url: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text or "null")


@runtime_checkable
class HttpTransport(Protocol):
    """The one method an adapter needs. Tests inject a cassette player here."""

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpReply: ...


class HttpClient:
    """Thin blocking httpx wrapper.

    Deliberately *not* the platform ``tools/http.py``: this layer must not
    depend on another engineer's module landing first, and a source adapter
    needs nothing beyond one request method.
    """

    def __init__(self, *, timeout: float = 30.0, user_agent: str = "ml-research-agent/0.1") -> None:
        self.timeout = timeout
        self.user_agent = user_agent
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                timeout=self.timeout,
                follow_redirects=True,
                headers={"User-Agent": self.user_agent},
            )
        return self._client

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpReply:
        import httpx

        client = self._ensure_client()
        try:
            response = client.request(
                method,
                url,
                params=dict(params) if params else None,
                headers=dict(headers) if headers else None,
                timeout=timeout or self.timeout,
            )
        except httpx.HTTPError as exc:
            raise SourceError(f"http request failed: {exc}", retryable=True, url=url) from exc
        return HttpReply(
            status_code=response.status_code,
            content=response.content,
            headers=dict(response.headers),
            url=str(response.url),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class TokenBucket:
    """Blocking token bucket. Public APIs rate-limit hard; ask politely."""

    def __init__(self, rate_per_second: float, *, burst: int = 1) -> None:
        self.rate_per_second = rate_per_second
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate_per_second <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    float(self.burst), self._tokens + (now - self._updated) * self.rate_per_second
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate_per_second
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Protocol + base
# ---------------------------------------------------------------------------


@runtime_checkable
class Source(Protocol):
    """What every adapter provides. Adding a source is one file, not a refactor."""

    name: str

    def search(self, query: SourceQuery) -> list[Paper]: ...
    def get(self, identifier: str) -> Paper | None: ...
    def references(self, paper: Paper) -> list[Paper]: ...
    def citations(self, paper: Paper) -> list[Paper]: ...


class BaseSource:
    """Shared plumbing: rate limiting, disk cache, retries, offline, logging.

    Snowballing is optional -- ``references``/``citations`` default to empty so a
    metadata-only source degrades the expansion instead of breaking it.
    """

    name: str = "base"

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
        self.cache = DiskCache(
            config.paths.cache / "literature",
            namespace=self.name,
            ttl_seconds=self.settings.cache_ttl_seconds,
        )
        self.limiter = TokenBucket(self.settings.rate_limit_per_second)

    # -- API ---------------------------------------------------------------

    def search(self, query: SourceQuery) -> list[Paper]:
        raise NotImplementedError(f"{self.name} does not implement search")

    def get(self, identifier: str) -> Paper | None:
        return None

    def references(self, paper: Paper) -> list[Paper]:
        return []

    def citations(self, paper: Paper) -> list[Paper]:
        return []

    def close(self) -> None:
        closer = getattr(self.http, "close", None)
        if callable(closer):
            closer()

    # -- plumbing ----------------------------------------------------------

    def log(self, level: str, event: str, /, **fields: Any) -> None:
        emit(self.logger, level, event, source=self.name, **fields)

    def fetch_json(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        cache_key: str | None = None,
    ) -> Any | None:
        """GET and parse JSON. ``None`` means "unavailable", never "empty"."""
        payload = self._fetch(url, params=params, headers=headers, cache_key=cache_key)
        if payload is None:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SourceError(
                f"{self.name}: malformed JSON response", retryable=False, url=url
            ) from exc

    def fetch_text(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        cache_key: str | None = None,
    ) -> str | None:
        return self._fetch(url, params=params, headers=headers, cache_key=cache_key)

    def _fetch(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        cache_key: str | None,
    ) -> str | None:
        key = cache_key or self._cache_key(url, params)
        cached = self.cache.get(key)
        if isinstance(cached, str):
            return cached
        if self.config.offline:
            self.log("warning", "literature.offline_cache_miss", url=url)
            return None

        def _once() -> str:
            self.limiter.acquire()
            reply = self.http.request("GET", url, params=params, headers=headers)
            self._raise_for_status(reply, url)
            return str(reply.text)

        body = retry_sync(
            _once,
            attempts=_RETRY_ATTEMPTS,
            retry_on=(SourceError,),
            on_retry=lambda attempt, exc: self.log(
                "warning", "literature.retry", url=url, attempt=attempt, error=str(exc)
            ),
        )
        self.cache.set(key, body)
        return body

    def _raise_for_status(self, reply: Any, url: str) -> None:
        status = int(getattr(reply, "status_code", 0))
        if 200 <= status < 300:
            return
        raise SourceError(
            f"{self.name}: HTTP {status}",
            retryable=status in _RETRYABLE_STATUS,
            url=url,
            status=status,
        )

    def _cache_key(self, url: str, params: Mapping[str, Any] | None) -> str:
        if not params:
            return f"{self.name}:GET:{url}"
        rendered = "&".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{self.name}:GET:{url}?{rendered}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def registered_source_names() -> list[str]:
    return list(_SOURCE_MODULES)


def get_source(name: str, config: Config, **kw: Any) -> Source:
    """Construct one adapter by registry name."""
    entry = _SOURCE_MODULES.get(name)
    if entry is None:
        raise SourceError(
            f"unknown source {name!r}", retryable=False, known=sorted(_SOURCE_MODULES)
        )
    module_name, class_name = entry
    module = importlib.import_module(module_name, package=__name__)
    factory = getattr(module, class_name)
    source: Source = factory(config, **kw)
    return source


def available_sources(config: Config, **kw: Any) -> list[Source]:
    """Every configured source that can actually be constructed, in config order.

    A source that needs a key it does not have is skipped with a warning: one
    missing credential must degrade a survey, never abort it.
    """
    logger = kw.get("logger")
    sources: list[Source] = []
    for name in config.literature.sources:
        try:
            sources.append(get_source(name, config, **kw))
        except Exception as exc:  # noqa: BLE001 - a broken adapter must not stop the survey
            emit(logger, "warning", "literature.source_unavailable", source=name, error=str(exc))
    return sources


__all__ = [
    "BaseSource",
    "HttpClient",
    "HttpReply",
    "HttpTransport",
    "Source",
    "TokenBucket",
    "available_sources",
    "get_source",
    "registered_source_names",
]
