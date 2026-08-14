"""Outbound HTTP with domain policy, rate limiting, caching and robots respect."""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from ..config import Config
from ..errors import ConfigError, SandboxViolation, SourceError, ToolError
from ..observability.logging import StructuredLogger, get_logger
from ..utils.cache import DiskCache
from ..utils.concurrency import retry_sync
from ..utils.hashing import hash_obj
from .registry import ToolContext, ToolResult

MAX_BODY_CHARS = 100_000

# Hosts nothing in this system has any business reaching. Kept as a deny-list
# on top of the optional allow-list because SSRF into cloud metadata is the
# failure mode that turns "fetch a URL" into credential theft.
DENIED_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254", "metadata.google.internal"}
)

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class _Throttle:
    """Minimum-interval throttle.

    ``utils.concurrency.RateLimiter`` is async-only and this client is sync
    (the literature layer calls it from threads), so politeness is enforced
    here with the same token-bucket intent.
    """

    def __init__(self, rate_per_second: float) -> None:
        self.min_interval = 1.0 / rate_per_second if rate_per_second > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class HttpClient:
    """Sync HTTP with domain policy, rate limiting, on-disk caching and retries.

    Shared by the ``http`` tool and by every literature source adapter, so
    there is exactly one place that knows about politeness, caching and the
    offline switch.
    """

    def __init__(
        self,
        config: Config,
        *,
        namespace: str = "http",
        allow_domains: list[str] | None = None,
        deny_domains: list[str] | None = None,
        rate_limit_per_second: float | None = None,
        ttl_seconds: float | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.config = config
        self.allow_domains = [d.lower() for d in (allow_domains or [])]
        self.deny_domains = {d.lower() for d in (deny_domains or [])} | DENIED_HOSTS
        self.logger = logger or get_logger("tools.http")
        self.throttle = _Throttle(
            rate_limit_per_second
            if rate_limit_per_second is not None
            else config.literature.rate_limit_per_second
        )
        self.cache = DiskCache(
            config.paths.cache,
            namespace=namespace,
            ttl_seconds=(
                ttl_seconds if ttl_seconds is not None else config.literature.cache_ttl_seconds
            ),
        )
        self._client: httpx.Client | None = None

    # -- policy -------------------------------------------------------------

    def check_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise SandboxViolation("only http(s) URLs are permitted", url=url[:200])
        host = (parsed.hostname or "").lower()
        if not host:
            raise SandboxViolation("URL has no host", url=url[:200])
        if any(host == d or host.endswith("." + d) for d in self.deny_domains):
            raise SandboxViolation("host is denied", host=host)
        if self.allow_domains and not any(
            host == d or host.endswith("." + d) for d in self.allow_domains
        ):
            raise SandboxViolation("host is not on the allow-list", host=host)
        return host

    # -- requests -----------------------------------------------------------

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> str:
        return str(self._get(url, params=params, headers=headers, use_cache=use_cache)["text"])

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = True,
    ) -> Any:
        import json

        payload = self._get(url, params=params, headers=headers, use_cache=use_cache)
        try:
            return json.loads(payload["text"])
        except json.JSONDecodeError as exc:
            raise SourceError("response was not JSON", url=url[:200], detail=str(exc)) from exc

    def get_bytes(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """Binary fetch (PDFs, tarballs). Deliberately not cached as JSON."""
        self.check_url(url)
        if self.config.offline:
            raise ConfigError("offline mode: refusing a network fetch", url=url[:200])
        response = self._request(url, params=params, headers=headers)
        return response.content

    def _get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        headers: dict[str, str] | None,
        use_cache: bool,
    ) -> dict[str, Any]:
        host = self.check_url(url)
        key = hash_obj({"url": url, "params": params or {}})
        if use_cache:
            cached = self.cache.get(key)
            if isinstance(cached, dict):
                self.logger.debug("http cache hit", host=host, url=url[:200])
                return cached
        if self.config.offline:
            raise ConfigError("offline mode: no cached response for URL", url=url[:200])
        response = self._request(url, params=params, headers=headers)
        payload = {
            "url": str(response.url),
            "status": response.status_code,
            "text": response.text,
            "content_type": response.headers.get("content-type", ""),
        }
        if use_cache:
            self.cache.set(key, payload)
        return payload

    def _request(
        self, url: str, *, params: dict[str, Any] | None, headers: dict[str, str] | None
    ) -> httpx.Response:
        merged = {"user-agent": self.config.literature.user_agent, **(headers or {})}

        def _attempt() -> httpx.Response:
            self.throttle.acquire()
            try:
                response = self.client().get(url, params=params, headers=merged)
            except httpx.HTTPError as exc:
                raise SourceError("http request failed", url=url[:200], detail=str(exc)) from exc
            if response.status_code in _RETRYABLE_STATUS:
                raise SourceError(
                    "retryable http status", url=url[:200], status=response.status_code
                )
            if response.status_code >= 400:
                raise SourceError(
                    "http error",
                    retryable=False,
                    url=url[:200],
                    status=response.status_code,
                    body=response.text[:300],
                )
            return response

        def _on_retry(attempt: int, exc: BaseException) -> None:
            self.logger.warning("http retry", url=url[:200], attempt=attempt, detail=str(exc)[:200])

        return retry_sync(_attempt, attempts=3, on_retry=_on_retry, retry_on=(SourceError,))

    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.config.literature.request_timeout_seconds,
                follow_redirects=True,
                headers={"user-agent": self.config.literature.user_agent},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class HttpArgs(BaseModel):
    url: str = Field(description="Absolute http(s) URL.")
    max_chars: int = Field(default=20_000, ge=100, le=MAX_BODY_CHARS)


class HttpTool:
    """Fetch a URL as text, under the same policy every source adapter uses."""

    name = "http_get"
    description = (
        "Fetch an http(s) URL and return its body as text. Subject to domain policy, rate "
        "limiting and on-disk caching; refuses local and cloud-metadata addresses. Responses "
        "are truncated at max_chars with an explicit marker."
    )
    parameters = HttpArgs

    def __init__(self, config: Config, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient(config, namespace="tool")

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        if not isinstance(args, HttpArgs):
            raise ToolError("unexpected argument type", expected="HttpArgs")
        text = self.client.get_text(args.url)
        truncated = len(text) > args.max_chars
        body = text[: args.max_chars]
        if truncated:
            body += f"\n... [truncated {len(text) - args.max_chars} chars]"
        return ToolResult(
            ok=True,
            output=body,
            metadata={"url": args.url[:200], "chars": len(text), "truncated": truncated},
        )


__all__ = ["DENIED_HOSTS", "HttpArgs", "HttpClient", "HttpTool"]
