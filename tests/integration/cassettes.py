"""A tiny cassette player for the literature source adapters.

Deliberately hand-rolled rather than vcrpy: the adapters take an injected
transport, so standing in for one is a dozen lines, and an unmatched request
must be a loud ``AssertionError`` -- a test that silently reaches the network
is worse than a test that fails.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ml_research_agent.literature.sources import HttpReply

CASSETTE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "cassettes"


def _canonical(params: Mapping[str, Any] | None) -> dict[str, str]:
    return {str(key): str(value) for key, value in (params or {}).items()}


class CassettePlayer:
    """Replays recorded HTTP replies; refuses to invent one.

    An entry matches when the method and url are equal and every parameter the
    entry names matches the request. Parameters the entry does not name are
    ignored, so a cassette pins the request fields that carry meaning rather
    than every field the adapter happens to send.
    """

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.played: list[str] = []

    @classmethod
    def load(cls, *names: str) -> CassettePlayer:
        entries: list[dict[str, Any]] = []
        for name in names:
            path = CASSETTE_DIR / f"{name}.json"
            entries.extend(json.loads(path.read_text(encoding="utf-8")))
        return cls(entries)

    @property
    def request_count(self) -> int:
        return len(self.requests)

    @property
    def unplayed(self) -> list[str]:
        return [
            e.get("name", "<unnamed>") for e in self.entries if e.get("name") not in self.played
        ]

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> HttpReply:
        canonical = _canonical(params)
        self.requests.append((method.upper(), url, canonical))
        for entry in self.entries:
            if entry.get("method", "GET").upper() != method.upper():
                continue
            if entry["url"] != url:
                continue
            expected = _canonical(entry.get("params"))
            if any(canonical.get(key) != value for key, value in expected.items()):
                continue
            self.played.append(entry.get("name", "<unnamed>"))
            return HttpReply(
                status_code=int(entry.get("status", 200)),
                content=_body_of(entry).encode("utf-8"),
                headers=dict(entry.get("headers") or {}),
                url=url,
            )
        raise AssertionError(
            f"no cassette entry for {method.upper()} {url} params={canonical!r}; "
            f"available: {[e.get('name') for e in self.entries]}"
        )

    def close(self) -> None:  # the adapters call this on shutdown
        return None


class ExplodingTransport:
    """A transport that must never be reached (offline tests)."""

    def __init__(self) -> None:
        self.calls = 0

    def request(self, method: str, url: str, **_kw: Any) -> HttpReply:
        self.calls += 1
        raise AssertionError(f"the network was touched: {method} {url}")

    def close(self) -> None:
        return None


def _body_of(entry: Mapping[str, Any]) -> str:
    if "json" in entry:
        return json.dumps(entry["json"])
    body = entry.get("body", "")
    return "\n".join(body) if isinstance(body, list) else str(body)


__all__ = ["CASSETTE_DIR", "CassettePlayer", "ExplodingTransport"]
