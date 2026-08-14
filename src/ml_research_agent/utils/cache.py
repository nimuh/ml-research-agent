"""Generic on-disk cache with TTL and namespacing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .hashing import hash_text
from .io import ensure_dir, read_json, write_json


class DiskCache:
    """A namespaced content-addressed cache under one root directory.

    Keys are hashed, so callers may use arbitrarily long strings (URLs, prompt
    payloads). Entries store ``{"stored_at", "value"}`` so a TTL can be applied
    at read time without a background sweeper.
    """

    def __init__(
        self, root: str | Path, *, namespace: str = "default", ttl_seconds: float | None = None
    ) -> None:
        self.root = ensure_dir(Path(root) / namespace)
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        digest = hash_text(key)
        # Two-level fan-out keeps directory listings sane at 10k+ entries.
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: str, *, default: Any = None) -> Any:
        path = self.path_for(key)
        record = read_json(path)
        if record is None:
            self.misses += 1
            return default
        if (
            self.ttl_seconds is not None
            and time.time() - float(record.get("stored_at", 0)) > self.ttl_seconds
        ):
            path.unlink(missing_ok=True)
            self.misses += 1
            return default
        self.hits += 1
        return record.get("value", default)

    def set(self, key: str, value: Any) -> None:
        write_json(self.path_for(key), {"stored_at": time.time(), "key": key, "value": value})

    def has(self, key: str) -> bool:
        return self.get(key, default=_MISSING) is not _MISSING

    def delete(self, key: str) -> None:
        self.path_for(key).unlink(missing_ok=True)

    def clear(self) -> int:
        removed = 0
        for path in self.root.rglob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    @property
    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }


class _Missing:
    pass


_MISSING = _Missing()


class BlobCache:
    """Content-addressed blob store for PDFs, LaTeX tarballs and repo snapshots.

    The digest *is* the identity, which is what makes ingest idempotent: the
    same bytes fetched twice occupy one path and are ingested once.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = ensure_dir(root)

    def path_for(self, digest: str, *, suffix: str = "") -> Path:
        return self.root / digest[:2] / f"{digest}{suffix}"

    def put(self, data: bytes, *, suffix: str = "") -> tuple[str, Path]:
        from .hashing import hash_bytes

        digest = hash_bytes(data)
        path = self.path_for(digest, suffix=suffix)
        if not path.exists():
            ensure_dir(path.parent)
            path.write_bytes(data)
        return digest, path

    def get(self, digest: str, *, suffix: str = "") -> bytes | None:
        path = self.path_for(digest, suffix=suffix)
        return path.read_bytes() if path.exists() else None


__all__ = ["BlobCache", "DiskCache"]
