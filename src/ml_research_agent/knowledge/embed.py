"""Embedding backends (API or local), batched with a persistent cache keyed by
(model, content hash)."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import Config
from ..errors import ConfigError
from ..utils.cache import DiskCache
from ..utils.hashing import hash_text

_TOKEN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into L2-normalized float32 vectors."""

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic, offline, dependency-free embeddings.

    A signed hashed bag of unigrams and bigrams with sublinear term weighting:
    each feature is hashed with SHA-256 (never Python's salted ``hash``) into a
    bucket and a sign, so two texts sharing vocabulary land near each other and
    collisions cancel instead of accumulating.

    This is the default backend on purpose. The whole test suite -- and any
    offline run -- must be able to build and query the KB with no API key, and a
    lexical-ish signal fused with BM25 is enough for that.
    """

    def __init__(self, dim: int = 768, seed: int = 0) -> None:
        self.dim = int(dim)
        self.seed = int(seed)
        self.model_name = f"hashing-{self.dim}-{self.seed}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[str, int] = {}
            tokens = _TOKEN.findall(text.lower())
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for first, second in zip(tokens, tokens[1:], strict=False):
                bigram = f"{first}_{second}"
                counts[bigram] = counts.get(bigram, 0) + 1
            for feature, count in counts.items():
                bucket, sign = self._bucket(feature)
                out[row, bucket] += sign * (1.0 + math.log(count))
        return _normalize(out)

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.sha256(f"{self.seed}:{feature}".encode()).digest()
        value = int.from_bytes(digest[:8], "big")
        return value % self.dim, 1.0 if digest[8] & 1 else -1.0


class CachedEmbedder:
    """Wraps an embedder with a persistent cache keyed by (model, content hash).

    Embedding is the expensive half of ingest for API backends and the repeated
    half for local ones; caching here is what keeps re-ingest and index rebuilds
    cheap enough to be routine.
    """

    def __init__(self, inner: Embedder, cache_dir: Path, *, model_name: str) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.model_name = model_name
        self.cache = DiskCache(cache_dir, namespace=f"embed-{hash_text(model_name, length=8)}")
        self._memory: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing: list[int] = []
        for i, text in enumerate(texts):
            key = self._key(text)
            vector = self._memory.get(key)
            if vector is None:
                cached = self.cache.get(key)
                vector = np.asarray(cached, dtype=np.float32) if cached is not None else None
                if vector is not None:
                    self._memory[key] = vector
            if vector is None:
                missing.append(i)
            else:
                self.hits += 1
                out[i] = vector
        if missing:
            self.misses += len(missing)
            computed = self.inner.embed([texts[i] for i in missing])
            for slot, i in enumerate(missing):
                vector = np.asarray(computed[slot], dtype=np.float32)
                out[i] = vector
                key = self._key(texts[i])
                self._memory[key] = vector
                self.cache.set(key, [float(v) for v in vector])
        return out

    def _key(self, text: str) -> str:
        return f"{self.model_name}:{hash_text(text)}"


def get_embedder(config: Config) -> Embedder:
    """Build the configured embedder, always behind the cache."""
    backend = config.knowledge.embedding_backend
    if backend == "hashing":
        inner: Embedder = HashingEmbedder(dim=config.knowledge.embedding_dim)
        # The configured model name belongs in the cache identity even though the
        # hashing backend does not consult it: changing the model must invalidate
        # vectors computed by the previous one, or retrieval silently mixes them.
        model_name = f"hashing-{config.knowledge.embedding_dim}-{config.knowledge.embedding_model}"
    else:
        # Deliberate: an API-backed embedder belongs to the platform layer's
        # client, which knowledge/ must not import. Wire it in there and pass it
        # to HybridIndex rather than reaching upward from here.
        raise ConfigError(
            "only the offline 'hashing' embedding backend is implemented in knowledge/",
            backend=backend,
        )
    return CachedEmbedder(inner, Path(config.paths.cache) / "embeddings", model_name=model_name)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized: np.ndarray = (matrix / norms).astype(np.float32)
    return normalized


__all__ = ["CachedEmbedder", "Embedder", "HashingEmbedder", "get_embedder"]
