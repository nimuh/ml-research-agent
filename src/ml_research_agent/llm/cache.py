"""Deterministic response cache keyed by (model, prompt hash, params) so replays
and tests are cheap and repeatable."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.cache import DiskCache
from ..utils.hashing import hash_obj

NAMESPACE = "llm"

# Params that change the sampled text. Anything else (timeouts, retry counts,
# correlation ids) must not participate in the key or replays would never hit.
KEYED_PARAMS = (
    "max_tokens",
    "temperature",
    "stop_sequences",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "effort",
)


class ResponseCache:
    """Content-addressed LLM response cache.

    Determinism is the whole point: the same ``(model, messages, params)`` must
    always produce the same key, which is what makes golden-prompt tests fast
    and a replayed run free.
    """

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache = DiskCache(root, namespace=NAMESPACE)

    def key(self, *, model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
        relevant = {k: params[k] for k in KEYED_PARAMS if k in params and params[k] is not None}
        return hash_obj({"model": model, "messages": messages, "params": relevant}, length=32)

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        value = self._cache.get(key)
        if value is None or not isinstance(value, dict):
            return None
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._cache.set(key, value)

    def clear(self) -> int:
        return self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "root": str(self._cache.root), **self._cache.stats}


__all__ = ["KEYED_PARAMS", "NAMESPACE", "ResponseCache"]
