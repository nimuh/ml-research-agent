"""Stable content hashing for dedup, caching and reproducibility ids."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def _canonical(value: Any) -> Any:
    """Reduce a value to a canonical, JSON-serializable form.

    Ordering is made deterministic (dict keys sorted, sets sorted) so that two
    semantically equal objects always hash the same. Floats keep full repr;
    rounding is the caller's decision, not the hasher's.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, set | frozenset):
        return sorted((_canonical(v) for v in value), key=repr)
    if isinstance(value, bytes | bytearray):
        return hashlib.sha256(bytes(value)).hexdigest()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Iterable):
        return [_canonical(v) for v in value]
    dumped = getattr(value, "model_dump", None)
    if callable(dumped):
        return _canonical(dumped(mode="json"))
    return repr(value)


def canonical_json(value: Any) -> str:
    """Deterministic JSON encoding used as the input to every structural hash."""
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_text(text: str, *, length: int = 64) -> str:
    """SHA-256 of a string, truncated to ``length`` hex chars."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def hash_bytes(data: bytes, *, length: int = 64) -> str:
    """SHA-256 of raw bytes, truncated to ``length`` hex chars."""
    return hashlib.sha256(data).hexdigest()[:length]


def hash_obj(value: Any, *, length: int = 16) -> str:
    """Structural hash of any (nested) object via canonical JSON."""
    return hash_text(canonical_json(value), length=length)


def hash_file(path: str | Path, *, length: int = 64) -> str:
    """Content hash of a file, streamed so large PDFs/checkpoints stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def hash_dir(
    path: str | Path,
    *,
    ignore: Iterable[str] = (".git", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"),
    length: int = 16,
) -> str:
    """Content hash of a directory tree: relative paths plus file contents.

    This is the ``code_hash`` half of the reproducibility contract, so it must
    depend on file *content* and *layout* but not on mtimes or absolute paths.
    """
    root = Path(path)
    ignored = set(ignore)
    entries: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ignored)
        for name in sorted(filenames):
            if name in ignored:
                continue
            full = Path(dirpath) / name
            if full.is_symlink() or not full.is_file():
                continue
            entries.append((full.relative_to(root).as_posix(), hash_file(full)))
    return hash_obj(entries, length=length)


def short_hash(value: Any, *, length: int = 8) -> str:
    """A short structural hash for human-facing ids."""
    return hash_obj(value, length=length)


__all__ = [
    "canonical_json",
    "hash_bytes",
    "hash_dir",
    "hash_file",
    "hash_obj",
    "hash_text",
    "short_hash",
]
