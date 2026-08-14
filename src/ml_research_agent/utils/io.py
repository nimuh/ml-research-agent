"""Atomic file writes, JSONL/YAML helpers, safe path handling."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from ..errors import SandboxViolation


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if missing; return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def atomic_write(
    path: str | Path, *, mode: str = "w", encoding: str | None = "utf-8"
) -> Iterator[Any]:
    """Write via a temp file in the same directory, then ``os.replace``.

    A crash mid-write leaves the previous file intact rather than a truncated
    one -- the KB and the event log both depend on that.
    """
    target = Path(path)
    ensure_dir(target.parent)
    binary = "b" in mode
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        with open(tmp_path, mode, encoding=None if binary else encoding) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_text(path: str | Path, text: str) -> Path:
    """Atomically write UTF-8 text."""
    with atomic_write(path) as handle:
        handle.write(text)
    return Path(path)


def read_text(path: str | Path, *, default: str | None = None) -> str:
    """Read UTF-8 text, optionally falling back to ``default`` when absent."""
    p = Path(path)
    if not p.exists():
        if default is None:
            raise FileNotFoundError(str(p))
        return default
    return p.read_text(encoding="utf-8")


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    """Atomically write JSON."""
    with atomic_write(path) as handle:
        json.dump(obj, handle, indent=indent, ensure_ascii=False, default=str)
        handle.write("\n")
    return Path(path)


def read_json(path: str | Path, *, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file is absent."""
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Any) -> None:
    """Append one JSON record plus newline.

    Deliberately *not* atomic-by-replace: the event log is append-only and a
    single ``write`` of a short line is atomic enough on POSIX for our use.
    """
    ensure_dir(Path(path).parent)
    line = json.dumps(record, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def read_jsonl(path: str | Path) -> Iterator[Any]:
    """Stream JSON records; tolerates a truncated final line."""
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str | Path, records: Iterable[Any]) -> Path:
    """Atomically (re)write a whole JSONL file."""
    with atomic_write(path) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return Path(path)


def read_yaml(path: str | Path, *, default: Any = None) -> Any:
    """Read YAML, returning ``default`` when the file is absent."""
    p = Path(path)
    if not p.exists():
        return default
    loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
    return default if loaded is None else loaded


def write_yaml(path: str | Path, obj: Any) -> Path:
    """Atomically write YAML."""
    with atomic_write(path) as handle:
        yaml.safe_dump(obj, handle, sort_keys=False, allow_unicode=True)
    return Path(path)


def safe_join(root: str | Path, *parts: str) -> Path:
    """Join under ``root`` and refuse to escape it.

    Path traversal is prevented *by construction* here rather than by checking
    for ``..`` -- symlinks and absolute components are resolved first.
    """
    base = Path(root).resolve()
    candidate = base
    for part in parts:
        candidate = candidate / part
    resolved = candidate.resolve()
    if resolved != base and base not in resolved.parents:
        raise SandboxViolation(
            "path escapes its workspace root", root=str(base), path=str(resolved)
        )
    return resolved


def slugify(text: str, *, max_length: int = 64) -> str:
    """Filesystem-safe slug for note/report filenames."""
    cleaned = "".join(c.lower() if c.isalnum() else "-" for c in text.strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:max_length] or "untitled"


def front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split ``---`` YAML front matter from a Markdown body."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        return {}, text
    return meta, parts[2].lstrip("\n")


def with_front_matter(meta: dict[str, Any], body: str) -> str:
    """Render YAML front matter + Markdown body."""
    head = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{head}\n---\n\n{body.strip()}\n"


__all__ = [
    "append_jsonl",
    "atomic_write",
    "ensure_dir",
    "front_matter",
    "read_json",
    "read_jsonl",
    "read_text",
    "read_yaml",
    "safe_join",
    "slugify",
    "with_front_matter",
    "write_json",
    "write_jsonl",
    "write_text",
    "write_yaml",
]
