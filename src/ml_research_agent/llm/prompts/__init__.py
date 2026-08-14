"""Versioned prompt assets, one file per agent role. Prompts are data, not code:
they are diffable, testable against golden cases, and referenced by version."""

from __future__ import annotations

import builtins
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...config import REPO_ROOT
from ...errors import ConfigError
from ...utils.io import front_matter, read_text

DEFAULT_ROOT = REPO_ROOT / "prompts"

_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class Prompt:
    """One versioned prompt: a system message plus a user-message template."""

    name: str
    version: str
    system: str
    template: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, **variables: Any) -> str:
        """Fill the template. A missing variable is a config error, not a hole.

        ``str.format`` semantics, so a literal brace in a prompt body must be
        doubled -- the alternative (silently leaving unknown placeholders in
        place) would ship ``{brief}`` to the model and look like a model bug.
        """
        try:
            return self.template.format(**variables)
        except KeyError as exc:
            raise ConfigError(
                "prompt template is missing a variable",
                prompt=self.name,
                version=self.version,
                variable=str(exc).strip("'"),
            ) from exc
        except (IndexError, ValueError) as exc:
            raise ConfigError(
                "prompt template is malformed",
                prompt=self.name,
                version=self.version,
                detail=str(exc),
            ) from exc

    @property
    def ref(self) -> str:
        """Provenance-friendly identifier, e.g. ``framer@v1``."""
        return f"{self.name}@{self.version}"


class PromptLibrary:
    """Loads ``prompts/<name>/<version>.md`` assets and caches them in memory.

    Storage format: YAML front matter carrying ``system:`` and any metadata,
    body is the user-message template.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self._cache: dict[tuple[str, str], Prompt] = {}

    def list(self) -> builtins.list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and self.versions(p.name))

    def versions(self, name: str) -> builtins.list[str]:
        directory = self.root / name
        if not directory.is_dir():
            return []
        found: builtins.list[str] = [p.stem for p in directory.glob("*.md")]
        return sorted(found, key=_version_sort_key)

    def get(self, name: str, *, version: str | None = None) -> Prompt:
        resolved = version or self.latest(name)
        cached = self._cache.get((name, resolved))
        if cached is not None:
            return cached
        path = self.root / name / f"{resolved}.md"
        if not path.exists():
            raise ConfigError(
                "prompt not found",
                prompt=name,
                version=resolved,
                path=str(path),
                available=self.versions(name),
            )
        prompt = _parse(name, resolved, read_text(path))
        self._cache[(name, resolved)] = prompt
        return prompt

    def latest(self, name: str) -> str:
        versions = self.versions(name)
        if not versions:
            raise ConfigError("no versions for prompt", prompt=name, root=str(self.root))
        return versions[-1]

    def clear_cache(self) -> None:
        self._cache.clear()


def _parse(name: str, version: str, text: str) -> Prompt:
    meta, body = front_matter(text)
    system = str(meta.pop("system", "")).strip()
    if not system:
        raise ConfigError("prompt has no system message", prompt=name, version=version)
    if not body.strip():
        raise ConfigError("prompt has an empty template body", prompt=name, version=version)
    return Prompt(name=name, version=version, system=system, template=body.strip(), metadata=meta)


def _version_sort_key(version: str) -> tuple[int, int, str]:
    """Numeric ordering, so ``v10`` sorts after ``v2``; odd names sort last."""
    match = _VERSION_RE.match(version)
    if match:
        return (0, int(match.group(1)), version)
    return (1, 0, version)


__all__ = ["DEFAULT_ROOT", "Prompt", "PromptLibrary"]
