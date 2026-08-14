"""Shallow clone/pin to a commit into a content-addressed cache; never runs
repo code at fetch time."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..config import Config
from ..errors import SourceError
from ..observability.logging import StructuredLogger, get_logger
from ..types import CodeRepo, Provenance
from ..utils.hashing import hash_dir, hash_text
from ..utils.io import ensure_dir
from .license import detect_from_repo_dir

# Cloning is not running. Every one of these disables a git feature that would
# execute repository-controlled code during the clone itself.
_SAFE_GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "true",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_ALLOW_PROTOCOL": "https",
    "GCM_INTERACTIVE": "never",
}
_SAFE_GIT_FLAGS = [
    "-c",
    "core.hooksPath=/dev/null",  # repo-supplied hooks never run
    "-c",
    "core.fsmonitor=false",
    "-c",
    "protocol.ext.allow=never",
    "-c",
    "submodule.recurse=false",
]


def cache_path(config: Config, repo: CodeRepo) -> Path:
    """Content-addressed by URL and pinned commit, so a pin is never overwritten."""
    digest = hash_text(f"{repo.url}@{repo.commit or 'HEAD'}", length=16)
    return Path(config.paths.repos) / f"{repo.name or 'repo'}-{digest}"


def fetch_repo(
    repo: CodeRepo,
    config: Config,
    *,
    logger: StructuredLogger | None = None,
    force: bool = False,
) -> CodeRepo:
    """Shallow-clone a repo and pin it to a commit. Executes nothing from it.

    Returns an updated ``CodeRepo`` carrying ``local_path``, the resolved
    ``commit`` and ``code_hash``. The license is re-read from the clone, because
    the host's metadata and the actual LICENSE file disagree more often than
    they should.
    """
    log = (logger or get_logger("code.fetch")).bind(repo=repo.url)
    target = cache_path(config, repo)

    if target.exists() and not force:
        log.info("repo_cache_hit", path=str(target))
        return _finalize(repo, target, log)

    if config.offline:
        raise SourceError(
            "offline: repository is not in the local cache", url=repo.url, retryable=False
        )
    if shutil.which("git") is None:
        raise SourceError("git is not available on this machine", retryable=False)

    ensure_dir(target.parent)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

    command = [
        "git",
        *_SAFE_GIT_FLAGS,
        "clone",
        "--depth",
        str(config.code.clone_depth),
        "--no-tags",
        "--no-recurse-submodules",
        "--quiet",
        repo.url,
        str(target),
    ]
    log.info("repo_clone", depth=config.code.clone_depth)
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        command,
        capture_output=True,
        text=True,
        timeout=600,
        env={**_SAFE_GIT_ENV, "PATH": "/usr/bin:/bin:/usr/local/bin"},
        check=False,
    )
    if result.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        raise SourceError("git clone failed", url=repo.url, stderr=result.stderr[-800:])

    if repo.commit:
        _checkout(target, repo.commit, log)

    size_mb = _size_mb(target)
    if size_mb > config.code.max_repo_mb:
        shutil.rmtree(target, ignore_errors=True)
        raise SourceError(
            "repository exceeds the size ceiling",
            url=repo.url,
            size_mb=round(size_mb, 1),
            limit_mb=config.code.max_repo_mb,
            retryable=False,
        )
    return _finalize(repo, target, log)


def _checkout(path: Path, commit: str, log: StructuredLogger) -> None:
    """Pin to an exact commit; a shallow clone may need to fetch it first."""
    for command in (
        ["git", *_SAFE_GIT_FLAGS, "-C", str(path), "fetch", "--depth", "1", "origin", commit],
        ["git", *_SAFE_GIT_FLAGS, "-C", str(path), "checkout", "--quiet", commit],
    ):
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=300,
            env={**_SAFE_GIT_ENV, "PATH": "/usr/bin:/bin:/usr/local/bin"},
            check=False,
        )
        if result.returncode != 0 and "checkout" in command:
            log.warning("commit_pin_failed", commit=commit, stderr=result.stderr[-300:])


def resolve_commit(path: Path) -> str | None:
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _finalize(repo: CodeRepo, path: Path, log: StructuredLogger) -> CodeRepo:
    """Attach local facts to the repo record. Still nothing has been executed."""
    readme = ""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        candidate = path / name
        if candidate.exists():
            readme = candidate.read_text(encoding="utf-8", errors="replace")[:40000]
            break

    license_info = detect_from_repo_dir(path)
    if license_info.spdx_id is None and repo.license.spdx_id:
        license_info = repo.license  # keep the host's answer when the file is absent

    code_hash = hash_dir(path)
    log.info("repo_ready", path=str(path), code_hash=code_hash, license=license_info.spdx_id)
    return repo.model_copy(
        update={
            "local_path": str(path),
            "commit": repo.commit or resolve_commit(path),
            "readme": readme or repo.readme,
            "license": license_info,
            "code_hash": code_hash,
            "provenance": [*repo.provenance, Provenance(source=f"file:{path}", locator=code_hash)],
        }
    )


def _size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total / (1024 * 1024)


def clear_repo_cache(config: Config) -> int:
    removed = 0
    root = Path(config.paths.repos)
    if not root.exists():
        return 0
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


__all__ = ["cache_path", "clear_repo_cache", "fetch_repo", "resolve_commit"]
