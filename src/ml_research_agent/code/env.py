"""Environment resolution: python/CUDA versions, lockfile or container spec,
so a Recipe can be rebuilt deterministically."""

from __future__ import annotations

import re
from pathlib import Path

from ..types import CodeRepo, EnvSpec, RepoMap

_CUDA = re.compile(r"cuda[_\-/]?(\d{1,2})[._](\d)", re.I)
_TORCH_CUDA = re.compile(r"torch.*?\+cu(\d{2,3})", re.I)
_FROM_IMAGE = re.compile(r"^\s*FROM\s+(\S+)", re.I | re.M)
_PY_VERSION = re.compile(r"(\d+\.\d+)")

# Pins that make an environment reproducible. Anything looser than `==` means a
# rebuild can silently pick up a different version, which breaks the env_hash's
# claim to identify the environment.
_PINNED = re.compile(r"^([A-Za-z0-9_.\-\[\]]+)\s*==\s*([\w.\-+]+)")


def resolve_env(repo: CodeRepo, repo_map: RepoMap, *, fallback_python: str = "3.11") -> EnvSpec:
    """Best-effort reconstruction of the environment a repo expects."""
    root = Path(repo.local_path) if repo.local_path else None
    python_version = _normalize_python(repo_map.python_version) or fallback_python
    packages = _packages(root) if root else []
    cuda = _cuda_version(root, repo)
    image = _container_image(root) if root else None
    lockfile = _lockfile(root) if root else None

    return EnvSpec(
        python_version=python_version,
        cuda_version=cuda,
        packages=packages,
        lockfile_path=lockfile,
        container_image=image,
    )


def _normalize_python(raw: str | None) -> str | None:
    """Turn ``>=3.9,<3.12`` into a concrete version we can actually install."""
    if not raw:
        return None
    versions = _PY_VERSION.findall(raw)
    if not versions:
        return None
    if raw.strip().startswith(">="):
        # Take the floor: the repo was tested against it, and picking the ceiling
        # of an open range means installing a version nobody has tried.
        return str(versions[0])
    return str(versions[-1] if "<" in raw else versions[0])


def _packages(root: Path) -> list[str]:
    requirements = root / "requirements.txt"
    packages: list[str] = []
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.split("#")[0].strip()
            if line and not line.startswith("-"):
                packages.append(line)
    environment = root / "environment.yml"
    if environment.exists() and not packages:
        for line in environment.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip().lstrip("- ")
            if stripped and not stripped.endswith(":") and "=" in stripped:
                packages.append(
                    stripped.replace("=", "==", 1) if "==" not in stripped else stripped
                )
    return packages[:200]


def _cuda_version(root: Path | None, repo: CodeRepo) -> str | None:
    haystacks = [repo.readme or ""]
    if root:
        for name in ("Dockerfile", "requirements.txt", "environment.yml", "setup.py"):
            path = root / name
            if path.exists():
                haystacks.append(path.read_text(encoding="utf-8", errors="replace")[:20000])
    for text in haystacks:
        torch_match = _TORCH_CUDA.search(text)
        if torch_match:
            raw = torch_match.group(1)
            return f"{raw[:-1]}.{raw[-1]}" if len(raw) >= 3 else raw
        match = _CUDA.search(text)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
    return None


def _container_image(root: Path) -> str | None:
    dockerfile = root / "Dockerfile"
    if not dockerfile.exists():
        return None
    match = _FROM_IMAGE.search(dockerfile.read_text(encoding="utf-8", errors="replace"))
    return match.group(1) if match else None


def _lockfile(root: Path) -> str | None:
    for name in ("uv.lock", "poetry.lock", "Pipfile.lock", "conda-lock.yml", "requirements.lock"):
        if (root / name).exists():
            return name
    return None


def pin_report(env: EnvSpec) -> list[str]:
    """Name every way this environment is not actually reproducible."""
    problems: list[str] = []
    unpinned = [p for p in env.packages if not _PINNED.match(p)]
    if unpinned:
        problems.append(
            f"{len(unpinned)} of {len(env.packages)} dependencies are unpinned "
            f"(e.g. {', '.join(unpinned[:4])}); a rebuild may not match this run"
        )
    if not env.packages and not env.container_image:
        problems.append(
            "no dependency manifest and no container image; the environment is unspecified"
        )
    if env.cuda_version is None and any("torch" in p or "jax" in p for p in env.packages):
        problems.append("a GPU framework is required but no CUDA version is declared")
    if not env.lockfile_path and not env.container_image:
        problems.append("no lockfile; exact transitive versions are not captured")
    return problems


def is_reproducible(env: EnvSpec) -> bool:
    """A container or a lockfile is the bar; a requirements.txt is not."""
    return bool(env.container_image or env.lockfile_path)


def install_commands(env: EnvSpec, *, workspace: str = ".") -> list[str]:
    """The commands that rebuild this environment, in order."""
    if env.container_image:
        return [f"docker pull {env.container_image}"]
    commands = [f"python{env.python_version} -m venv {workspace}/.venv"]
    if env.lockfile_path and env.lockfile_path == "uv.lock":
        commands.append("uv sync --frozen")
    elif env.lockfile_path == "poetry.lock":
        commands.append("poetry install --no-root")
    elif env.packages:
        commands.append(f"{workspace}/.venv/bin/pip install -r requirements.txt")
    return commands


__all__ = ["install_commands", "is_reproducible", "pin_report", "resolve_env"]
