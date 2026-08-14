"""Static repo mapping: entrypoints, config system, data pipeline, model defs,
train/eval loops, dependency graph, hardware assumptions."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

from ..config import Config
from ..errors import MRAError
from ..types import CodeRepo, RepoMap

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".eggs",
    "site-packages",
    ".idea",
}
CONFIG_FILES = (
    "config.yaml",
    "config.yml",
    "conf",
    "configs",
    "hydra",
    "params.yaml",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "environment.yml",
    "Makefile",
    "Dockerfile",
    "pipfile",
    "poetry.lock",
    "uv.lock",
)
_TRAIN_HINT = re.compile(r"\b(train|finetune|fit|pretrain)\b", re.I)
_EVAL_HINT = re.compile(r"\b(eval|evaluate|test|inference|predict|benchmark)\b", re.I)
_MODEL_HINT = re.compile(r"\b(model|network|architecture|modeling|modules?)\b", re.I)
_DATA_HINT = re.compile(r"\b(data|dataset|dataloader|loader|corpus|tokeniz)\b", re.I)

_HARDWARE = (
    (re.compile(r"\bA100\b|\bH100\b|\bV100\b|\bTPU\b", re.I), "specific accelerator named"),
    (
        re.compile(r"torch\.distributed|DistributedDataParallel|deepspeed|fsdp|accelerate", re.I),
        "multi-GPU / distributed training",
    ),
    (re.compile(r"\b(\d+)\s*x\s*(?:GPU|A100|V100|H100)\b", re.I), "multiple GPUs assumed"),
    (re.compile(r"\.cuda\(\)|device\s*=\s*[\"']cuda", re.I), "CUDA assumed available"),
    (re.compile(r"bf16|fp16|amp\.autocast|mixed_precision", re.I), "mixed precision"),
)


def analyze_repo(repo: CodeRepo, config: Config) -> RepoMap:
    """Map a fetched repo statically. Reads files; runs nothing.

    Everything here is heuristic and deliberately so -- its job is to give the
    CodeAnalyst agent a map good enough to write a recipe against, not to be a
    build system.
    """
    if not repo.local_path:
        raise MRAError("cannot analyze a repo that was never fetched", repo=repo.url)
    root = Path(repo.local_path)
    if not root.exists():
        raise MRAError("repo path does not exist", path=str(root))

    files = list(_walk(root, limit=config.code.max_files_to_analyze))
    python_files = [f for f in files if f.suffix == ".py"]

    entrypoints: list[str] = []
    train_scripts: list[str] = []
    eval_scripts: list[str] = []
    model_files: list[str] = []
    data_files: list[str] = []
    dependencies: set[str] = set()
    hardware: set[str] = set()
    loc = 0

    for path in python_files:
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        loc += text.count("\n")

        if _has_main_guard(text):
            entrypoints.append(relative)
        if _TRAIN_HINT.search(relative) or "def train" in text:
            train_scripts.append(relative)
        if _EVAL_HINT.search(relative) or "def evaluate" in text:
            eval_scripts.append(relative)
        if _MODEL_HINT.search(relative) or "nn.Module" in text:
            model_files.append(relative)
        if _DATA_HINT.search(relative) or "Dataset" in text:
            data_files.append(relative)

        dependencies |= _imports(text)
        for pattern, label in _HARDWARE:
            if pattern.search(text):
                hardware.add(label)

    config_files = sorted(
        f.relative_to(root).as_posix()
        for f in files
        if f.name in CONFIG_FILES or f.parent.name in ("configs", "conf")
    )
    dependencies |= _declared_dependencies(root)

    readme = repo.readme or ""
    for pattern, label in _HARDWARE:
        if pattern.search(readme):
            hardware.add(label)

    return RepoMap(
        repo_id=repo.id,
        entrypoints=sorted(set(entrypoints))[:30],
        config_files=config_files[:30],
        config_system=_config_system(root, config_files),
        train_scripts=sorted(set(train_scripts))[:20],
        eval_scripts=sorted(set(eval_scripts))[:20],
        model_files=sorted(set(model_files))[:30],
        data_files=sorted(set(data_files))[:30],
        dependencies=sorted(d for d in dependencies if d)[:80],
        python_version=_python_version(root),
        hardware_assumptions=sorted(hardware),
        file_count=len(files),
        loc=loc,
        notes=_notes(root, files),
    )


def file_excerpts(
    repo: CodeRepo, repo_map: RepoMap, *, limit: int = 12, chars: int = 4000
) -> dict[str, str]:
    """The files the agent most needs to read, excerpted for the prompt."""
    if not repo.local_path:
        return {}
    root = Path(repo.local_path)
    ordered = [
        *repo_map.entrypoints[:4],
        *repo_map.train_scripts[:3],
        *repo_map.eval_scripts[:2],
        *repo_map.config_files[:3],
    ]
    excerpts: dict[str, str] = {}
    for relative in ordered:
        if len(excerpts) >= limit or relative in excerpts:
            continue
        path = root / relative
        if path.exists() and path.is_file():
            try:
                excerpts[relative] = path.read_text(encoding="utf-8", errors="replace")[:chars]
            except OSError:
                continue
    return excerpts


def _walk(root: Path, *, limit: int) -> Iterable[Path]:
    count = 0
    for path in sorted(root.rglob("*")):
        if count >= limit:
            return
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and not path.is_symlink():
            count += 1
            yield path


def _has_main_guard(text: str) -> bool:
    return '__name__ == "__main__"' in text or "__name__ == '__main__'" in text


def _imports(text: str) -> set[str]:
    """Top-level third-party imports, parsed rather than regexed where possible."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return {m for m in modules if m and not m.startswith("_")} - _STDLIB


def _declared_dependencies(root: Path) -> set[str]:
    deps: set[str] = set()
    requirements = root / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "-")):
                deps.add(re.split(r"[=<>!~\[; ]", line)[0])
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        deps |= {
            re.split(r"[=<>!~\[; ]", d.strip(" \"'"))[0]
            for d in re.findall(r'"([A-Za-z0-9_.-]+[^"]*)"', text)
            if d and not d.startswith(("http", "./"))
        }
    return {d for d in deps if d and len(d) < 40}


def _config_system(root: Path, config_files: list[str]) -> str | None:
    if any("hydra" in f or f.startswith("conf/") for f in config_files):
        return "hydra"
    if (root / "params.yaml").exists():
        return "dvc/params"
    if any(f.startswith("configs/") for f in config_files):
        return "yaml configs"
    if any(f in ("setup.py", "pyproject.toml") for f in config_files):
        return "argparse or packaging metadata"
    return None


def _python_version(root: Path) -> str | None:
    for name, pattern in (
        (".python-version", re.compile(r"(\d+\.\d+(?:\.\d+)?)")),
        ("pyproject.toml", re.compile(r'requires-python\s*=\s*"([^"]+)"')),
        ("setup.py", re.compile(r'python_requires\s*=\s*["\']([^"\']+)')),
    ):
        path = root / name
        if path.exists():
            match = pattern.search(path.read_text(encoding="utf-8", errors="replace"))
            if match:
                return match.group(1).strip()
    return None


def _notes(root: Path, files: list[Path]) -> list[str]:
    """Things a human would want flagged before trying to run this."""
    notes: list[str] = []
    names = {f.name for f in files}
    if "Dockerfile" in names:
        notes.append("ships a Dockerfile; prefer it for environment fidelity")
    if not (names & {"requirements.txt", "pyproject.toml", "environment.yml", "setup.py"}):
        notes.append("no dependency manifest; the environment must be reconstructed by hand")
    if not any(f.name.startswith("test") for f in files):
        notes.append(
            "no visible tests; correctness cannot be checked before running the experiment"
        )
    if (root / ".gitmodules").exists():
        notes.append("uses git submodules, which the shallow clone did not fetch")
    if len(files) >= 400:
        notes.append("large repository; static analysis was truncated")
    return notes


_STDLIB = {
    "abc",
    "argparse",
    "ast",
    "asyncio",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "glob",
    "hashlib",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "shutil",
    "socket",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "typing",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "zipfile",
}

__all__ = ["CONFIG_FILES", "SKIP_DIRS", "analyze_repo", "file_excerpts"]
