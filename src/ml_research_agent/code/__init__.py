"""Codebase knowledge: find, vet, and understand the reference implementations
that make an idea testable rather than merely describable."""

from .analyze import analyze_repo, file_excerpts
from .discover import discover_repos, fidelity_score, parse_repo_url
from .env import install_commands, is_reproducible, pin_report, resolve_env
from .fetch import cache_path, fetch_repo
from .license import dataset_gate, detect_license, license_gate
from .recipe import draft_recipe, extract_reference_numbers, is_runnable, recipe_gaps

__all__ = [
    "analyze_repo",
    "cache_path",
    "dataset_gate",
    "detect_license",
    "discover_repos",
    "draft_recipe",
    "extract_reference_numbers",
    "fetch_repo",
    "fidelity_score",
    "file_excerpts",
    "install_commands",
    "is_reproducible",
    "is_runnable",
    "license_gate",
    "parse_repo_url",
    "pin_report",
    "recipe_gaps",
    "resolve_env",
]
