"""Recipe: the runnable distillation of a repo -- exact commands, config knobs,
expected artifacts, reference numbers to reproduce, and known gotchas."""

from __future__ import annotations

import re
from pathlib import Path

from ..config import Config
from ..types import CodeRepo, MetricReport, Provenance, Recipe, RecipeStep, RepoMap
from .env import pin_report, resolve_env

# README table rows like "| ResNet-50 | 76.1 | 92.9 |" and prose like
# "achieves 76.1% top-1". Both are candidates only -- the CodeAnalyst decides.
_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$", re.M)
_INLINE_NUMBER = re.compile(
    r"(?:achieves?|reaches?|obtains?|reports?)\s+(?:an?\s+)?([\d.]+)\s*%?\s*"
    r"(accuracy|top-?1|top-?5|f1|bleu|rouge|mAP|AUC|perplexity|WER)",
    re.I,
)


def draft_recipe(
    repo: CodeRepo, repo_map: RepoMap, config: Config, *, paper_key: str | None = None
) -> Recipe:
    """A deterministic first draft the CodeAnalyst agent refines.

    Having a draft matters: the agent reviews and corrects concrete commands
    rather than inventing them, which is markedly harder to get subtly wrong.
    """
    env = resolve_env(repo, repo_map)
    setup = [
        RecipeStep(command=command, description="environment setup")
        for command in _setup_commands(repo, env.lockfile_path)
    ]
    smoke = _smoke_steps(repo_map)
    train = [
        RecipeStep(command=f"python {script}", description="training entrypoint (verify arguments)")
        for script in repo_map.train_scripts[:2]
    ]
    evaluate = [
        RecipeStep(
            command=f"python {script}", description="evaluation entrypoint (verify arguments)"
        )
        for script in repo_map.eval_scripts[:2]
    ]

    return Recipe(
        repo_id=repo.id,
        paper_key=paper_key or (repo.paper_keys[0] if repo.paper_keys else None),
        summary=f"Draft recipe for {repo.url} at {repo.commit or 'HEAD'}",
        setup=setup,
        smoke=smoke,
        train=train,
        evaluate=evaluate,
        config_knobs=_config_knobs(repo_map),
        datasets=_datasets(repo),
        reference_numbers=extract_reference_numbers(repo),
        gotchas=[*repo_map.notes, *pin_report(env)],
        env=env,
        confidence=0.25,  # a draft, not a verified recipe
        provenance=[Provenance(source=f"repo:{repo.url}", locator=repo.commit)],
    )


def _setup_commands(repo: CodeRepo, lockfile: str | None) -> list[str]:
    commands = []
    if repo.local_path:
        commands.append(f"cd {Path(repo.local_path).name}")
    if lockfile == "uv.lock":
        commands.append("uv sync --frozen")
    elif lockfile == "poetry.lock":
        commands.append("poetry install --no-root")
    else:
        commands.append("pip install -r requirements.txt")
    return commands


def _smoke_steps(repo_map: RepoMap) -> list[RecipeStep]:
    """Always produce a smoke path, even a weak one.

    An empty smoke list would let the scale ladder start at `small`, which is
    the exact ordering the system refuses to allow.
    """
    candidates = repo_map.train_scripts or repo_map.entrypoints
    if not candidates:
        return [
            RecipeStep(
                command="python -c 'import sys; print(sys.version)'",
                description="placeholder smoke: no entrypoint detected, verify manually",
                timeout_seconds=120,
            )
        ]
    script = candidates[0]
    return [
        RecipeStep(
            command=f"python {script} --help",
            description="confirm the entrypoint imports and exposes its arguments",
            timeout_seconds=180,
        ),
        RecipeStep(
            command=f"python {script} --max-steps 5 --batch-size 2",
            description="minimal training step; adjust flag names to this repo's parser",
            expected_artifacts=["metrics.json"],
            timeout_seconds=900,
        ),
    ]


def _config_knobs(repo_map: RepoMap) -> dict[str, str]:
    knobs: dict[str, str] = {}
    if repo_map.config_system:
        knobs["<config system>"] = repo_map.config_system
    for path in repo_map.config_files[:8]:
        knobs[path] = "configuration file; inspect for experimental variables"
    return knobs


def _datasets(repo: CodeRepo) -> list[str]:
    """Dataset names mentioned in the README, as candidates for the agent."""
    known = (
        "ImageNet",
        "CIFAR-10",
        "CIFAR-100",
        "MNIST",
        "COCO",
        "SQuAD",
        "GLUE",
        "SuperGLUE",
        "WikiText",
        "C4",
        "The Pile",
        "GSM8K",
        "MATH",
        "HumanEval",
        "MMLU",
        "LibriSpeech",
    )
    readme = repo.readme or ""
    return [name for name in known if name.lower() in readme.lower()]


def extract_reference_numbers(repo: CodeRepo, *, limit: int = 12) -> list[MetricReport]:
    """Pull claimed results out of the README, with provenance to that README.

    These are what "reproduces a published number" is checked against, so each
    one carries where it came from rather than floating free.
    """
    readme = repo.readme or ""
    if not readme:
        return []
    source = f"repo:{repo.url}#README"
    reports: list[MetricReport] = []

    for match in _INLINE_NUMBER.finditer(readme):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        reports.append(
            MetricReport(
                name=match.group(2).lower().replace(" ", "-"),
                value=value,
                provenance=Provenance(source=source, locator="README prose", quote=match.group(0)),
            )
        )
        if len(reports) >= limit:
            return reports

    for row in _TABLE_ROW.findall(readme):
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if len(cells) < 2 or set(cells[0]) <= set("-: "):
            continue
        for cell in cells[1:]:
            try:
                value = float(cell.replace("%", "").strip())
            except ValueError:
                continue
            reports.append(
                MetricReport(
                    name="reported",
                    value=value,
                    method=cells[0][:80],
                    provenance=Provenance(source=source, locator="README table", quote=row[:200]),
                )
            )
            break
        if len(reports) >= limit:
            break
    return reports


def is_runnable(recipe: Recipe) -> bool:
    """A recipe with no smoke path cannot start the ladder, so it is not runnable."""
    return bool(recipe.smoke) and all(step.command.strip() for step in recipe.smoke)


def recipe_gaps(recipe: Recipe) -> list[str]:
    """What still stands between this recipe and a trustworthy reproduction."""
    gaps: list[str] = []
    if not recipe.smoke:
        gaps.append("no smoke path: the cheap correctness check cannot run")
    if not recipe.reference_numbers:
        gaps.append("no reference numbers: there is nothing to check a reproduction against")
    if not recipe.datasets:
        gaps.append("no dataset identified: the data pipeline is unverified")
    if not recipe.env.packages and not recipe.env.container_image:
        gaps.append("environment is unspecified")
    if recipe.confidence < 0.5:
        gaps.append("recipe is a draft that no agent has verified against the repo")
    return gaps


__all__ = ["draft_recipe", "extract_reference_numbers", "is_runnable", "recipe_gaps"]
