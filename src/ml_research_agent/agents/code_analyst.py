"""CodeAnalyst: for each relevant repo, maps entrypoints, configs, data loaders,
training loops and eval scripts into a reusable `Recipe` we can actually run."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import (
    CodeRepo,
    EnvSpec,
    MetricReport,
    ModelTier,
    Provenance,
    Recipe,
    RecipeStep,
    RepoMap,
)
from .base import AgentContext, BaseAgent


class CodeAnalystInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: CodeRepo
    repo_map: RepoMap
    paper_key: str | None = None
    paper_title: str = ""
    target_numbers: list[MetricReport] = Field(
        default_factory=list, description="Published results we want to reproduce."
    )
    file_excerpts: dict[str, str] = Field(
        default_factory=dict, description="path -> excerpt, gathered by static analysis."
    )


class RecipeDraft(BaseModel):
    """The distillation that turns "there is a repo" into "we can run this"."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    setup: list[RecipeStep] = Field(default_factory=list)
    smoke: list[RecipeStep] = Field(
        min_length=1,
        description="Minutes-long correctness check. Required -- cheap before expensive.",
    )
    train: list[RecipeStep] = Field(default_factory=list)
    evaluate: list[RecipeStep] = Field(default_factory=list)
    config_knobs: dict[str, str] = Field(
        default_factory=dict,
        description="knob -> what it controls, for the experiment's variables.",
    )
    datasets: list[str] = Field(default_factory=list)
    reference_numbers: list[MetricReport] = Field(default_factory=list)
    gotchas: list[str] = Field(
        default_factory=list, description="What will silently break for someone running this."
    )
    python_version: str = "3.11"
    cuda_version: str | None = None
    packages: list[str] = Field(default_factory=list)
    runnable: bool = Field(description="False if this repo cannot realistically be run as-is.")
    blocking_issues: list[str] = Field(default_factory=list)


class CodeAnalyst(BaseAgent[CodeAnalystInput, RecipeDraft]):
    """Exists to prevent "there's a repo" that turns out to be unrunnable.

    Reads the repo statically. Nothing here executes repo code -- that only
    ever happens inside the sandbox, and only after the license gate.
    """

    name: ClassVar[str] = "code_analyst"
    description: ClassVar[str] = "Distills a fetched repo into a runnable Recipe."
    tier: ClassVar[ModelTier] = ModelTier.STANDARD
    prompt_name: ClassVar[str] = "code_analyst"
    tool_allowlist: ClassVar[tuple[str, ...]] = ("read_file", "list_dir")
    output_model: ClassVar[type[BaseModel]] = RecipeDraft

    def prompt_variables(self, payload: CodeAnalystInput, ctx: AgentContext) -> dict[str, object]:
        rmap = payload.repo_map
        excerpts = (
            "\n\n".join(
                f"--- {path} ---\n{text[:3000]}"
                for path, text in list(payload.file_excerpts.items())[:15]
            )
            or "none provided; use the read_file tool"
        )
        return {
            "repo": payload.repo.url,
            "commit": payload.repo.commit or "unpinned",
            "license": payload.repo.license.spdx_id or "unknown",
            "paper": payload.paper_title or payload.paper_key or "unlinked",
            "readme": (payload.repo.readme or "")[:6000],
            "entrypoints": "\n".join(f"- {e}" for e in rmap.entrypoints) or "- none detected",
            "configs": "\n".join(f"- {c}" for c in rmap.config_files) or "- none detected",
            "train_scripts": "\n".join(f"- {s}" for s in rmap.train_scripts) or "- none detected",
            "eval_scripts": "\n".join(f"- {s}" for s in rmap.eval_scripts) or "- none detected",
            "dependencies": ", ".join(rmap.dependencies[:40]) or "unknown",
            "hardware": "; ".join(rmap.hardware_assumptions) or "unstated",
            "target_numbers": "\n".join(
                f"- {m.name}={m.value} on {m.dataset or '?'}" for m in payload.target_numbers
            )
            or "- none specified",
            "excerpts": excerpts,
        }

    def validate_output(
        self, output: RecipeDraft, payload: CodeAnalystInput, ctx: AgentContext
    ) -> list[str]:
        violations: list[str] = []
        if output.runnable and not output.smoke:
            violations.append("claimed runnable but gave no smoke step to prove it cheaply")
        if not output.runnable and not output.blocking_issues:
            violations.append("claimed not runnable but named no blocking issue")
        for step in [*output.setup, *output.smoke, *output.train, *output.evaluate]:
            if not step.command.strip():
                violations.append("a recipe step has an empty command")
        return violations

    def to_recipe(self, draft: RecipeDraft, payload: CodeAnalystInput) -> Recipe:
        return Recipe(
            repo_id=payload.repo.id,
            paper_key=payload.paper_key,
            summary=draft.summary,
            setup=draft.setup,
            smoke=draft.smoke,
            train=draft.train,
            evaluate=draft.evaluate,
            config_knobs=draft.config_knobs,
            datasets=draft.datasets,
            reference_numbers=draft.reference_numbers,
            gotchas=[*draft.gotchas, *draft.blocking_issues],
            env=EnvSpec(
                python_version=draft.python_version,
                cuda_version=draft.cuda_version,
                packages=draft.packages,
            ),
            confidence=0.7 if draft.runnable else 0.2,
            provenance=[
                Provenance(
                    source=f"repo:{payload.repo.url}",
                    locator=payload.repo.commit,
                    quote=draft.summary[:280],
                )
            ],
        )


__all__ = ["CodeAnalyst", "CodeAnalystInput", "RecipeDraft"]
