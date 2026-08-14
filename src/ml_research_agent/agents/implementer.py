"""Implementer: writes the experiment code in an isolated workspace, preferring
adaptation of a vetted Recipe over greenfield code. Owns smoke tests."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import ExperimentSpec, ModelTier, Recipe
from .base import AgentContext, BaseAgent


class ImplementerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ExperimentSpec
    recipe: Recipe | None = None
    workspace_path: str
    existing_files: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class GeneratedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Workspace-relative. Absolute paths and '..' are rejected.")
    content: str
    purpose: str = ""


class ImplementationDraft(BaseModel):
    """Code plus, explicitly, what was changed relative to the reference.

    Silent divergence from a reference implementation is the failure mode this
    agent exists to prevent, so divergences are a required output field.
    """

    model_config = ConfigDict(extra="forbid")

    files: list[GeneratedFile] = Field(min_length=1)
    entrypoint: str = Field(description="Command that runs one arm at one seed.")
    smoke_command: str = Field(
        description="Command that finishes in minutes and proves correctness."
    )
    metrics_path: str = Field(
        default="metrics.json",
        description="Where the run writes its metrics, for tracking to parse.",
    )
    adapted_from_recipe: bool = False
    divergences: list[str] = Field(
        default_factory=list, description="Every intentional difference from the reference."
    )
    assumptions: list[str] = Field(default_factory=list)


class Implementer(BaseAgent[ImplementerInput, ImplementationDraft]):
    """Exists to prevent greenfield code with silent divergences from the reference."""

    name: ClassVar[str] = "implementer"
    description: ClassVar[str] = (
        "Builds the experiment workspace, adapting a Recipe where possible."
    )
    tier: ClassVar[ModelTier] = ModelTier.STANDARD
    prompt_name: ClassVar[str] = "implementer"
    tool_allowlist: ClassVar[tuple[str, ...]] = ("read_file", "list_dir", "write_file")
    max_tool_turns: ClassVar[int] = 10
    output_model: ClassVar[type[BaseModel]] = ImplementationDraft

    def prompt_variables(self, payload: ImplementerInput, ctx: AgentContext) -> dict[str, object]:
        spec = payload.spec
        recipe = payload.recipe
        return {
            "title": spec.title,
            "description": spec.description,
            "scale": spec.scale.value,
            "independent_variables": "\n".join(
                f"- {v.name}: {v.values} ({v.description})" for v in spec.independent_variables
            ),
            "metrics": "\n".join(
                f"- {m.name} ({'higher' if m.higher_is_better else 'lower'} is better)"
                for m in spec.dependent_variables
            ),
            "controls": "\n".join(f"- {c}" for c in spec.controls),
            "dataset": f"{spec.dataset.name} [{spec.dataset.split}]",
            "arms": ", ".join(spec.arms),
            "seeds": spec.seeds,
            "decision_rule": spec.decision_rule.statement,
            "max_runtime_minutes": spec.max_runtime_minutes,
            "workspace": payload.workspace_path,
            "existing_files": "\n".join(f"- {f}" for f in payload.existing_files[:80])
            or "- empty workspace",
            "recipe": (
                f"{recipe.summary}\n"
                f"setup: {[s.command for s in recipe.setup]}\n"
                f"smoke: {[s.command for s in recipe.smoke]}\n"
                f"train: {[s.command for s in recipe.train]}\n"
                f"eval: {[s.command for s in recipe.evaluate]}\n"
                f"knobs: {recipe.config_knobs}\n"
                f"gotchas: {recipe.gotchas}"
                if recipe
                else "no vetted recipe available; write minimal, honest code from scratch"
            ),
            "notes": "\n".join(f"- {n}" for n in payload.notes) or "- none",
        }

    def validate_output(
        self, output: ImplementationDraft, payload: ImplementerInput, ctx: AgentContext
    ) -> list[str]:
        violations: list[str] = []
        for f in output.files:
            if f.path.startswith("/") or ".." in f.path.split("/"):
                violations.append(f"file path escapes the workspace: {f.path}")
            if not f.content.strip():
                violations.append(f"file {f.path} is empty")
        if not output.smoke_command.strip():
            violations.append("no smoke command; the scale ladder cannot start")
        if payload.recipe is not None and output.adapted_from_recipe and not output.divergences:
            violations.append(
                "adapted from a recipe but declared no divergences -- silent divergence is the "
                "exact failure this step exists to prevent; state them, or say "
                "explicitly that there are none"
            )
        arms = set(payload.spec.arms)
        joined = " ".join(f.content for f in output.files)
        missing = [a for a in arms if a and a not in joined and a not in output.entrypoint]
        if missing:
            violations.append(f"no code path implements these arms: {missing}")
        return violations


__all__ = ["GeneratedFile", "ImplementationDraft", "Implementer", "ImplementerInput"]
