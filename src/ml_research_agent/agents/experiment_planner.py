"""ExperimentPlanner: converts hypotheses into ExperimentSpecs -- variables,
controls, datasets, baselines, metrics, seeds, budget, and the decision rule
that says in advance what result would confirm or refute the hypothesis."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..config import ExperimentsConfig
from ..types import (
    DatasetRef,
    DecisionRule,
    EnvSpec,
    ExperimentSpec,
    Hypothesis,
    MetricDef,
    ModelTier,
    Provenance,
    Recipe,
    Scale,
    Variable,
)
from .base import AgentContext, BaseAgent


class PlannerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: Hypothesis
    baselines: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    recipes: list[Recipe] = Field(default_factory=list)
    budget_usd: float = 5.0
    default_seeds: int = 3
    scale: Scale = Scale.SMOKE


class SpecDraft(BaseModel):
    """A pre-registration. The decision rule is required, not encouraged."""

    model_config = ConfigDict(extra="forbid")

    title: str
    description: str = ""
    independent_variables: list[Variable] = Field(
        min_length=1, description="What you deliberately change."
    )
    dependent_variables: list[MetricDef] = Field(min_length=1, description="What you measure.")
    controls: list[str] = Field(
        min_length=1, description="What is held fixed so the comparison means something."
    )
    dataset: DatasetRef
    baselines: list[str] = Field(min_length=1)
    treatment: str = Field(description="The condition under test.")
    seeds: list[int] = Field(min_length=1)
    decision_rule: DecisionRule = Field(
        description="What result would REFUTE the hypothesis. Written before any compute is spent."
    )
    stopping_rule: str = Field(default="", description="When to stop early, win or lose.")
    estimated_cost_usd: float = 1.0
    max_runtime_minutes: int = 30
    recipe_id: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)


class ExperimentPlanner(BaseAgent[PlannerInput, SpecDraft]):
    """Exists to prevent uncontrolled experiments that can't answer anything."""

    name: ClassVar[str] = "experiment_planner"
    description: ClassVar[str] = "Turns a hypothesis into a pre-registered, controlled spec."
    tier: ClassVar[ModelTier] = ModelTier.DEEP
    prompt_name: ClassVar[str] = "experiment_planner"
    output_model: ClassVar[type[BaseModel]] = SpecDraft

    def prompt_variables(self, payload: PlannerInput, ctx: AgentContext) -> dict[str, object]:
        recipes = (
            "\n".join(
                f"- {r.id} ({r.repo_id}): {r.summary[:200]} | "
                f"knobs: {', '.join(list(r.config_knobs)[:8])}"
                for r in payload.recipes
            )
            or "- none; the experiment must be written from scratch"
        )
        return {
            "statement": payload.hypothesis.statement,
            "rationale": payload.hypothesis.rationale,
            "prediction": payload.hypothesis.prediction,
            "falsifier": payload.hypothesis.falsifier,
            "baselines": "\n".join(f"- {b}" for b in payload.baselines) or "- none identified",
            "benchmarks": "\n".join(f"- {b}" for b in payload.benchmarks) or "- none identified",
            "recipes": recipes,
            "scale": payload.scale.value,
            "budget_usd": payload.budget_usd,
            "default_seeds": payload.default_seeds,
        }

    def validate_output(
        self, output: SpecDraft, payload: PlannerInput, ctx: AgentContext
    ) -> list[str]:
        """Pre-registration enforced here -- the highest-value constraint in the system."""
        violations: list[str] = []
        rule = output.decision_rule

        if not rule.statement.strip():
            violations.append("the decision rule has no stated falsifier")
        measured = {m.name.lower() for m in output.dependent_variables}
        if rule.metric.lower() not in measured:
            violations.append(
                f"the decision rule keys on '{rule.metric}', which is not among the "
                "measured metrics"
            )
        if rule.min_seeds < 2:
            violations.append(
                "a decision rule accepting fewer than 2 seeds cannot distinguish signal from noise"
            )
        if len(output.seeds) < rule.min_seeds:
            violations.append(
                f"the spec runs {len(output.seeds)} seeds but the rule requires {rule.min_seeds}"
            )
        if rule.threshold == 0 and rule.min_effect_size == 0:
            violations.append(
                "a threshold of 0 with no minimum effect size makes the hypothesis unfalsifiable"
            )
        if not output.treatment.strip():
            violations.append("no treatment condition specified")
        if output.treatment in output.baselines:
            violations.append(
                "the treatment is listed as its own baseline; the comparison is vacuous"
            )
        if payload.budget_usd and output.estimated_cost_usd > payload.budget_usd:
            violations.append(
                f"estimated cost ${output.estimated_cost_usd:.2f} exceeds the "
                f"${payload.budget_usd:.2f} ceiling"
            )
        return violations

    def to_spec(
        self,
        draft: SpecDraft,
        payload: PlannerInput,
        *,
        config: ExperimentsConfig | None = None,
        env: EnvSpec | None = None,
    ) -> ExperimentSpec:
        """Assemble the real spec, refusing an unregistered one when policy says so."""
        if (
            config is not None
            and config.require_preregistered_decision_rule
            and not draft.decision_rule.statement.strip()
        ):
            from ..errors import AgentFailure

            raise AgentFailure(
                "spec rejected at DESIGN: no pre-registered decision rule",
                hypothesis=payload.hypothesis.id,
            )
        return ExperimentSpec(
            hypothesis_id=payload.hypothesis.id,
            title=draft.title,
            description=draft.description,
            scale=payload.scale,
            independent_variables=draft.independent_variables,
            dependent_variables=draft.dependent_variables,
            controls=draft.controls,
            dataset=draft.dataset,
            baselines=draft.baselines,
            treatment=draft.treatment,
            seeds=draft.seeds,
            decision_rule=draft.decision_rule,
            stopping_rule=draft.stopping_rule,
            budget_usd=min(draft.estimated_cost_usd, payload.budget_usd) or payload.budget_usd,
            max_runtime_minutes=draft.max_runtime_minutes,
            recipe_id=draft.recipe_id,
            env=env or EnvSpec(),
            parameters=dict(draft.parameters),
            provenance=[
                Provenance(
                    source=f"agent:{self.name}@v1",
                    locator=payload.hypothesis.id,
                    quote=draft.decision_rule.statement,
                )
            ],
        )


__all__ = ["ExperimentPlanner", "PlannerInput", "SpecDraft"]
