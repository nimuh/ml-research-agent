"""Analyst: aggregates run records into results -- effect sizes, seed variance,
significance, plots -- and states whether the pre-registered decision rule fired."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import (
    Comparison,
    ExperimentSpec,
    ModelTier,
    Provenance,
    Result,
    RunRecord,
    Verdict,
    VerdictStatus,
)
from .base import AgentContext, BaseAgent


def _rel(value: float | None) -> str:
    """Relative effect, rendered so an absent value reads as absent."""
    return "n/a" if value is None else f"{value:.4f}"


class AnalystInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ExperimentSpec
    runs: list[RunRecord] = Field(min_length=1)
    result: Result
    rule_status: VerdictStatus = Field(
        description="What the decision rule returned when evaluated arithmetically."
    )
    rule_detail: str = ""


class VerdictDraft(BaseModel):
    """Interpretation only. The status itself is computed, not argued for.

    The model may explain and caveat the outcome; it may not overturn the
    pre-registered rule, which is the whole point of pre-registering it.
    """

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="Why the numbers came out this way.")
    seed_variance_note: str = Field(description="What the spread across seeds implies.")
    threats_to_validity: list[str] = Field(
        min_length=1, description="Reasons this result might not mean what it appears to."
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    disagrees_with_rule: bool = Field(
        default=False, description="Set only if the rule was misapplied -- explain in reasoning."
    )


class Analyst(BaseAgent[AnalystInput, VerdictDraft]):
    """Exists to prevent reading noise across 1 seed as a result."""

    name: ClassVar[str] = "analyst"
    description: ClassVar[str] = "Interprets aggregated runs against the pre-registered rule."
    tier: ClassVar[ModelTier] = ModelTier.DEEP
    prompt_name: ClassVar[str] = "analyst"
    output_model: ClassVar[type[BaseModel]] = VerdictDraft

    def prompt_variables(self, payload: AnalystInput, ctx: AgentContext) -> dict[str, object]:
        summaries = (
            "\n".join(
                f"- {s.arm} / {s.name}: mean={s.mean:.5g} sd={s.std:.5g} n={s.n} "
                f"range=[{s.min:.5g}, {s.max:.5g}]"
                for s in payload.result.summaries
            )
            or "- no summaries computed"
        )
        comparisons = (
            "\n".join(
                f"- {c.metric}: {c.treatment_arm} vs {c.baseline_arm} -> effect={c.effect:.5g} "
                f"(rel={_rel(c.relative_effect)}, "
                f"p={c.p_value}, d={c.effect_size}, n={c.n_seeds})"
                for c in payload.result.comparisons
            )
            or "- no comparisons computed"
        )
        failures = (
            "\n".join(
                f"- {r.arm} seed={r.seed}: {r.status.value} {r.failure.value if r.failure else ''} "
                f"{(r.failure_detail or '')[:160]}"
                for r in payload.runs
                if not r.succeeded
            )
            or "- none"
        )
        return {
            "title": payload.spec.title,
            "decision_rule": payload.spec.decision_rule.statement,
            "rule_status": payload.rule_status.value,
            "rule_detail": payload.rule_detail or "n/a",
            "summaries": summaries,
            "comparisons": comparisons,
            "seeds": payload.spec.seeds,
            "completed_runs": sum(1 for r in payload.runs if r.succeeded),
            "total_runs": len(payload.runs),
            "failures": failures,
        }

    def validate_output(
        self, output: VerdictDraft, payload: AnalystInput, ctx: AgentContext
    ) -> list[str]:
        violations: list[str] = []
        if not output.threats_to_validity:
            violations.append("no threats to validity listed; every result has at least one")
        if output.disagrees_with_rule and len(output.reasoning) < 80:
            violations.append(
                "you disagreed with the pre-registered rule without a substantive explanation"
            )
        if not output.seed_variance_note.strip():
            violations.append("no seed-variance note; spread is never optional")
        return violations

    def to_verdict(self, draft: VerdictDraft, payload: AnalystInput) -> Verdict:
        """Status comes from the rule. Prose comes from the model. Not the reverse."""
        return Verdict(
            hypothesis_id=payload.spec.hypothesis_id,
            result_id=payload.result.id,
            status=payload.rule_status,
            reasoning=draft.reasoning,
            decision_rule_statement=payload.spec.decision_rule.statement,
            rule_fired=payload.rule_status is not VerdictStatus.INCONCLUSIVE,
            seed_variance_note=draft.seed_variance_note,
            threats_to_validity=draft.threats_to_validity,
            confidence=draft.confidence,
            provenance=[
                Provenance(source=f"result:{payload.result.id}", locator=payload.spec.spec_hash),
                *(Provenance(source=f"run:{r.id}") for r in payload.runs if r.succeeded),
            ],
        )


def strongest_comparison(result: Result, metric: str) -> Comparison | None:
    """The comparison the decision rule keys on, if the Analyst computed one."""
    matches = [c for c in result.comparisons if c.metric.lower() == metric.lower()]
    return max(matches, key=lambda c: abs(c.effect)) if matches else None


__all__ = ["Analyst", "AnalystInput", "VerdictDraft", "strongest_comparison"]
