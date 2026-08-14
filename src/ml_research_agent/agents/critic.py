"""Critic (red team): adversarially attacks every stage -- leaky evals, unfair
baselines, cherry-picked seeds, unsupported claims. Gates phase exits."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import CritiqueReport, Finding, ModelTier, Phase
from .base import AgentContext, BaseAgent

# What the Critic is told to hunt for at each gate. Naming the failure modes
# beats "find problems" -- a generic prompt returns generic approval.
PHASE_ATTACKS: dict[Phase, tuple[str, ...]] = {
    Phase.CURATE: (
        "notes that assert numbers no cited passage contains",
        "papers included for topical adjacency rather than relevance to the brief",
        "seminal prior work that is conspicuously absent from the KB",
        "extraction that flattened a paper's stated limitations into nothing",
    ),
    Phase.SYNTHESIZE: (
        "a novelty claim that the surveyed literature already settles",
        "'contested' framing where the papers actually agree, or vice versa",
        "a baseline set that omits the strongest published method",
        "gaps asserted without evidence the gap is real rather than unsearched",
    ),
    Phase.DESIGN: (
        "a decision rule that cannot be refuted by any achievable outcome",
        "train/test leakage in the proposed split or preprocessing",
        "an unfair baseline: undertuned, undertrained, or given fewer resources",
        "a confound the controls do not hold fixed",
        "seeds too few to distinguish the predicted effect from noise",
        "a metric that does not measure the construct the hypothesis is about",
    ),
    Phase.ANALYZE: (
        "a conclusion drawn from single-seed or overlapping-error-bar differences",
        "a decision rule quietly reinterpreted after seeing the result",
        "cherry-picked seeds, steps, or checkpoints",
        "an effect explained by a confound rather than the treatment",
        "failed or missing runs silently excluded from the aggregate",
    ),
}

SEVERITY_WEIGHT: dict[str, float] = {"blocker": 1.0, "major": 0.4, "minor": 0.15, "note": 0.0}


class CritiqueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: Phase
    target_id: str
    artifact: dict[str, Any] = Field(description="The artifact under review, serialized.")
    context: dict[str, Any] = Field(
        default_factory=dict, description="Brief, KB stats, prior runs."
    )


class CritiqueDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    summary: str = Field(description="The single most important objection, stated plainly.")
    would_you_stake_your_reputation: bool = Field(
        description="Would you defend this artifact's conclusions publicly, as-is?"
    )


class Critic(BaseAgent[CritiqueInput, CritiqueDraft]):
    """Prompted to refute, never to approve.

    The gate score is computed from the findings in code -- if the Critic could
    also grade itself, a fluent critique would be able to wave itself through.
    """

    name: ClassVar[str] = "critic"
    description: ClassVar[str] = "Adversarial reviewer that gates every phase exit."
    tier: ClassVar[ModelTier] = ModelTier.DEEP
    prompt_name: ClassVar[str] = "critic"
    output_model: ClassVar[type[BaseModel]] = CritiqueDraft

    def prompt_variables(self, payload: CritiqueInput, ctx: AgentContext) -> dict[str, object]:
        import json

        attacks = PHASE_ATTACKS.get(
            payload.phase, ("any defect that would invalidate the conclusion",)
        )
        return {
            "phase": payload.phase.value,
            "attacks": "\n".join(f"- {a}" for a in attacks),
            "artifact": json.dumps(payload.artifact, indent=2, default=str)[:24000],
            "context": json.dumps(payload.context, indent=2, default=str)[:8000],
        }

    def validate_output(
        self, output: CritiqueDraft, payload: CritiqueInput, ctx: AgentContext
    ) -> list[str]:
        violations: list[str] = []
        if not output.summary.strip():
            violations.append("critique has no summary")
        for i, finding in enumerate(output.findings):
            if not finding.statement.strip():
                violations.append(f"finding {i + 1} states nothing")
            if finding.severity in ("blocker", "major") and not finding.suggested_fix.strip():
                violations.append(f"finding {i + 1} is {finding.severity} but proposes no fix")
        if not output.findings and output.would_you_stake_your_reputation is False:
            violations.append(
                "you declined to stand behind the artifact but listed no finding explaining why"
            )
        return violations

    def to_report(self, draft: CritiqueDraft, payload: CritiqueInput) -> CritiqueReport:
        """Score the findings in code: severity drives the gate, not self-assessment.

        ``passed`` is computed by :func:`gate_passed` rather than restated here,
        so there is exactly one definition of what passing means. An earlier
        version had three subtly different ones and the phase machine happened
        to use the weakest.
        """
        penalty = sum(SEVERITY_WEIGHT.get(f.severity, 0.0) for f in draft.findings)
        report = CritiqueReport(
            phase=payload.phase,
            target_id=payload.target_id,
            findings=draft.findings,
            score=max(0.0, min(1.0, 1.0 - penalty)),
            passed=False,
            summary=draft.summary,
        )
        return report.model_copy(update={"passed": gate_passed(report)})


#: The score a phase artifact must clear to proceed. Accumulated majors sink a
#: critique below it even with no single blocking finding.
GATE_MIN_SCORE = 0.5


def gate_passed(report: CritiqueReport, *, min_score: float = GATE_MIN_SCORE) -> bool:
    """Whether a phase may exit: no blockers, and a score above the floor.

    The Critic's own ``would_you_stake_your_reputation`` is deliberately not a
    term here. It is a self-assessment, and ``validate_output`` already requires
    that declining to stand behind an artifact be backed by a finding -- which
    then moves the score. Counting it twice would let a model veto a phase on
    sentiment alone.
    """
    return not report.blockers and report.score >= min_score


__all__ = [
    "GATE_MIN_SCORE",
    "PHASE_ATTACKS",
    "SEVERITY_WEIGHT",
    "Critic",
    "CritiqueDraft",
    "CritiqueInput",
    "gate_passed",
]
