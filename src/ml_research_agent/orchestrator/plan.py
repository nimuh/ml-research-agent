"""ResearchPlan: decomposition of an idea into questions, hypotheses and a
prioritized experiment ladder (cheap sanity checks -> full comparisons).
Supports re-planning when results invalidate assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ExperimentsConfig
from ..types import (
    ExperimentSpec,
    FollowUp,
    Hypothesis,
    Provenance,
    Question,
    ResearchBrief,
    Scale,
    Verdict,
    VerdictStatus,
)


@dataclass
class LadderRung:
    """One step of the scale ladder for one hypothesis."""

    hypothesis_id: str
    scale: Scale
    spec_id: str | None = None
    status: str = "pending"
    estimated_usd: float = 1.0


@dataclass
class ResearchPlan:
    """Questions -> hypotheses -> an ordered experiment ladder.

    Ordering is cheapest-and-most-diagnostic first, which is the same reason
    the scale ladder exists: information per dollar, not ambition per run.
    """

    brief_id: str
    questions: list[Question] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    ladder: list[LadderRung] = field(default_factory=list)
    round: int = 0

    def next_rung(self) -> LadderRung | None:
        return next((r for r in self.ladder if r.status == "pending"), None)

    def rungs_for(self, hypothesis_id: str) -> list[LadderRung]:
        return [r for r in self.ladder if r.hypothesis_id == hypothesis_id]

    def mark(self, rung: LadderRung, status: str) -> None:
        rung.status = status


def questions_from_brief(brief: ResearchBrief) -> list[Question]:
    """One empirical question per falsifiable claim, plus the reproduction question.

    Deterministic on purpose: the decomposition should be inspectable and the
    same brief should always yield the same starting question set.
    """
    questions = [
        Question(
            brief_id=brief.id,
            text=f"Does the evidence support: {claim.statement}?",
            kind="empirical",
            priority=claim.confidence,
            provenance=[Provenance(source=f"brief:{brief.id}", quote=claim.statement)],
        )
        for claim in brief.claims
    ]
    questions.append(
        Question(
            brief_id=brief.id,
            text="Can the strongest published baseline be reproduced at smoke scale?",
            kind="reproduction",
            priority=0.9,
            provenance=[Provenance(source=f"brief:{brief.id}")],
        )
    )
    return questions


def hypotheses_from_brief(
    brief: ResearchBrief, *, questions: list[Question] | None = None
) -> list[Hypothesis]:
    """Seed hypotheses straight from the brief's claims, carrying the falsifier through.

    The falsifier is copied, never re-derived -- it was agreed at FRAME, and a
    hypothesis whose falsifier drifts is no longer testing the same idea.
    """
    by_claim = {q.text: q for q in (questions or [])}
    out: list[Hypothesis] = []
    for claim in brief.claims:
        question = by_claim.get(f"Does the evidence support: {claim.statement}?")
        out.append(
            Hypothesis(
                brief_id=brief.id,
                question_id=question.id if question else None,
                statement=claim.statement,
                rationale=brief.problem_statement,
                prediction=claim.statement,
                falsifier=claim.falsifier,
                prior_confidence=claim.confidence,
                provenance=[Provenance(source=f"brief:{brief.id}", quote=claim.statement)],
            )
        )
    return out


def build_plan(brief: ResearchBrief, config: ExperimentsConfig) -> ResearchPlan:
    questions = questions_from_brief(brief)
    hypotheses = hypotheses_from_brief(brief, questions=questions)
    plan = ResearchPlan(brief_id=brief.id, questions=questions, hypotheses=hypotheses)
    plan.ladder = build_ladder(hypotheses, config)
    return plan


def build_ladder(hypotheses: list[Hypothesis], config: ExperimentsConfig) -> list[LadderRung]:
    """Every hypothesis starts at smoke. The ladder is enforced here, not by agents.

    Interleaved by scale rather than grouped by hypothesis: one smoke failure
    should stop the whole round cheaply, before anything runs at `small`.
    """
    rungs: list[LadderRung] = []
    for scale in config.scale_ladder:
        for hypothesis in sorted(hypotheses, key=lambda h: -h.prior_confidence):
            rungs.append(
                LadderRung(
                    hypothesis_id=hypothesis.id,
                    scale=scale,
                    estimated_usd={Scale.SMOKE: 0.5, Scale.SMALL: 3.0, Scale.MAIN: 20.0}.get(
                        scale, 1.0
                    ),
                )
            )
    return rungs


def replan(
    plan: ResearchPlan,
    verdict: Verdict,
    followups: list[FollowUp],
    config: ExperimentsConfig,
) -> ResearchPlan:
    """Fold a verdict back into the plan.

    ``supported`` climbs the ladder, ``refuted`` closes the hypothesis out, and
    ``inconclusive`` schedules the best follow-up -- until the round cap, at
    which point burning more compute on the same question is the wrong answer.
    """
    plan.round += 1
    hypothesis = next((h for h in plan.hypotheses if h.id == verdict.hypothesis_id), None)
    if hypothesis is None:
        return plan

    match verdict.status:
        case VerdictStatus.SUPPORTED:
            hypothesis.status = "supported"
            _promote(plan, hypothesis.id)
        case VerdictStatus.REFUTED:
            hypothesis.status = "refuted"
            for rung in plan.rungs_for(hypothesis.id):
                if rung.status == "pending":
                    rung.status = "cancelled"
        case VerdictStatus.INCONCLUSIVE:
            hypothesis.status = "inconclusive"
            hypothesis.rounds += 1
            if hypothesis.rounds >= config.max_inconclusive_rounds:
                hypothesis.status = "escalate"
                for rung in plan.rungs_for(hypothesis.id):
                    if rung.status == "pending":
                        rung.status = "escalated"
            else:
                best = max(followups, key=lambda f: f.gain_per_dollar, default=None)
                if best is not None:
                    plan.ladder.insert(
                        0,
                        LadderRung(
                            hypothesis_id=hypothesis.id,
                            scale=Scale.SMOKE,
                            estimated_usd=best.estimated_cost_usd,
                        ),
                    )
    return plan


def _promote(plan: ResearchPlan, hypothesis_id: str) -> None:
    """Unlock the next rung up for a hypothesis whose current scale passed."""
    for rung in plan.rungs_for(hypothesis_id):
        if rung.status == "pending":
            rung.status = "ready"
            return


def order_specs(specs: list[ExperimentSpec]) -> list[ExperimentSpec]:
    """Cheapest and most diagnostic first: smoke before small before main."""
    order = {Scale.SMOKE: 0, Scale.SMALL: 1, Scale.MAIN: 2}
    return sorted(specs, key=lambda s: (order.get(s.scale, 9), s.budget_usd, s.title))


__all__ = [
    "LadderRung",
    "ResearchPlan",
    "build_ladder",
    "build_plan",
    "hypotheses_from_brief",
    "order_specs",
    "questions_from_brief",
    "replan",
]
