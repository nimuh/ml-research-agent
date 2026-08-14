"""Hypothesis objects: claim, rationale, prediction, falsifier, prior confidence,
and the KB evidence it was derived from."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import (
    Claim,
    Hypothesis,
    Note,
    Provenance,
    ResearchBrief,
    VerdictStatus,
)

# Phrases that look like a falsifier but rule nothing out. A hypothesis whose
# falsifier is one of these is untestable, and it is cheaper to catch that here
# than after the compute is spent.
_VACUOUS_FALSIFIERS = (
    "if it does not work",
    "if the results are bad",
    "if it fails",
    "if the hypothesis is wrong",
    "if performance does not improve",
)


def from_claim(
    claim: Claim, brief: ResearchBrief, *, falsifier: str, prediction: str = ""
) -> Hypothesis:
    """Promote a KB claim into a testable hypothesis, carrying its evidence."""
    return Hypothesis(
        brief_id=brief.id,
        statement=claim.statement,
        rationale=f"Derived from KB claim {claim.id}",
        prediction=prediction or claim.statement,
        falsifier=falsifier,
        prior_confidence=claim.confidence,
        evidence_ids=[str(e.provenance) for e in claim.evidence],
        provenance=[Provenance(source=f"claim:{claim.id}", quote=claim.statement)],
    )


def from_gap(
    gap: str, brief: ResearchBrief, *, supporting_notes: Sequence[Note] = ()
) -> Hypothesis:
    """Turn a coverage gap into a hypothesis with an explicit falsifier."""
    return Hypothesis(
        brief_id=brief.id,
        statement=gap,
        rationale="Identified as an evidence gap during synthesis.",
        prediction=f"Filling this gap will show a measurable effect: {gap}",
        falsifier=f"No measurable effect is observed for: {gap}",
        prior_confidence=0.4,
        evidence_ids=[n.id for n in supporting_notes],
        provenance=[Provenance(source=f"note:{n.id}") for n in supporting_notes]
        or [Provenance(source=f"brief:{brief.id}")],
    )


def validate_hypothesis(hypothesis: Hypothesis) -> list[str]:
    """Return the reasons this hypothesis is not testable. Empty means it is."""
    problems: list[str] = []
    falsifier = hypothesis.falsifier.strip().lower()
    if not falsifier:
        problems.append(
            "no falsifier: the hypothesis cannot be refuted and is therefore not a hypothesis"
        )
    if any(vacuous in falsifier for vacuous in _VACUOUS_FALSIFIERS):
        problems.append(
            f"the falsifier is vacuous ('{hypothesis.falsifier}'); it rules nothing out"
        )
    if falsifier and falsifier == hypothesis.statement.strip().lower():
        problems.append("the falsifier restates the claim rather than contradicting it")
    if not hypothesis.prediction.strip():
        problems.append("no prediction: nothing to compare a result against")
    if not hypothesis.rationale.strip():
        problems.append("no rationale: there is no stated mechanism to test")
    return problems


def is_testable(hypothesis: Hypothesis) -> bool:
    return not validate_hypothesis(hypothesis)


def update_status(
    hypothesis: Hypothesis, status: VerdictStatus, *, max_rounds: int = 2
) -> Hypothesis:
    """Fold a verdict into a hypothesis, escalating rather than looping forever.

    Two inconclusive rounds on the same hypothesis means the design is not
    working; spending a third round of compute is the wrong response.
    """
    match status:
        case VerdictStatus.SUPPORTED:
            return hypothesis.model_copy(update={"status": "supported"})
        case VerdictStatus.REFUTED:
            return hypothesis.model_copy(update={"status": "refuted"})
        case VerdictStatus.INCONCLUSIVE:
            rounds = hypothesis.rounds + 1
            return hypothesis.model_copy(
                update={
                    "rounds": rounds,
                    "status": "escalate" if rounds >= max_rounds else "inconclusive",
                }
            )


def rank_by_diagnosticity(hypotheses: Sequence[Hypothesis]) -> list[Hypothesis]:
    """Most-informative first: a hypothesis at even odds discriminates the most.

    A prediction you are already 95% sure of teaches you almost nothing when it
    comes out the way you expected.
    """
    return sorted(hypotheses, key=lambda h: (abs(h.prior_confidence - 0.5), h.rounds))


__all__ = [
    "from_claim",
    "from_gap",
    "is_testable",
    "rank_by_diagnosticity",
    "update_status",
    "validate_hypothesis",
]
