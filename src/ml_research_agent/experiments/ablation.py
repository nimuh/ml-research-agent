"""Follow-up generation: given a verdict, propose the ablations or controls that
would most reduce remaining uncertainty, ranked by information gain per dollar."""

from __future__ import annotations

from ..config import Config
from ..types import (
    ExperimentSpec,
    FollowUp,
    Result,
    Verdict,
    VerdictStatus,
)
from .metrics import overlapping_error_bars
from .spec import SCALE_COST_FACTOR, next_scale


def propose_followups(
    spec: ExperimentSpec, result: Result, verdict: Verdict, config: Config
) -> list[FollowUp]:
    """Rank the next experiments by expected information gain per dollar.

    Deterministic on purpose: what to run next after an inconclusive result is
    mostly a function of *why* it was inconclusive, and that is knowable from
    the numbers. Cheap diagnostics outrank expensive scale-ups.
    """
    followups: list[FollowUp] = []
    base_cost = spec.budget_usd or 1.0
    hypothesis_id = spec.hypothesis_id

    match verdict.status:
        case VerdictStatus.INCONCLUSIVE:
            followups += _inconclusive_followups(spec, result, base_cost, hypothesis_id, config)
        case VerdictStatus.SUPPORTED:
            followups += _supported_followups(spec, result, base_cost, hypothesis_id, config)
        case VerdictStatus.REFUTED:
            followups += _refuted_followups(spec, result, base_cost, hypothesis_id)

    for threat in verdict.threats_to_validity[:3]:
        followups.append(
            FollowUp(
                title=f"Control for: {threat[:80]}",
                rationale=f"The verdict names this as a threat to validity: {threat}",
                kind="control",
                expected_information_gain=0.55,
                estimated_cost_usd=base_cost,
                hypothesis_id=hypothesis_id,
            )
        )

    return sorted(followups, key=lambda f: -f.gain_per_dollar)


def _inconclusive_followups(
    spec: ExperimentSpec, result: Result, base_cost: float, hypothesis_id: str, config: Config
) -> list[FollowUp]:
    """Diagnose the reason, then propose the cheapest thing that would resolve it."""
    out: list[FollowUp] = []
    rule = spec.decision_rule

    if result.n_seeds < rule.min_seeds:
        # By far the cheapest fix: the experiment was never given enough seeds
        # to distinguish the effect it was designed to detect.
        out.append(
            FollowUp(
                title=f"Re-run with {rule.min_seeds + 2} seeds",
                rationale=(
                    f"Only {result.n_seeds} seeds completed; the decision rule requires "
                    f"{rule.min_seeds}. More seeds is the cheapest way to make this decidable."
                ),
                kind="replication",
                expected_information_gain=0.9,
                estimated_cost_usd=base_cost * (rule.min_seeds + 2) / max(len(spec.seeds), 1),
                hypothesis_id=hypothesis_id,
            )
        )

    if result.failed_runs:
        out.append(
            FollowUp(
                title=f"Repair and re-run the {result.failed_runs} failed runs",
                rationale="The aggregate is over the runs that happened to succeed, which is a "
                "different quantity from the one the spec pre-registered.",
                kind="replication",
                expected_information_gain=0.75,
                estimated_cost_usd=base_cost * result.failed_runs / max(len(spec.seeds), 1),
                hypothesis_id=hypothesis_id,
            )
        )

    noisy = [c for c in result.comparisons if overlapping_error_bars(c)]
    if noisy:
        out.append(
            FollowUp(
                title="Reduce variance before scaling up",
                rationale=(
                    f"{len(noisy)} comparison(s) have overlapping error bars at 1 sd. Fixing the "
                    "variance source (data order, init, eval subsampling) costs less than "
                    "out-running it with a bigger experiment."
                ),
                kind="control",
                expected_information_gain=0.7,
                estimated_cost_usd=base_cost * 0.5,
                hypothesis_id=hypothesis_id,
            )
        )

    if not result.comparisons:
        out.append(
            FollowUp(
                title="Add the missing baseline arm",
                rationale="No treatment-vs-baseline comparison was computable, so the rule could "
                "never have fired.",
                kind="new_baseline",
                expected_information_gain=0.85,
                estimated_cost_usd=base_cost,
                hypothesis_id=hypothesis_id,
            )
        )
    return out


def _supported_followups(
    spec: ExperimentSpec, result: Result, base_cost: float, hypothesis_id: str, config: Config
) -> list[FollowUp]:
    """A supported result at smoke scale is a reason to climb, not to conclude."""
    out: list[FollowUp] = []
    target = next_scale(spec.scale, config.experiments)
    if target is not None:
        ratio = SCALE_COST_FACTOR[target] / SCALE_COST_FACTOR[spec.scale]
        out.append(
            FollowUp(
                title=f"Repeat at {target.value} scale",
                rationale=f"The effect held at {spec.scale.value}; the ladder's next rung tests "
                "whether it survives a more honest setting.",
                kind="scale_up",
                expected_information_gain=0.8,
                estimated_cost_usd=base_cost * ratio,
                hypothesis_id=hypothesis_id,
            )
        )
    for variable in spec.independent_variables[:2]:
        out.append(
            FollowUp(
                title=f"Ablate {variable.name}",
                rationale=f"Isolate how much of the effect {variable.name} is responsible for.",
                kind="ablation",
                expected_information_gain=0.65,
                estimated_cost_usd=base_cost,
                hypothesis_id=hypothesis_id,
            )
        )
    out.append(
        FollowUp(
            title="Strengthen the baseline and re-test",
            rationale="A supported result is most often an undertuned baseline. Give the baseline "
            "the same tuning budget as the treatment before believing the effect.",
            kind="new_baseline",
            expected_information_gain=0.75,
            estimated_cost_usd=base_cost,
            hypothesis_id=hypothesis_id,
        )
    )
    return out


def _refuted_followups(
    spec: ExperimentSpec, result: Result, base_cost: float, hypothesis_id: str
) -> list[FollowUp]:
    """A refutation is a finding; the follow-ups check it was a fair test."""
    return [
        FollowUp(
            title="Verify the implementation reproduces a known number",
            rationale="Before accepting the refutation, confirm the code reproduces a published "
            "reference result -- a refutation from broken code is not a refutation.",
            kind="replication",
            expected_information_gain=0.85,
            estimated_cost_usd=base_cost * 0.5,
            hypothesis_id=hypothesis_id,
        ),
        FollowUp(
            title="Test the boundary condition where the effect might still hold",
            rationale="The hypothesis failed under these conditions. Identify the narrower regime, "
            "if any, where the mechanism would still predict an effect.",
            kind="ablation",
            expected_information_gain=0.5,
            estimated_cost_usd=base_cost,
            hypothesis_id=hypothesis_id,
        ),
    ]


def best_followup(followups: list[FollowUp], *, budget_usd: float) -> FollowUp | None:
    """The highest gain-per-dollar option that fits inside the remaining budget."""
    affordable = [f for f in followups if f.estimated_cost_usd <= budget_usd]
    return max(affordable, key=lambda f: f.gain_per_dollar, default=None)


__all__ = ["best_followup", "propose_followups"]
