"""Statistical analysis + plots; evaluates the pre-registered decision rule and
returns a Verdict (supported / refuted / inconclusive) with the reasoning."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import (
    Comparison,
    ExperimentSpec,
    Provenance,
    Result,
    RunRecord,
    VerdictStatus,
)
from .metrics import (
    compare,
    effect_interval,
    overlapping_error_bars,
    rules_out_effect,
    summarize_runs,
)


def analyze_runs(spec: ExperimentSpec, runs: Sequence[RunRecord]) -> Result:
    """Aggregate a spec's runs into summaries and paired comparisons.

    Failed runs are counted, never dropped quietly: an aggregate over the runs
    that happened to succeed is a different quantity from the one the spec
    pre-registered, and the difference has to be visible.
    """
    metrics = [m.name for m in spec.dependent_variables]
    successful = [r for r in runs if r.succeeded]
    summaries = summarize_runs(successful, metrics)

    comparisons: list[Comparison] = []
    for metric in metrics:
        for baseline in spec.baselines:
            if spec.treatment:
                comparison = compare(
                    successful, metric, baseline_arm=baseline, treatment_arm=spec.treatment
                )
                if comparison is not None:
                    comparisons.append(comparison)

    seeds = {r.seed for r in successful}
    notes: list[str] = []
    failed = len(runs) - len(successful)
    if failed:
        notes.append(f"{failed} of {len(runs)} runs failed and are excluded from the aggregate")
    missing = set(spec.seeds) - seeds
    if missing:
        notes.append(f"seeds with no successful run: {sorted(missing)}")
    for comparison in comparisons:
        if overlapping_error_bars(comparison):
            notes.append(
                f"{comparison.metric}: {comparison.treatment_arm} and {comparison.baseline_arm} "
                "error bars overlap at 1 sd"
            )

    return Result(
        spec_id=spec.id,
        spec_hash=spec.spec_hash,
        run_ids=[r.id for r in runs],
        summaries=summaries,
        comparisons=comparisons,
        n_seeds=len(seeds),
        failed_runs=failed,
        total_cost_usd=sum(r.cost_usd for r in runs),
        notes=notes,
        provenance=[Provenance(source=f"spec:{spec.id}", locator=spec.spec_hash)],
    )


def evaluate_decision_rule(spec: ExperimentSpec, result: Result) -> tuple[VerdictStatus, str]:
    """Apply the pre-registered rule literally.

    No judgment enters here. The rule was written before the compute was spent
    precisely so that this step could be mechanical -- which is what makes a
    negative result trustworthy.
    """
    rule = spec.decision_rule
    comparison = _comparison_for(result, rule.metric, spec)

    if comparison is None:
        summaries = [s for s in result.summaries if s.name.lower() == rule.metric.lower()]
        if not summaries:
            return VerdictStatus.INCONCLUSIVE, f"metric '{rule.metric}' was never recorded"
        if rule.relative_to_baseline:
            return (
                VerdictStatus.INCONCLUSIVE,
                f"no baseline comparison available for '{rule.metric}'; "
                "the rule is relative to a baseline",
            )
        summary = summaries[0]
        status = rule.evaluate(
            summary.mean, p_value=0.0 if rule.max_p_value else None, seeds=summary.n
        )
        return (
            status,
            f"absolute comparison: {rule.metric} mean {summary.mean:.5g} over {summary.n} seeds",
        )

    effect = comparison.effect
    metric_def = next(
        (m for m in spec.dependent_variables if m.name.lower() == rule.metric.lower()), None
    )
    if metric_def is not None and not metric_def.higher_is_better:
        # The rule is written in terms of improvement; for a loss-like metric an
        # improvement is a decrease, so flip the sign rather than the rule.
        effect = -effect

    status = rule.evaluate(effect, p_value=comparison.p_value, seeds=comparison.n_seeds)

    if status is VerdictStatus.INCONCLUSIVE and comparison.n_seeds >= rule.min_seeds:
        # "We could not detect an effect" and "we can rule out an effect this
        # large" are different findings, and the p-value gate alone collapses
        # them into one. A tight interval sitting entirely below the
        # pre-registered threshold refutes the prediction -- which is what makes
        # a clean null a result rather than a shrug.
        higher_is_better = metric_def.higher_is_better if metric_def is not None else True
        if rule.comparator in (">", ">=") and rules_out_effect(
            comparison, rule.threshold, higher_is_better=higher_is_better
        ):
            interval = effect_interval(comparison)
            bounds = f"[{interval[0]:+.5g}, {interval[1]:+.5g}]" if interval else "n/a"
            return (
                VerdictStatus.REFUTED,
                f"{comparison.treatment_arm} vs {comparison.baseline_arm} on {rule.metric}: "
                f"effect={effect:+.5g}, 95% CI {bounds} excludes the pre-registered "
                f"threshold of {rule.threshold} -- an effect that large is ruled out, "
                f"so the hypothesis is refuted rather than undecided",
            )

    detail = (
        f"{comparison.treatment_arm} vs {comparison.baseline_arm} on {rule.metric}: "
        f"effect={effect:+.5g} (rule needs {rule.comparator} {rule.threshold}), "
        f"p={comparison.p_value if comparison.p_value is None else round(comparison.p_value, 5)}, "
        f"seeds={comparison.n_seeds} (rule needs >= {rule.min_seeds})"
    )
    if status is VerdictStatus.SUPPORTED and overlapping_error_bars(comparison):
        detail += "; note: error bars overlap at 1 sd"
    return status, detail


def _comparison_for(result: Result, metric: str, spec: ExperimentSpec) -> Comparison | None:
    """The comparison the rule keys on: the one the treatment does *worst* against.

    Deliberately not "the first declared baseline". That would let the order the
    planner happened to list baselines in decide the verdict, and it would let an
    extra weak baseline manufacture a win. Taking the least favourable comparison
    is order-independent and is the only choice a skeptical reader would accept.
    """
    matches = [c for c in result.comparisons if c.metric.lower() == metric.lower()]
    if not matches:
        return None
    return min(matches, key=lambda c: _improvement(c, metric, spec))


def _improvement(comparison: Comparison, metric: str, spec: ExperimentSpec) -> float:
    """Signed improvement, so 'worst' means the same thing for loss-like metrics."""
    higher_is_better = next(
        (m.higher_is_better for m in spec.dependent_variables if m.name.lower() == metric.lower()),
        True,
    )
    return comparison.effect if higher_is_better else -comparison.effect


def seed_variance_note(result: Result, metric: str) -> str:
    """Plain-language statement of whether arms are distinguishable at all."""
    summaries = [s for s in result.summaries if s.name.lower() == metric.lower()]
    if not summaries:
        return f"no values recorded for {metric}"
    if len(summaries) == 1:
        s = summaries[0]
        return f"{s.arm}: mean {s.mean:.5g} ± {s.std:.5g} over {s.n} seeds"
    best = max(summaries, key=lambda s: s.mean)
    worst = min(summaries, key=lambda s: s.mean)
    gap, noise = best.mean - worst.mean, best.std + worst.std
    verdict = "larger than combined seed spread" if gap > noise else "within combined seed spread"
    return (
        f"{best.arm} ({best.mean:.5g} ± {best.std:.5g}) vs {worst.arm} "
        f"({worst.mean:.5g} ± {worst.std:.5g}): gap {gap:+.5g} is {verdict}"
    )


def is_single_seed(result: Result) -> bool:
    """Single-seed deltas are inconclusive by construction, not by discretion."""
    return result.n_seeds < 2


__all__ = [
    "analyze_runs",
    "evaluate_decision_rule",
    "is_single_seed",
    "seed_variance_note",
]
