"""Metric definitions and aggregation across seeds/replicates: mean, spread,
confidence intervals, paired comparisons against the baseline."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from ..types import Comparison, MetricSummary, RunRecord

# Two-sided t critical values for small samples. Seeds default to 3, so the
# normal approximation would systematically understate the interval exactly
# where honesty about spread matters most.
_T_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}


def t_critical(df: int) -> float:
    if df <= 0:
        return float("nan")
    for key in sorted(_T_95):
        if df <= key:
            return _T_95[key]
    return 1.96


def summarize(values: Sequence[float], *, name: str, arm: str) -> MetricSummary:
    """Mean plus spread. A mean reported without spread is not a result."""
    data = [float(v) for v in values]
    n = len(data)
    if n == 0:
        return MetricSummary(
            name=name,
            arm=arm,
            n=0,
            mean=float("nan"),
            std=float("nan"),
            min=float("nan"),
            max=float("nan"),
            values=[],
        )
    mean = statistics.fmean(data)
    std = statistics.stdev(data) if n > 1 else 0.0
    ci_low = ci_high = None
    if n > 1:
        margin = t_critical(n - 1) * std / math.sqrt(n)
        ci_low, ci_high = mean - margin, mean + margin
    return MetricSummary(
        name=name,
        arm=arm,
        n=n,
        mean=mean,
        std=std,
        min=min(data),
        max=max(data),
        ci_low=ci_low,
        ci_high=ci_high,
        values=data,
    )


def collect(runs: Sequence[RunRecord], metric: str) -> dict[str, dict[int, float]]:
    """``arm -> seed -> final value``, so comparisons can be paired by seed."""
    out: dict[str, dict[int, float]] = {}
    for run in runs:
        if not run.succeeded:
            continue
        value = run.metric(metric)
        if value is None:
            continue
        out.setdefault(run.arm, {})[run.seed] = float(value)
    return out


def summarize_runs(runs: Sequence[RunRecord], metrics: Sequence[str]) -> list[MetricSummary]:
    summaries: list[MetricSummary] = []
    for metric in metrics:
        for arm, by_seed in sorted(collect(runs, metric).items()):
            summaries.append(summarize(list(by_seed.values()), name=metric, arm=arm))
    return summaries


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float] | None:
    """Welch's t (unequal variances). Returns ``(t, p)`` or None when undefined."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = statistics.variance(a), statistics.variance(b)
    denom = va / na + vb / nb
    if denom <= 0:
        return None
    t = (statistics.fmean(a) - statistics.fmean(b)) / math.sqrt(denom)
    df = denom**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, _t_sf(abs(t), df) * 2


def paired_t_test(pairs: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """Paired test over matched seeds -- the right test when seeds are shared.

    Pairing removes seed-to-seed variance from the comparison, which is usually
    the dominant term at n=3.
    """
    diffs = [x - y for x, y in pairs]
    n = len(diffs)
    if n < 2:
        return None
    sd = statistics.stdev(diffs)
    if sd == 0:
        return (
            float("inf") if statistics.fmean(diffs) != 0 else 0.0,
            0.0 if statistics.fmean(diffs) else 1.0,
        )
    t = statistics.fmean(diffs) / (sd / math.sqrt(n))
    return t, _t_sf(abs(t), n - 1) * 2


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Standardized effect size; None when the pooled variance is undefined."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled = ((na - 1) * statistics.variance(a) + (nb - 1) * statistics.variance(b)) / (na + nb - 2)
    if pooled <= 0:
        return None
    return (statistics.fmean(a) - statistics.fmean(b)) / math.sqrt(pooled)


def compare(
    runs: Sequence[RunRecord], metric: str, *, baseline_arm: str, treatment_arm: str
) -> Comparison | None:
    """Treatment vs baseline, paired by seed when the seeds match."""
    data = collect(runs, metric)
    base, treat = data.get(baseline_arm), data.get(treatment_arm)
    if not base or not treat:
        return None

    base_summary = summarize(list(base.values()), name=metric, arm=baseline_arm)
    treat_summary = summarize(list(treat.values()), name=metric, arm=treatment_arm)
    effect = treat_summary.mean - base_summary.mean

    shared = sorted(set(base) & set(treat))
    if len(shared) >= 2:
        test = paired_t_test([(treat[s], base[s]) for s in shared])
        note = f"paired over {len(shared)} shared seeds"
        n_seeds = len(shared)
    else:
        test = welch_t_test(list(treat.values()), list(base.values()))
        note = "unpaired (seeds do not match across arms)"
        n_seeds = min(len(base), len(treat))

    return Comparison(
        metric=metric,
        baseline_arm=baseline_arm,
        treatment_arm=treatment_arm,
        baseline=base_summary,
        treatment=treat_summary,
        effect=effect,
        relative_effect=(effect / base_summary.mean) if base_summary.mean else None,
        p_value=test[1] if test else None,
        effect_size=cohens_d(list(treat.values()), list(base.values())),
        n_seeds=n_seeds,
        note=note,
    )


def effect_interval(
    comparison: Comparison, *, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Confidence interval for the treatment-minus-baseline effect.

    Needed to tell "we could not detect an effect" apart from "we can rule out
    an effect this large" -- the difference between an inconclusive experiment
    and a genuine negative result.
    """
    n_t, n_b = comparison.treatment.n, comparison.baseline.n
    if n_t < 2 or n_b < 2:
        return None
    se = math.sqrt(comparison.treatment.std**2 / n_t + comparison.baseline.std**2 / n_b)
    if se == 0:
        return (comparison.effect, comparison.effect)
    margin = t_critical(min(n_t, n_b) - 1) * se
    return (comparison.effect - margin, comparison.effect + margin)


def rules_out_effect(
    comparison: Comparison, threshold: float, *, higher_is_better: bool = True
) -> bool:
    """Whether the data can exclude an improvement as large as ``threshold``.

    This is the equivalence-testing question, not the significance question: a
    tight interval sitting entirely below the pre-registered threshold refutes
    the prediction even though the effect is not distinguishable from zero.
    """
    interval = effect_interval(comparison)
    if interval is None:
        return False
    low, high = interval if higher_is_better else (-interval[1], -interval[0])
    return high < threshold


def overlapping_error_bars(comparison: Comparison) -> bool:
    """True when the arms are not visually distinguishable at 1 sd.

    Reported alongside p-values because a significant p with overlapping bars at
    n=3 is a claim worth doubting.
    """
    lo_t, hi_t = (
        comparison.treatment.mean - comparison.treatment.std,
        comparison.treatment.mean + comparison.treatment.std,
    )
    lo_b, hi_b = (
        comparison.baseline.mean - comparison.baseline.std,
        comparison.baseline.mean + comparison.baseline.std,
    )
    return not (hi_t < lo_b or hi_b < lo_t)


def _t_sf(t: float, df: float) -> float:
    """Upper-tail probability of Student's t.

    Uses scipy when present and an incomplete-beta fallback otherwise, so
    statistics never silently degrade to a normal approximation.
    """
    try:
        from scipy import stats

        return float(stats.t.sf(t, df))
    except ImportError:
        x = df / (df + t * t)
        return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta via its continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x >= (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)

    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + numerator / (1e-30 if abs(c) < 1e-30 else c)
        step = c * d
        f *= step
        if abs(1.0 - step) < 1e-10:
            break
    return front * (f - 1.0)


__all__ = [
    "cohens_d",
    "effect_interval",
    "collect",
    "compare",
    "overlapping_error_bars",
    "paired_t_test",
    "rules_out_effect",
    "summarize",
    "summarize_runs",
    "t_critical",
    "welch_t_test",
]
