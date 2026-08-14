#!/usr/bin/env python3
"""Seed statistics for an experiment: per-arm summaries and paired comparisons.

This exists because reading a mean off three numbers is the point at which a
language model is least trustworthy and most confident. Spread across seeds,
paired differences, and a confidence interval are arithmetic, and arithmetic
should be computed rather than estimated -- the difference between "0.70 vs
0.60, clearly better" and "0.70 ± 0.08 vs 0.60 ± 0.09, n=3, CI crosses zero" is
the difference between a finding and a story.

Pairing is by seed, deliberately. Two arms trained on seed 0 share their data
order and their initialisation, so the paired difference cancels most of the
variance an unpaired test would leave in -- which is what makes three seeds
informative at all.

Everything is stdlib: Student's t is computed here rather than imported, so the
script runs under bare `python3`.

Usage:
    stats.py experiments/exp-001/runs/          # markdown run records
    stats.py runs.json --baseline random        # or a JSON array
    stats.py experiments/exp-001/runs/ --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONFIDENCE = 0.95


@dataclass
class Run:
    arm: str
    seed: int
    status: str
    metrics: dict[str, float]


@dataclass
class ArmSummary:
    arm: str
    n: int
    mean: float
    sd: float
    values: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "n": self.n,
            "mean": round(self.mean, 6),
            "sd": round(self.sd, 6),
            "values": [round(v, 6) for v in self.values],
        }


@dataclass
class Comparison:
    arm: str
    baseline: str
    n_pairs: int
    mean_difference: float
    sd_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    seeds: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "baseline": self.baseline,
            "n_pairs": self.n_pairs,
            "mean_difference": round(self.mean_difference, 6),
            "sd_difference": round(self.sd_difference, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "p_value": round(self.p_value, 6),
            "seeds": self.seeds,
        }


# ---------------------------------------------------------------------------
# Student's t, without scipy
# ---------------------------------------------------------------------------


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - _log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """One-tailed survival function of Student's t."""
    if df <= 0:
        return float("nan")
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def t_two_sided_p(t: float, df: float) -> float:
    if df <= 0 or math.isnan(t):
        return float("nan")
    return min(1.0, 2.0 * t_sf(abs(t), df))


def t_critical(df: float, confidence: float = CONFIDENCE) -> float:
    """Two-tailed critical value, found by bisection on the CDF.

    Bisection rather than a table so the seed count is not silently capped at
    whatever a hard-coded table happened to cover.
    """
    if df <= 0:
        return float("nan")
    target = (1.0 - confidence) / 2.0
    low, high = 0.0, 1000.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if t_sf(mid, df) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Loading runs
# ---------------------------------------------------------------------------

_METRIC_PAIR = re.compile(r"([A-Za-z_][\w./-]*)\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def _parse_metrics(value: str) -> dict[str, float]:
    """Read `metrics: {accuracy: 0.70, loss: 1.2}` from front matter."""
    return {m.group(1): float(m.group(2)) for m in _METRIC_PAIR.finditer(value)}


def load_markdown_runs(directory: Path) -> list[Run]:
    runs: list[Run] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.startswith("---"):
            continue
        end = text.find("\n---", 3)
        raw = text[3 : end if end != -1 else len(text)]

        fields: dict[str, str] = {}
        for line in raw.splitlines():
            key, sep, value = line.partition(":")
            if sep and not line.startswith((" ", "\t", "-")):
                fields[key.strip()] = value.strip()

        if "arm" not in fields or "seed" not in fields:
            continue
        try:
            seed = int(fields["seed"])
        except ValueError:
            continue
        runs.append(
            Run(
                arm=fields["arm"].strip("\"'"),
                seed=seed,
                status=fields.get("status", "ok").strip("\"'"),
                metrics=_parse_metrics(fields.get("metrics", "")),
            )
        )
    return runs


def load_json_runs(path: Path) -> list[Run]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Run(
            arm=str(item["arm"]),
            seed=int(item["seed"]),
            status=str(item.get("status", "ok")),
            metrics={k: float(v) for k, v in (item.get("metrics") or {}).items()},
        )
        for item in data
    ]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def summarise(runs: list[Run], metric: str) -> list[ArmSummary]:
    by_arm: dict[str, list[float]] = {}
    for run in runs:
        if run.status == "ok" and metric in run.metrics:
            by_arm.setdefault(run.arm, []).append(run.metrics[metric])

    out: list[ArmSummary] = []
    for arm, values in sorted(by_arm.items()):
        n = len(values)
        mean = sum(values) / n
        # Sample standard deviation: n-1. With three seeds the difference from
        # the population form is not cosmetic.
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
        out.append(ArmSummary(arm=arm, n=n, mean=mean, sd=sd, values=sorted(values)))
    return out


def compare(runs: list[Run], metric: str, baseline: str) -> list[Comparison]:
    """Paired-by-seed comparison of every other arm against ``baseline``."""
    by_arm_seed: dict[tuple[str, int], float] = {
        (r.arm, r.seed): r.metrics[metric]
        for r in runs
        if r.status == "ok" and metric in r.metrics
    }
    arms = sorted({arm for arm, _ in by_arm_seed} - {baseline})

    out: list[Comparison] = []
    for arm in arms:
        seeds = sorted(
            seed
            for (a, seed) in by_arm_seed
            if a == arm and (baseline, seed) in by_arm_seed
        )
        diffs = [by_arm_seed[(arm, s)] - by_arm_seed[(baseline, s)] for s in seeds]
        n = len(diffs)
        if n == 0:
            continue

        mean = sum(diffs) / n
        if n > 1:
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
            stderr = sd / math.sqrt(n)
            df = n - 1
            crit = t_critical(df)
            t_stat = mean / stderr if stderr > 0 else math.inf if mean != 0 else 0.0
            p = t_two_sided_p(t_stat, df) if stderr > 0 else (0.0 if mean != 0 else 1.0)
            half = crit * stderr
        else:
            # One pair is a single observation. Report it with no interval
            # rather than an interval of zero width, which would read as
            # certainty about the least certain thing in the file.
            sd, p, half = 0.0, float("nan"), float("nan")

        out.append(
            Comparison(
                arm=arm,
                baseline=baseline,
                n_pairs=n,
                mean_difference=mean,
                sd_difference=sd,
                ci_low=mean - half,
                ci_high=mean + half,
                p_value=p,
                seeds=seeds,
            )
        )
    return out


def analyse(runs: list[Run], metric: str, baseline: str | None) -> dict[str, object]:
    summaries = summarise(runs, metric)
    if baseline is None and summaries:
        # Without a stated baseline, the lowest-mean arm is the conservative
        # guess -- but `decide.py` needs the spec's declared one, so this is a
        # convenience for reading, not a substitute for saying which it is.
        baseline = min(summaries, key=lambda s: s.mean).arm

    failures: dict[str, int] = {}
    for run in runs:
        if run.status != "ok":
            failures[run.status] = failures.get(run.status, 0) + 1

    return {
        "metric": metric,
        "baseline": baseline,
        "confidence": CONFIDENCE,
        "arms": [s.as_dict() for s in summaries],
        "comparisons": [
            c.as_dict() for c in (compare(runs, metric, baseline) if baseline else [])
        ],
        "runs_total": len(runs),
        "runs_ok": sum(1 for r in runs if r.status == "ok"),
        "failures": failures,
    }


def render(report: dict[str, object]) -> str:
    lines = [f"metric: {report['metric']}   baseline: {report['baseline']}"]
    ok, total = report["runs_ok"], report["runs_total"]
    lines.append(f"runs: {ok}/{total} ok")
    if report["failures"]:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(report["failures"].items()))  # type: ignore[union-attr]
        lines.append(f"failures: {detail}")
    lines.append("")

    lines.append(f"{'arm':<20} {'n':>3} {'mean':>10} {'sd':>10}   values")
    for arm in report["arms"]:  # type: ignore[union-attr]
        values = ", ".join(f"{v:.4f}" for v in arm["values"])
        lines.append(
            f"{arm['arm']:<20} {arm['n']:>3} {arm['mean']:>10.4f} {arm['sd']:>10.4f}   {values}"
        )

    comparisons = report["comparisons"]
    if comparisons:
        lines.append("")
        lines.append(f"paired against {report['baseline']}, {int(CONFIDENCE * 100)}% CI:")
        lines.append("")
        lines.append(f"{'arm':<20} {'pairs':>5} {'diff':>10} {'95% CI':>22} {'p':>9}")
        for c in comparisons:  # type: ignore[union-attr]
            if c["n_pairs"] < 2:
                interval, p = "(single pair)", "  n/a"
            else:
                interval = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
                p = f"{c['p_value']:.4f}"
            lines.append(
                f"{c['arm']:<20} {c['n_pairs']:>5} {c['mean_difference']:>+10.4f} "
                f"{interval:>22} {p:>9}"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="runs/ directory or a JSON array of runs")
    parser.add_argument("--metric", help="metric to analyse (default: the only one present)")
    parser.add_argument("--baseline", help="baseline arm, as named in the spec")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.source.is_dir():
        runs = load_markdown_runs(args.source)
    elif args.source.is_file():
        runs = load_json_runs(args.source)
    else:
        print(f"no such path: {args.source}", file=sys.stderr)
        return 2

    if not runs:
        print(f"no run records found in {args.source}", file=sys.stderr)
        return 2

    metric = args.metric
    if metric is None:
        names = sorted({m for r in runs for m in r.metrics})
        if len(names) != 1:
            print(
                f"several metrics present ({', '.join(names)}); pass --metric",
                file=sys.stderr,
            )
            return 2
        metric = names[0]

    report = analyse(runs, metric, args.baseline)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
