#!/usr/bin/env python3
"""Apply a pre-registered decision rule to the statistics, mechanically.

The rule was written before the numbers existed precisely so that applying it
would not be a judgment call. This script is what makes that true: it takes the
rule from `spec.md` and the output of `stats.py` and returns
supported / refuted / inconclusive without anybody weighing anything.

Three decisions inside it are worth stating, because each one changes verdicts:

*It compares against the baseline the treatment does worst against.* With
several baselines, using the first one declared would let the order someone
happened to list them in decide the outcome, and would let adding a weak
baseline manufacture a win.

*A clean null is `refuted`, not `inconclusive`.* A significance test alone
cannot tell "we ruled the effect out" from "we could not tell" -- an effect
indistinguishable from zero fails significance either way. So when the whole
confidence interval sits below the threshold, the prediction is refuted. That is
what makes a negative result legible instead of a shrug.

*The seed floor is checked first and is not negotiable.* Too few seeds is
inconclusive no matter how large the difference looks, because the equivalence
check above would otherwise smuggle a verdict past a sample of two.

Usage:
    decide.py --spec experiments/exp-001/spec.md --stats stats.json
    stats.py runs/ --json | decide.py --spec spec.md --stats -
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

SUPPORTED = "supported"
REFUTED = "refuted"
INCONCLUSIVE = "inconclusive"

COMPARATORS = {
    ">": lambda diff, threshold: diff > threshold,
    ">=": lambda diff, threshold: diff >= threshold,
    "<": lambda diff, threshold: diff < threshold,
    "<=": lambda diff, threshold: diff <= threshold,
}


def parse_rule(spec_path: Path) -> dict[str, object]:
    """Pull the decision rule out of a spec, from front matter or the body block.

    Both places are accepted because the template writes it twice -- once as
    YAML for machines, once in prose for humans -- and a spec that has drifted
    between them is a real thing that happens. The front matter wins; the body
    is the fallback so an older spec still evaluates.
    """
    text = spec_path.read_text(encoding="utf-8")

    block = ""
    marker = re.search(r"^decision_rule:\s*$", text, re.MULTILINE)
    if marker:
        rest = text[marker.end() :]
        lines: list[str] = []
        for line in rest.splitlines():
            if line.startswith((" ", "\t")) or not line.strip():
                lines.append(line)
            else:
                break
        block = "\n".join(lines)
    else:
        fenced = re.search(r"```ya?ml\n(.*?metric:.*?)```", text, re.DOTALL)
        if fenced:
            block = fenced.group(1)

    if not block.strip():
        raise SystemExit(f"{spec_path}: no decision_rule found")

    rule: dict[str, object] = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip().strip("\"'")
        if not key or key.startswith("#"):
            continue
        if key in ("threshold", "max_p_value"):
            try:
                rule[key] = float(value)
            except ValueError:
                continue
        elif key == "min_seeds":
            try:
                rule[key] = int(value)
            except ValueError:
                continue
        else:
            rule[key] = value
    return rule


def worst_comparison(stats: dict, arm: str | None) -> dict | None:
    """The comparison the treatment does worst on -- the honest one to judge by.

    With one treatment and several baselines, `stats.py` emits one comparison
    per pair; the smallest difference is the hardest baseline to beat. With
    several treatments and `--arm` unset, the same logic picks the weakest
    result rather than the most flattering.
    """
    comparisons = [c for c in stats.get("comparisons", []) if arm is None or c["arm"] == arm]
    if not comparisons:
        return None
    return min(comparisons, key=lambda c: c["mean_difference"])


def decide(rule: dict, stats: dict, arm: str | None = None) -> dict[str, object]:
    comparison = worst_comparison(stats, arm)
    threshold = float(rule.get("threshold", 0.0))
    comparator = str(rule.get("comparator", ">"))
    min_seeds = int(rule.get("min_seeds", 3))
    max_p = float(rule.get("max_p_value", 0.05))
    test = COMPARATORS.get(comparator, COMPARATORS[">"])

    verdict = {
        "verdict": INCONCLUSIVE,
        "reason": "",
        "metric": rule.get("metric", stats.get("metric")),
        "comparator": comparator,
        "threshold": threshold,
        "min_seeds": min_seeds,
        "max_p_value": max_p,
        "comparison": comparison,
        "checks": {},
    }

    if comparison is None:
        verdict["reason"] = "no arm could be compared against the baseline"
        return verdict

    n = int(comparison["n_pairs"])
    diff = float(comparison["mean_difference"])
    ci_low = float(comparison["ci_low"])
    ci_high = float(comparison["ci_high"])
    p = float(comparison["p_value"])

    checks = {
        "seeds": {"got": n, "required": min_seeds, "pass": n >= min_seeds},
        "effect": {"difference": diff, "threshold": threshold, "pass": test(diff, threshold)},
        "significance": {
            "p_value": p,
            "max": max_p,
            "pass": (not math.isnan(p)) and p < max_p,
        },
    }
    verdict["checks"] = checks

    # The seed floor first, and unconditionally. Everything below reasons about
    # an interval, and an interval from two points is not evidence.
    if n < min_seeds:
        verdict["reason"] = (
            f"{n} seed pair(s) completed, {min_seeds} required by the pre-registered rule. "
            "A difference across too few seeds is not a result whichever way it points."
        )
        return verdict

    if checks["effect"]["pass"] and checks["significance"]["pass"]:
        verdict["verdict"] = SUPPORTED
        verdict["reason"] = (
            f"paired difference {diff:+.4f} against `{comparison['baseline']}` "
            f"{comparator} {threshold:+.4f} with p={p:.4f} < {max_p}, over {n} seeds "
            f"(95% CI [{ci_low:+.4f}, {ci_high:+.4f}])."
        )
        return verdict

    # The equivalence check: the whole interval below the threshold rules the
    # effect out. This is the clause that lets a well-run experiment deliver a
    # clean negative instead of a shrug.
    ruled_out = (
        not math.isnan(ci_high) and ci_high < threshold
        if comparator in (">", ">=")
        else not math.isnan(ci_low) and ci_low > threshold
    )
    if ruled_out:
        verdict["verdict"] = REFUTED
        verdict["reason"] = (
            f"the 95% CI of the difference [{ci_low:+.4f}, {ci_high:+.4f}] lies entirely "
            f"{'below' if comparator in ('>', '>=') else 'above'} the pre-registered "
            f"threshold {threshold:+.4f} over {n} seeds. The effect is ruled out at this "
            "scale -- a result, not a failure."
        )
        verdict["checks"]["equivalence"] = {"pass": True, "ci": [ci_low, ci_high]}
        return verdict

    failed = [name for name, check in checks.items() if not check["pass"]]
    verdict["reason"] = (
        f"difference {diff:+.4f} with 95% CI [{ci_low:+.4f}, {ci_high:+.4f}], p={p:.4f} "
        f"over {n} seeds: {', '.join(failed)} did not pass, and the interval is too wide "
        f"to rule the effect out. Too noisy to tell."
    )
    return verdict


def render(result: dict) -> str:
    lines = [f"VERDICT: {result['verdict'].upper()}", "", result["reason"], ""]
    comparison = result.get("comparison")
    if comparison:
        lines.append(
            f"  {comparison['arm']} vs {comparison['baseline']} "
            f"on {result['metric']}, {comparison['n_pairs']} paired seeds"
        )
    lines.append("")
    lines.append("  pre-registered rule:")
    lines.append(
        f"    {result['metric']} {result['comparator']} {result['threshold']}  "
        f"min_seeds={result['min_seeds']}  p<{result['max_p_value']}"
    )
    if result["checks"]:
        lines.append("")
        for name, check in result["checks"].items():
            mark = "pass" if check.get("pass") else "FAIL"
            detail = ", ".join(f"{k}={v}" for k, v in check.items() if k != "pass")
            lines.append(f"    [{mark}] {name}: {detail}")
    lines.append("")
    lines.append(
        "  A blocking finding from `red-team` downgrades this to inconclusive, "
        "whatever the arithmetic said."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="the pre-registered spec.md")
    parser.add_argument("--stats", required=True, help="stats.py --json output, or - for stdin")
    parser.add_argument("--arm", help="treatment arm to judge (default: the worst-performing)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.spec.is_file():
        print(f"no such spec: {args.spec}", file=sys.stderr)
        return 2

    raw = sys.stdin.read() if args.stats == "-" else Path(args.stats).read_text(encoding="utf-8")
    stats = json.loads(raw)
    result = decide(parse_rule(args.spec), stats, args.arm)

    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
