---
type: result
id: exp-001
spec_hash: 8f3ad2c1b4e50917
verdict: supported             # supported | refuted | inconclusive
metric: accuracy
baseline: random
treatment: curriculum
n_pairs: 3
mean_difference: 0.0994
ci: [0.0871, 0.1117]
p_value: 0.0012
runs_ok: 6
runs_total: 6
downgraded_by_critique: false
added: 2026-08-14
tags: [result]
---

# Results — exp-001

Spec: [[spec]] (pre-registration verified intact) · Scale: smoke

## Verdict: {{SUPPORTED}}

{{The `reason` line decide.py returned, verbatim. Not a rephrasing — this is the
sentence the pre-registered rule produced.}}

## Numbers

| Arm | n | Mean | SD | Values |
|---|---|---|---|---|
| random | 3 | 0.6021 | 0.0041 | 0.5983, 0.6018, 0.6062 |
| curriculum | 3 | 0.7015 | 0.0035 | 0.6981, 0.7013, 0.7051 |

Paired by seed against `random`: **+0.0994**, 95% CI [+0.0871, +0.1117],
p = 0.0012, n = 3.

Runs: [[curriculum-seed0]] [[curriculum-seed1]] [[curriculum-seed2]]
[[random-seed0]] [[random-seed1]] [[random-seed2]]

## Runs that did not complete

{{Category and count, or "none". Never omit this section — eight runs where four
crashed is not a clean n=4, and the crashes may correlate with the arm.}}

## Threats to validity

{{What would make this wrong. Include every major finding red-team raised, and
the scope limits from the brief that still bind.}}

- Smoke scale only; nothing here licenses a claim at the scale the brief cares about
- Single dataset
- {{...}}

## What this does not show

{{The claims a reader might take from the number that the experiment does not
support. This section is the honest half of a positive result.}}

## Critique

Red-team: {{clean / N major findings / BLOCKED}}. {{If blocked, the verdict above
is inconclusive regardless of the arithmetic, and the reason is recorded here.}}

## Follow-up

{{If inconclusive: the cheapest experiment with the best information gain, and
what it would settle. If supported or refuted: what the next rung of the ladder
would test, if anything.}}
