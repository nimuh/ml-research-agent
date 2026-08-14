---
type: spec
id: exp-001
spec_hash: ""                  # written by scripts/spec_hash.py --write
title: "{{Curriculum vs random ordering at smoke scale}}"
status: preregistered          # preregistered | implemented | running | complete
hypothesis: "{{Curriculum ordering improves accuracy at matched compute}}"
arms: [random, curriculum]
baseline: random
dataset: {name: toy, split: test}
metrics: [accuracy]
controls: ["matched compute: equal parameters and equal tokens", "identical data budget", "same seeds per arm"]
seeds: [0, 1, 2]
scale: smoke                   # smoke | small | main
max_runtime_minutes: 10
cost_ceiling_usd: 1.0
preregistered_at: {{ISO8601}}
added: {{DATE}}
tags: [spec]
---

# {{TITLE}}

Tests [[brief]] C1. Comes from the gap in [[map]].

## Hypothesis

{{Stated so it could be false.}}

## Decision rule

*Fixed before any run exists. Covered by `spec_hash`; editing it after the fact
is a blocker in `lint-vault`, not a revision.*

```yaml
metric: accuracy
comparator: ">"
threshold: 0.02
min_seeds: 3
max_p_value: 0.05
refutes_if: "The 95% CI of the paired difference sits entirely below +0.02"
```

- **Supported** if the paired mean difference against the worst-performing
  baseline exceeds +0.02 with p < 0.05 across at least 3 seeds.
- **Refuted** if the 95% CI of that difference lies entirely below +0.02 — the
  effect is ruled out, which is a result, not a failure.
- **Inconclusive** otherwise, or if fewer than 3 seeds completed.

## Design

| | |
|---|---|
| Independent variable | `ordering` ∈ {random, curriculum} |
| Dependent variable | accuracy on [[toy]] test split, higher is better |
| Controls | {{each one specific — "matched compute" names what is matched}} |
| Baselines | {{and the tuning budget each received}} |
| Seeds | 0, 1, 2 |
| Scale | smoke — correctness only, answers nothing on its own |

## Reference numbers

{{What the literature reports for this baseline in this setting, with the source
note. Reproducing it within tolerance at smoke scale is how a broken harness is
caught before it costs a conclusion.}}

| Setting | Reported | Source |
|---|---|---|
| {{random order, toy}} | {{0.604}} | [[arxiv-2401.00001]] (E3) |

## Leakage check

- Eval split is disjoint from training data: {{how this was verified}}
- Hyperparameters were not selected on the test split: {{...}}
- The held-out set is held out from *every* arm: {{...}}

## Stopping rule

{{What ends this experiment regardless of results: wall-clock, run count, spend.}}

## Cost

{{arms × seeds}} runs × {{minutes}} = {{total}}. Ceiling: {{...}}
