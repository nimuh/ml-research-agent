---
name: design-experiment
description: Design a pre-registered experiment — independent variables, controls, baselines, seeds, a scale ladder, and a decision rule stating what result would refute the hypothesis, hashed before any run exists. Use when the user wants to design, plan, or pre-register an experiment, an ablation, or an A/B comparison.
user-invocable: true
---

# design-experiment

Write down what would prove you wrong, before you spend anything finding out.

This is the highest-value step in the whole system. Fixing the decision rule in
advance is what makes a negative result a *finding* rather than a *failure*, and
what removes the incentive to rationalise whatever comes out. Everything else
here is bookkeeping in service of that one commitment.

## What you produce

`experiments/<id>/spec.md`, hashed and marked `status: preregistered`. No code,
no runs. `run-experiment` comes next and is not allowed to start without this.

## Steps

### 1. Take the question from the map

`map.md` ends with the open question the vault exposed; `brief.md` has the
claims and the success thresholds. If the question is not stated as a comparison
with a measurable outcome, stop and go back — an experiment designed around a
vague question produces a number nobody can interpret.

### 2. Name the variables and the controls

- **Independent variables** — what you deliberately change, with the exact
  values. One at a time unless you have a reason and say what it is.
- **Dependent variables** — the metrics, and which direction is better.
- **Controls** — everything held equal. Be specific: "matched compute" means
  nothing until it says matched on *what* — parameters, tokens, wall-clock, FLOPs.
  This line is where most unfair comparisons enter, and they enter by being
  vague rather than by being wrong.

### 3. Choose baselines that can win

A baseline exists to beat your treatment if your treatment is not better. That
means it gets the same tuning budget, the same data, and the same care.

An undertuned baseline is the most common defect in ML results and the easiest
to commit accidentally — you tune what you are excited about. Write down what
tuning each arm received; if they differ, the comparison is decoration.

Where the literature reports a number for your baseline setting, record it in
the spec. Reproducing it within tolerance at smoke scale is how you find out
your harness is wrong before it costs you a conclusion.

### 4. Seeds, and the ladder

- **Seeds: 3 minimum.** A single-seed delta is noise with a decimal point, and
  `analyze-results` reports it as inconclusive by construction. If three is
  genuinely impossible, say so in the spec and expect the verdict to reflect it.
- **Scale ladder: `smoke` → `small` → `main`.** Smoke checks correctness in
  minutes and answers nothing. Small is a reduced but honest setting. Main is
  the real thing. Design the smoke rung now; the later rungs are only reached if
  the earlier one passes, and that sequencing is enforced by `run-experiment`,
  not left to enthusiasm.

### 5. Write the decision rule — before anything runs

The heart of it:

```yaml
decision_rule:
  metric: accuracy
  comparator: ">"
  threshold: 0.02          # the effect size worth caring about
  min_seeds: 3
  max_p_value: 0.05
  refutes_if: "The 95% CI of the paired difference sits entirely below +0.02"
```

Three properties it must have:

- **The threshold is an effect size, not a p-value.** "Statistically significant"
  is not a finding; "at least two points, which is what would change what
  someone does" is.
- **`refutes_if` describes an observation you could actually make.** This is what
  separates a clean negative from "we could not tell": if the confidence interval
  sits entirely below the threshold, the effect is ruled out, and that is a
  result. Without this clause a well-run experiment that genuinely settles the
  question in the negative reads identically to one too noisy to say anything.
- **It compares against the baseline the treatment does *worst* against**, when
  there are several. Otherwise the order you happened to list baselines in
  decides the verdict, and adding a weak one manufactures a win.

Also fix the **stopping rule** and the **cost ceiling** now: how long a run may
take, how many runs total, what wall-clock or spend ends the experiment
regardless of results.

### 6. Hash it

```bash
python3 scripts/spec_hash.py experiments/<id>/spec.md --write
```

This writes `spec_hash` into the frontmatter, covering the hypothesis, arms,
controls, seeds, scale and the decision rule. From here, any edit to those is
detectable — `lint-vault` re-computes the hash and reports a mismatch as a
blocker.

**If you later need to change the design, register a new experiment id.** Do not
edit a pre-registered spec and re-hash it. The old spec staying wrong in the
record is the point; that is what an audit trail is.

### 7. Hand off

Append to `log.md`, update `README.md`, then run `red-team` on the spec before
any compute is spent. Show the user the arms × seeds count, the estimated cost,
and the decision rule, and get approval before `run-experiment`.

## Getting this wrong

**The rule written after the numbers.** The single failure this skill exists to
prevent. If you find yourself adjusting a threshold once you have seen a result,
you are no longer running an experiment.

**A decision rule that cannot fail.** "Supported if the treatment is better"
with no threshold and no seed floor will be satisfied by noise roughly half the
time.

**Leakage.** Check that the eval set is not in the training data, that
hyperparameters were not selected on the test split, and that the "held-out" set
is held out from *every* arm. Write down how you checked.

**Confounded arms.** If the treatment arm also changes the learning rate, the
data order and the batch size, the experiment tests nothing. One change, or a
factorial design that names every cell.
