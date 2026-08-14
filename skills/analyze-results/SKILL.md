---
name: analyze-results
description: Analyze experiment runs against a pre-registered decision rule — per-arm seed statistics, paired differences with confidence intervals, and a mechanical supported/refuted/inconclusive verdict. Use after running an experiment, or when the user asks what the results mean, whether an effect is real, or whether a difference is significant.
user-invocable: true
---

# analyze-results

Apply the rule that was fixed before the numbers existed, and report what it
says — including when what it says is "we could not tell."

The scripts do the deciding. That is deliberate: the whole value of
pre-registering a decision rule is that applying it is not a judgment call, and
a model reading three numbers and forming an impression is exactly the judgment
call it was meant to remove.

## Steps

### 1. Check the pre-registration is intact

```bash
python3 ../design-experiment/scripts/spec_hash.py experiments/<id>/spec.md --verify
```

A mismatch means the rule changed after runs existed. Stop, and report it — no
verdict computed against an edited rule is worth anything, and the whole
apparatus around it makes that detectable rather than deniable.

### 2. Compute the statistics

```bash
python3 scripts/stats.py experiments/<id>/runs/ --json > /tmp/stats.json
python3 scripts/stats.py experiments/<id>/runs/          # human-readable
```

You get per-arm mean, sample standard deviation and the raw values; paired
differences by seed against the baseline, with a 95% CI and a two-sided p-value;
and the failure counts by category.

Pairing is by seed on purpose. Two arms at seed 0 share their initialisation and
data order, so the paired difference cancels most of the variance an unpaired
test would leave in — which is what makes three seeds informative at all.

### 3. Apply the rule

```bash
python3 scripts/decide.py --spec experiments/<id>/spec.md --stats /tmp/stats.json
```

It returns one of:

- **supported** — the effect exceeded the pre-registered threshold, significantly,
  over enough seeds.
- **refuted** — the confidence interval sits entirely below the threshold. The
  effect is ruled out at this scale. **This is a result**, and often a more
  useful one than a positive: report it with the same confidence.
- **inconclusive** — too few seeds, or an interval too wide to distinguish "no
  effect" from "no signal". Say which, because they call for different
  follow-ups.

It judges against the baseline the treatment does *worst* against. Otherwise the
order baselines were listed in would decide the verdict, and adding a weak one
would manufacture a win.

### 4. Write results.md

`experiments/<id>/results.md`, from `references/results-template.md`. The
numbers come from the scripts — do not retype them, and do not round them into
something friendlier.

The body is where you add what the scripts cannot: what the failures were, which
threats to validity survive, and what the result does *not* license. Every run
referenced gets a `[[link]]` so `lint-vault` can check the citation resolves.

### 5. Red-team before believing it

Run `red-team` on the result. If it returns a blocker, **the verdict is
inconclusive whatever the arithmetic said** — a defensible-looking number with a
leaky eval behind it is not a result. Record the downgrade and the reason.

Major findings do not overturn a measurement, but they belong in the threats
section; dropping them loses the caveats a reader needs.

### 6. What comes next

- **Supported or refuted** → `write-memo`.
- **Inconclusive** → propose the cheapest follow-up with the best information
  gain: more seeds if the interval is wide, a bigger scale rung if the effect
  may exist but is small, a design fix if red-team found a confound. Go back to
  `design-experiment` with a *new* spec id.
- **Inconclusive twice on the same hypothesis** → stop and ask the user. Two
  rounds means the design is not answering the question, and more compute will
  not change that.

## Getting this wrong

**Reading the means and forming a view.** 0.70 versus 0.60 looks decisive and
may be nothing at n=3. Run the script.

**Treating p < 0.05 as the finding.** The threshold is an effect size. A
significant difference below the pre-registered threshold is a *refutation*, not
a win.

**Silently dropping failed runs.** Eight runs where four crashed is not a
clean n=4; the crashes may correlate with the arm. Report the failure counts
alongside the verdict, always.

**Reporting smoke-scale numbers as results.** Smoke checks the harness. Its
numbers belong in the log.

**Rerunning until it works.** Every extra unplanned run inflates the false
positive rate. If the design was wrong, fix the design and say so — that is what
new spec ids are for.
