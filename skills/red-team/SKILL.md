---
name: red-team
description: Adversarially review a research artifact — a knowledge vault, a literature map, an experiment spec, or a result — hunting for leaky evaluations, undertuned baselines, cherry-picked seeds, fabricated citations and unsupported claims. Use to critique, red-team, or sanity-check research before acting on it, and after building a vault, mapping it, designing an experiment, or analysing results.
argument-hint: "<path to the artifact to attack>"
user-invocable: true
---

# red-team

Try to refute it. Not to approve it, not to balance strengths against
weaknesses — to find the reason it is wrong.

This is the main defence against a system fluent enough to be convincing while
being wrong, and it only works if you actually adopt the adversarial stance. A
critique that concludes "looks solid overall, some minor caveats" has done
nothing. Start from the assumption that there is a defect and go looking for it.

## When this runs

After **build-vault** (is the corpus real?), after **map-vault** (is the novelty
claim honest?), after **design-experiment** (can this experiment answer
anything?), and after **analyze-results** (does the number mean what it says?).

## Steps

### 1. Pick the checklist for the artifact

**A vault or a source note**

- Pull five notes at random. For each, take one number from `## Results` and find
  the quote in `## Evidence` that states it. **Then open `raw/` and confirm the
  quote is in the dossier verbatim.** A quote that is near the claim but does not
  contain it is the most common real defect here, and no script finds it.
- Any identifier — arXiv id, DOI, URL — that looks constructed rather than
  copied. Follow one and see if it resolves.
- Notes marked `read: full` whose evidence is all abstract.
- Claims marked settled that still list contradicting sources.

**A map**

- Is anything called settled on the strength of one paper, or three papers from
  one lab reusing one benchmark? That is one piece of evidence, not three.
- Is anything called *open* that the survey simply failed to search for? Check
  `raw/_coverage.md` for what was unreachable, and the brief's seminal-works
  checklist for what never surfaced.
- Is the novelty claim honest? Name the closest existing work and say precisely
  how the brief differs. "No prior work does exactly this" is unfalsifiable and
  nearly always false.

**A spec**

- **Leakage.** Is the eval set in the training data? Were hyperparameters
  selected on the test split? Is the held-out set held out from *every* arm?
- **Baseline fairness.** Did each arm get the same tuning budget, the same data,
  the same care? An undertuned baseline is the most common defect in ML results
  and it is committed by accident, by being more excited about the treatment.
- **Confounds.** Does the treatment arm change more than one thing?
- **Vague controls.** "Matched compute" that does not say matched on what.
- **A decision rule that cannot fail** — no threshold, no seed floor, or a
  threshold so small that noise satisfies it.
- Fewer seeds than `min_seeds`.

**A result**

- Does the verdict follow from the pre-registered rule, or from a reading of it?
- Were failed runs dropped? Do the failures correlate with the arm?
- Is a smoke-scale number being reported as a result?
- Is the effect size worth caring about, or merely significant?
- Were there unplanned extra runs? Each one inflates the false-positive rate.
- Does the prose claim more than the interval supports — "consistently
  outperforms" on a CI that nearly touches zero?

### 2. Write findings, with severity

Each finding: what is wrong, **where** (file and line or section), why it
matters, and the concrete fix.

| Severity | Meaning | Effect |
|---|---|---|
| **blocker** | The artifact cannot be relied on | Phase does not pass. A blocker on a result forces the verdict to `inconclusive` regardless of the arithmetic |
| **major** | A real defect that does not invalidate everything | Phase passes; the finding is recorded in threats to validity and must appear in the memo |
| **minor** | Worth fixing, changes no conclusion | Noted |

Be honest about severity in both directions. Inflating a minor to a blocker
stops work for nothing; burying a blocker as a minor is the failure this skill
exists to prevent. If you genuinely find nothing, say so plainly and briefly —
but only after actually running the checks above, and say which ones you ran.

### 3. Record it

Append to `_meta/critiques.md`: the date, the artifact, the findings, the
verdict. Then update the artifact's own note — a result gets the majors added to
its threats section; a blocked result gets its `verdict:` downgraded to
`inconclusive` and `downgraded_by_critique: true`.

### 4. Report

Tell the user the counts by severity and the single most important finding. If
you blocked a phase, say exactly what would unblock it.

## The stance

You are not being asked whether you would sign off on this. You are being asked
what is wrong with it. Those produce different reviews, and only the second one
is useful — the first drifts toward approval because approving is easier and
because the artifact was made by the same process doing the reviewing.

Concretely: for every section you read, try to complete the sentence "this is
wrong because ___" before you decide it is fine.
