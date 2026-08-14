---
name: write-memo
description: Write the final research memo from a vault — a report where every claim cites a source note or a run id, negative and inconclusive results are stated as plainly as positive ones, and what the evidence does not support is named. Use to write up, report, or summarise the findings of a research project or knowledge vault.
argument-hint: "[question or path to a vault]"
user-invocable: true
---

# write-memo

Write what the evidence supports. Nothing else.

The failure this exists to prevent is confident prose the evidence does not
back — which is the natural output of a fluent writer with a folder of material,
and is what makes an unreliable research system dangerous rather than merely
useless.

## The rule

**Every claim cites a source note or a run id.** Not "prior work suggests" —
`[[arxiv-2312.00752]]` suggests, and you can say which quote. If you cannot
attach a citation, you have three options: find one, weaken the sentence until
it is supportable, or delete it. Writing it anyway is not among them.

## Steps

### 1. Build the citation list first

Before writing a word: list every source note in `sources/` and every run record
in `experiments/*/runs/`. **That list is the complete set of things you are
permitted to assert.** Working the other way round — writing the argument, then
finding support — is how citations end up attached to sentences they do not
support.

### 2. Lead with the answer

The first paragraph states what was found. Not the background, not the method,
not a summary of what the memo will cover. If the answer is "the effect is ruled
out at this scale" or "we could not tell," that goes first, in those words.

### 3. Write it

Use `references/memo-template.md`. The sections earn their place:

- **Answer** — the finding, with its verdict and effect size.
- **What the literature already knew** — settled and contested, from `map.md`,
  every line cited.
- **What we tested** — the pre-registered hypothesis and rule, quoted from
  `spec.md`. State that it was fixed before running, because that is the reason
  the result means anything.
- **What we found** — the numbers from `results.md`, with the interval and the
  seed count. Never a mean without its spread.
- **What this does not show** — scope limits, threats to validity, every major
  finding red-team raised. This section is not a disclaimer; it is the part that
  makes the rest trustworthy.
- **What would settle it** — the next experiment, honestly costed.

### 4. Say what happened, including when nothing did

- A **refuted** hypothesis is a finding. "Curriculum ordering does not improve
  accuracy at this scale; the 95% CI of the difference is [-0.004, +0.011],
  entirely below the 0.02 threshold pre-registered before running." That is a
  useful sentence and a complete result.
- An **inconclusive** result is reported as inconclusive. Give the effect, the
  spread, and what it would take to tell. Do not round it toward a story.
- If the **literature already answered the question**, that is the memo. Cite it,
  say what it cost to find out, and give the sharper question the survey exposed.

### 5. Check before delivering

Read every sentence and ask: *what backs this?* Then:

- Every citation resolves to a note or run that exists. Run `lint-vault` on the
  vault to confirm mechanically.
- Every number appears in a `results.md` or a source note's `## Evidence`.
- No claim exceeds its scope limits — smoke-scale results are labelled as such
  everywhere they appear, not only in the caveats.
- The abstract makes no claim the body does not support with a citation.

Write to `reports/<slug>.md`, append to `log.md`, update `README.md`.

## Getting this wrong

**Hedged prose standing in for evidence.** "Results suggest a promising trend"
means nothing and cites nothing. Either the CI excludes the threshold or it does
not — say which.

**The citation that does not support the sentence.** Attaching `[[note]]` to a
claim the note does not make is worse than no citation, because it looks
checked. This is what red-team samples for.

**Scope creep in the abstract.** A smoke-scale result on one toy dataset becomes
"curriculum learning improves language models" by the third draft. Watch for the
sentence getting more general as it gets shorter.

**Burying the negative.** If the answer is no, the first paragraph says no.

**Citing your own summary.** A claim traces to a *quote* in a source note's
`## Evidence`, not to the summary paragraph above it — which is your prose, and
citing it is citing yourself.
