---
name: frame-idea
description: Turn a loose research idea into a falsifiable brief — claims with explicit falsifiers, measurable success criteria, scope limits, and the search vocabulary a survey needs. Use before surveying literature or designing any experiment, or when the user asks to sharpen, frame, or make a research question testable.
argument-hint: "<your research idea>"
user-invocable: true
---

# frame-idea

Turn what the user said into something that could turn out to be wrong.

Most research ideas arrive in a form no evidence could contradict — "does X help
Y?", "is A better than B?". Framing is the step that converts one of those into a
claim with a stated falsifier, before any effort is spent. An unframeable idea
is worth discovering now rather than after a survey and a week of compute.

## What you produce

`brief.md` in the vault root, following `_meta/schema.md`. If there is no vault
yet, the `research-idea` skill scaffolds one — read its `references/schema.md`
for the contract.

## Steps

### 1. Understand the idea before reshaping it

Read what the user wrote. If it is genuinely ambiguous in a way that changes the
work — two readings that would send the survey to different literatures, or a
comparison with no stated baseline — ask **one** round of questions. Otherwise
make the reasonable call, state the assumption in the brief, and move on. Do not
interrogate someone who gave you a clear question.

### 2. Write the claims

Each claim needs three things:

- **The statement.** Specific enough to be checked. "Mamba-1 beats transformers
  on long-context genomic modelling" is not yet checkable; "at ≥32k bases, a
  Mamba-1 model matched on parameters and tokens reaches lower perplexity than a
  transformer of the same budget" is.
- **The falsifier.** *What result would make this false?* Write it as a concrete
  outcome, not a hedge. "The difference is within seed-to-seed variance at every
  context length" is a falsifier. "It might not work" is not.
- **A prior.** Your honest confidence, 0–1, before looking anything up. It is
  there to be compared with what the survey turns up; a claim you were 0.9 sure
  of that the literature refutes is the most informative thing a survey can find.

If you cannot write the falsifier, say so plainly and offer the user the nearest
question that has one. That conversation is the whole value of this phase.

### 3. Write the success criteria

Measurable, with a metric, a comparator, a threshold and a dataset. The
threshold is the effect size worth caring about, decided now — not the one that
turns out to be significant later.

### 4. Write the scope limits

What this project explicitly does not cover: scales you will not reach, settings
you will not test, claims you will not make. Scope limits are what stop a
smoke-scale result being reported as a general one.

### 5. Write the search vocabulary and the seminal-works checklist

The survey depends on this and cannot fix a bad one:

- **Search terms** — the words the field actually uses, including the older name
  for the same idea and the term the adjacent subfield uses for it. One phrasing
  reaches one community.
- **Seminal works** — papers you would be embarrassed to miss, **by title**.
  These are checked against what the survey returns; a named work that never
  surfaces is reported as a recall failure. Commit to them here, before the
  search can influence the list, or the check means nothing.
- **Adjacent areas**, **key authors**, **venues**.

### 6. Write the file and log it

Use `references/brief-template.md`. Append an entry to `log.md`, update the
status table in `README.md`, and show the user the claims and their falsifiers.

## Getting this wrong

**The unfalsifiable claim.** "Attention is important for long context." Nothing
could refute it. Push for the version with a number in it.

**The falsifier that is just the negation.** "It is false if curriculum ordering
does not improve accuracy" restates the claim. A falsifier names the *observation*
you would make: "if the paired difference across 3 seeds has a 95% CI containing
zero."

**Thresholds chosen to be beatable.** Pick the effect size that would change
what someone does, then let the experiment come out how it comes out.

**Silent scope creep.** If the idea is really three questions, say so and ask
which one to run. Do not quietly research the easiest.
