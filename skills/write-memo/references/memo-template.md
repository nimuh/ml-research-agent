---
type: report
title: "{{Does curriculum ordering help small-model math reasoning?}}"
verdict: supported             # supported | refuted | inconclusive | settled-by-literature
sources_cited: 12
runs_cited: 6
added: {{DATE}}
tags: [report]
---

# {{TITLE}}

## Answer

{{The finding, first. One paragraph. If it is "no" or "we could not tell", that
goes here in those words — not after three paragraphs of context.}}

{{Example: Curriculum ordering improved accuracy by +9.9 points (95% CI [+8.7,
+11.2], p=0.001, 3 seeds) at smoke scale on a toy split — exceeding the +2 point
threshold pre-registered before any run. The effect is large and consistent, and
nothing here licenses a claim beyond this scale or dataset.}}

## What the literature already knew

**Settled.** {{Statement}} — [[source-a]], [[source-b]], [[source-c]]

**Contested.** {{Statement}}. [[source-a]] reports X (E1); [[source-c]] reports
the opposite (E2). {{Why they disagree, if the vault worked it out.}}

**Not addressed.** {{The gap this project went after, and the closest work to
it.}}

## What we tested

Pre-registered as [[spec]] on {{date}}, **before any run existed** — the hash in
its front matter covers the decision rule, so the rule below is provably the one
committed to in advance.

> {{The decision rule, quoted verbatim from the spec.}}

Design: {{arms}}, {{seeds}} seeds, {{scale}} scale, baseline {{baseline}}.
Controls: {{the specific ones}}.

## What we found

| Arm | n | Mean | SD |
|---|---|---|---|
| {{random}} | 3 | 0.6021 | 0.0041 |
| {{curriculum}} | 3 | 0.7015 | 0.0035 |

Paired by seed: **{{+0.0994}}**, 95% CI [{{+0.0871}}, {{+0.1117}}], p = {{0.0012}}.

Full detail in [[results]]. Runs: {{6}} of {{6}} completed{{, or: 6 of 8; two
OOMed on the treatment arm, which is itself a finding}}.

## What this does not show

- {{Scope limit from the brief that still binds}}
- {{Every major finding red-team raised}}
- {{The claim a reader might take away that this does not support}}

{{This section is not a disclaimer. It is what makes everything above it
trustworthy.}}

## What would settle it

{{The next experiment, honestly costed, and what it would resolve. If the answer
is "nothing further is worth running", say that.}}

## Sources

{{Every source note and run cited above. Generated from the vault, not from
memory — a reference list is the easiest place for a fabricated citation to
survive review.}}

- [[arxiv-2312.00752]] — {{Authors, year, venue}}
- [[curriculum-seed0]] — run, exp-001
