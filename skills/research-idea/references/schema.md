# Vault schema

This file travels with the vault, at `_meta/schema.md`. It is the contract every
skill reads before writing a note, and the thing `lint-vault` checks against.
If you change the schema, change it here — a skill that disagrees with this file
is the bug.

The vault is plain Markdown with YAML frontmatter and `[[wikilinks]]`. That is
not decoration: it is why the folder opens in Obsidian as a graph with no import
step, why `grep` works, and why nothing here depends on a database that can drift
out of sync with the files. **The files are the truth. Every index, map and
report is derived and may be rebuilt from them at any time.**

## Layout

```
README.md          entry point and status
brief.md           the falsifiable question
log.md             append-only journal
TODO.md            open gaps, written by lint-vault
sources/           one note per paper or repo that was read
concepts/          methods, datasets, metrics, and claims as nodes
experiments/<id>/  spec.md, runs/, results.md
reports/           memos
raw/               dossiers exactly as deep research returned them
_meta/schema.md    this file
```

## Universal frontmatter

Every note carries these. `lint-vault` fails a note that omits one.

```yaml
type: source | concept | method | dataset | metric | claim | spec | run | result | report
title: Human-readable title
added: 2026-08-14          # ISO date, the day the note was created
tags: [type, topic, ...]   # first tag is always the type
```

## `sources/` — one note per thing that was read

Filename: the citation key with `:` and `/` replaced by `-`
(`arxiv-2312.00752.md`). Stable, so links do not break when a title is fixed.

```yaml
---
type: source
key: arxiv:2312.00752
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
authors: [Albert Gu, Tri Dao]
year: 2023
venue: COLM 2024
url: https://arxiv.org/abs/2312.00752
code: [https://github.com/state-spaces/mamba]
citations: 4200            # omit entirely if no page stated it — never estimate
read: full                 # full | partial | abstract-only | unreachable
confidence: 0.8            # confidence in the reading, not in the paper
raw: raw/arxiv-2312.00752.md
added: 2026-08-14
tags: [source, ssm, long-context]
---
```

Body sections, in this order: `## Summary`, `## Method`, `## Setup`,
`## Results`, `## Limitations`, `## Relevance`, `## Evidence`.

**`## Evidence` is not optional.** It holds the verbatim quotes every other
section rests on, as blockquotes with a locator:

```markdown
## Evidence

> Mamba reaches 1.02 bits-per-byte at 8k context, against 1.09 for the
> transformer baseline.
— §4.2 Results, p.7 · [source](https://arxiv.org/abs/2312.00752)
```

Any number, comparison or claim elsewhere in the note must appear in a quote
here. A number with no quote behind it is a fabrication with a citation stapled
to it, which is worse than an uncited one because it looks checked.

`read: abstract-only` is a valid and useful state. `read: unreachable` means the
note exists to record that the source could not be obtained — it has no
`## Results`, and it belongs in `TODO.md`.

## `concepts/` — the graph's nodes

Filename: the human name (`Mamba.md`, `The Pile.md`). Obsidian's shortest-path
links then read naturally in prose.

```yaml
---
type: method            # method | dataset | metric | concept
title: Mamba
aliases: [selective state space model, S6]
added: 2026-08-14
tags: [method, sequence-model]
---
```

Body: what it is in two or three sentences, then `## Seen in` linking every
source note that uses it. Concepts are how the graph becomes navigable — a
source note that links no concepts is an island, and `map-vault` will say so.

## `concepts/` — claims are first-class

A claim is an assertion the literature makes, kept as its own node so that
disagreement is representable:

```yaml
---
type: claim
title: "SSMs beat transformers beyond 32k context"
status: contested        # settled | contested | open | refuted
supported_by: ["[[arxiv-2312.00752]]", "[[arxiv-2302.10866]]"]
contradicted_by: ["[[arxiv-2405.11111]]"]
added: 2026-08-14
tags: [claim]
---
```

**A `contradicted_by` edge is a finding, not an error.** Two papers disagreeing
is the single most useful thing a survey can surface: it is a candidate
replication, and often the experiment worth running. Never resolve a
contradiction by deleting one side.

## `experiments/<id>/spec.md` — pre-registration

```yaml
---
type: spec
id: exp-001
title: Curriculum vs random ordering at smoke scale
spec_hash: 8f3ad2...          # written by spec_hash.py, covers the fields below
status: preregistered          # preregistered | implemented | running | complete
hypothesis: "Curriculum ordering improves accuracy at matched compute"
independent_variables:
  ordering: [random, curriculum]
arms: [random, curriculum]
baseline: random
dataset: {name: toy, split: test}
metrics: [accuracy]
controls: [matched compute, identical data budget, same seeds]
seeds: [0, 1, 2]
scale: smoke                   # smoke | small | main
decision_rule:
  metric: accuracy
  comparator: ">"
  threshold: 0.02
  min_seeds: 3
  max_p_value: 0.05
  refutes_if: "The 95% CI of the difference sits entirely below 0.02"
preregistered_at: 2026-08-14T10:00:00Z
added: 2026-08-14
tags: [spec]
---
```

`decision_rule` and `refutes_if` are written **before a single run exists**.
This is the highest-value constraint in the system: it turns a negative result
into a finding rather than a failure, and removes the incentive to rationalise
whatever came out. `spec_hash` covers the rule, so an edit after the fact is
detectable — and `lint-vault` looks for exactly that.

## `experiments/<id>/runs/<arm>-seed<n>.md` — one run

```yaml
---
type: run
id: exp-001/curriculum-seed0
spec_hash: 8f3ad2...
code_hash: 91bb0c...
env_hash: 4d21fe...
arm: curriculum
seed: 0
status: ok                     # ok | oom | nan | hang | diverged | error | skipped
metrics: {accuracy: 0.7031}
runtime_seconds: 41
added: 2026-08-14
tags: [run]
---
```

**The identity of a run is `(spec_hash, code_hash, env_hash, arm, seed)`.** The
arm belongs in it because one spec runs several arms at the same seed; without
it, baseline and treatment share a key and every experiment looks
irreproducible against itself.

## `experiments/<id>/results.md` — the verdict

Frontmatter holds the numbers `stats.py` computed and the verdict `decide.py`
returned; the body explains them and lists threats to validity. Both are written
by script rather than by hand, because the whole point of pre-registering the
rule is that applying it is not a judgment call.

## Linking rules

- Link with `[[note-name]]`, not Markdown links — Obsidian's graph is built from
  wikilinks, and this is what makes the vault a graph rather than a folder.
- Link the shortest unambiguous name (`[[Mamba]]`, not `[[concepts/Mamba]]`).
- A source note links every concept it touches. A concept links back via
  `## Seen in`. Backlinks make the graph traversable in both directions.
- Never link to a note you have not created. A broken link is a lie about what
  the vault contains, and `lint-vault` reports every one.

## What is derived, and may be deleted at any time

`README.md`'s status table, `TODO.md`, any index or map-of-content page, and
everything in `reports/`. All of it is rebuildable from `sources/`, `concepts/`,
`experiments/` and `raw/`. If rebuilding produces something different, the
derived file was stale — trust the files.
