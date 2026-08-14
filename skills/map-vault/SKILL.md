---
name: map-vault
description: Build the navigable layer over a knowledge vault — a map of content, a glossary, and a method-by-dataset coverage matrix that exposes what is settled, what is contested, and where the gap is. Use after building or extending a vault, or when the user asks what the literature says, where the open question is, or for an index, glossary, or concept map.
user-invocable: true
---

# map-vault

A vault of good notes is still a pile until something says what it *adds up to*.
This skill builds that layer — and its real job is finding the hole worth
running an experiment in.

Everything here is derived. Delete it and run again; if the rebuild differs, the
old version was stale and the files were right.

## What you produce

- `map.md` — settled / contested / gaps, the map of content
- `glossary.md` — every concept node, one line each
- `_meta/matrix.md` — method × dataset coverage, the gap finder

## Steps

### 1. Read the whole vault

`sources/`, `concepts/`, `brief.md`. Not the previous `map.md` — regenerate from
the notes, or you will inherit last round's conclusions and quietly stop
noticing that the evidence moved.

### 2. Sort the claims

Every claim node lands in one of four buckets, and the bucket is decided by the
evidence, not by how confident the papers sound:

- **Settled** — several independent sources agree, nothing contradicts. Testing
  it again wastes the project.
- **Contested** — sources disagree. Write down *why* you think they disagree:
  different datasets, different baseline tuning, different definition of the
  quantity. This paragraph is usually the most valuable thing in the vault.
- **Open** — the question is asked but not answered anywhere you found.
- **Refuted** — the literature settles it in the negative. If the brief's claim
  lands here, say so loudly: that is the project's answer, arrived at for the
  price of a survey.

"Independent" means different groups, different data. Three papers from one lab
reusing one benchmark are one piece of evidence, and treating them as three is
how a field convinces itself of something.

### 3. Build the coverage matrix

Rows = methods, columns = datasets (or scales, or context lengths — whatever the
brief's axis actually is). Each cell holds the source notes that cover it.

**The empty cells are the point.** A method-by-dataset matrix with a hole in it
is a candidate experiment, and it is a far better basis for one than a hunch.
Write the matrix even when it is mostly empty — a sparse matrix says the field
is young, which is itself worth knowing.

### 4. Write the map

`references/map-template.md`. Every statement links the claim node and the
source notes behind it. `map.md` asserts nothing new; it is an index over
things already written down, which is what makes it safe to regenerate.

Close with **the open question this vault exposes**, stated in the form
`design-experiment` can pick up: a comparison, a setting, a metric, and what
result would settle it.

### 5. Assess the brief's novelty, honestly

Compare the brief's claims against what you just sorted. Then answer, in
`map.md`:

> Does the literature already answer this?

If yes — say so. Do not manufacture a gap to justify continuing. A survey that
ends "this is settled, here is the citation, here is the sharper question next
door" is a successful project that cost a day instead of a month.

### 6. Hand off

Update the glossary and `README.md`, append to `log.md`, then run `red-team` on
the map before anything is designed. Report the settled/contested/open counts,
the gap, and the novelty verdict.

## Getting this wrong

**Settling something on one paper.** One source is not a consensus. Mark it
open and say what would settle it.

**Absence of contest treated as agreement.** If only one group has ever looked,
the claim is *open*, not settled. The distinction is the difference between a
field that agrees and a field nobody has checked.

**A gap that is a gap because nobody cares.** Empty cells are candidates, not
conclusions. Ask why it is empty — often the answer is that the combination is
meaningless, and that is worth a sentence rather than an experiment.

**Novelty by assertion.** "No prior work does exactly this" is nearly always
false and always unfalsifiable. Name the closest thing that exists and say
precisely how the brief differs from it.
