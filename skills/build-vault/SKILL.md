---
name: build-vault
description: Turn raw research dossiers into a linked Obsidian knowledge vault — screen sources against the brief, write one note per paper with every assertion backed by a quote, and extract methods, datasets, metrics and claims as graph nodes. Use after surveying literature, or when the user wants notes, a knowledge base, or an Obsidian vault built from collected material.
user-invocable: true
---

# build-vault

Turn `raw/` into a knowledge graph: screened source notes, concept nodes, and
claim nodes wired together with `[[wikilinks]]`.

Read `_meta/schema.md` in the vault first. It is the contract — frontmatter
fields, section order, filenames, linking rules — and this skill assumes it.

## Steps

### 1. Screen, and record the reason either way

For each dossier in `raw/`, decide include or exclude against `brief.md`'s
claims and scope limits. Write every decision — both directions — to
`_meta/screening.md` as a table: key, decision, one-line reason.

The exclusions matter as much as the inclusions. They are what makes the
vault's boundary auditable instead of arbitrary, and six months from now they
are the only record of why an obvious-looking paper is not here.

Cap the vault at ~40 included sources unless the user asked for more. If the
cap bites, **say so in the log** with how many were dropped and on what
ranking. A silently truncated corpus looks identical to a complete one.

### 2. Write one source note per included paper

`sources/<key>.md`, from `references/source-note.md`. The mechanical part:
copy frontmatter across from the dossier, keep `read:` honest.

The part that matters:

**Every assertion in the note traces to a quote in its own `## Evidence`
section.** Write `## Evidence` first, pasting the dossier's quotes with their
locators, then write the analytical sections above it — and as you write each
number or comparison, point it at the evidence that carries it:

```markdown
## Results

Reaches 1.02 bits-per-byte at 8k context against a 1.09 transformer
baseline (E1), at equal parameter count.
```

If you find yourself writing something no quote supports, you have two honest
options: go back to `raw/` and find the quote, or delete the sentence. Adding
it anyway is how a vault becomes confidently wrong.

`## Relevance` is where you say what this paper does for *this brief* — including
"barely anything, but it is the standard baseline everyone compares against,"
which is a real reason to keep a note.

### 3. Extract the graph

While writing each source note, create or update a node in `concepts/` for
every method, dataset and metric it uses. Link both ways: the source note links
`[[Mamba]]`, and `Mamba.md` lists the source under `## Seen in`.

This is what makes the vault a graph rather than a folder of PDFs-as-prose. A
source note that links no concepts is an island; `map-vault` will flag it and
it will sit alone in Obsidian's graph view, which is usually the visual
signal that a note was written without being read.

Keep concept names human (`concepts/The Pile.md`) so links read naturally in
prose. Record `aliases:` for the other names the field uses — that is what
stops the same idea becoming three nodes.

### 4. Extract claims

Every substantive assertion the literature makes becomes a claim node, with
`supported_by` and `contradicted_by` pointing at source notes.

**When two sources disagree, record both sides and set `status: contested`.**
Do not average them, do not pick the more recent one, and never resolve a
disagreement by deleting a side. A contested claim is the single most useful
thing a survey produces: it is a candidate replication and usually the
experiment worth running. `map-vault` looks for exactly these.

### 5. Check the work before handing off

- Every note has the frontmatter `_meta/schema.md` requires.
- Every `[[link]]` resolves to a file that exists.
- Every number in a note appears in a quote in its `## Evidence`.
- Every dossier is either an included note or an exclusion with a reason.

Then run `lint-vault`, which checks all of this mechanically and writes what it
finds to `TODO.md`. Append to `log.md`, update `README.md`, and report to the
user: sources included, excluded, abstract-only, and contested claims found.

## Getting this wrong

**Notes that summarise the abstract.** If a note could have been written
without opening the paper, it was — and the vault is now a pile of abstracts
with citations attached. The `## Evidence` requirement exists to make this
visible: an abstract-only note has only abstract quotes, and it says
`read: abstract-only`.

**Quiet upgrading.** A dossier marked `abstract-only` produces a note marked
`abstract-only`. Never let the note claim more than the dossier read.

**Concept sprawl.** "State space model", "SSM" and "S4/S6" as three nodes make
the graph useless. One node, `aliases:` for the rest.

**Deleting the disagreement.** See step 4. This is the failure that costs the
most and is the easiest to commit, because a tidy vault feels finished.
