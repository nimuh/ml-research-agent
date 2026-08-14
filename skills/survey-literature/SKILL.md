---
name: survey-literature
description: Survey the literature on a research question using deep research — parallel web search and page fetching across arXiv, OpenReview, Semantic Scholar and the open web, snowballing citations until the yield dries up, and saving verbatim dossiers of what was actually read. Use when the user wants a literature review, prior work, related work, or "what's already known about X".
argument-hint: "<question or path to a vault>"
user-invocable: true
---

# survey-literature

Go and read the field. This is the only acquisition path the system has: if you
do not find a paper here, the project never sees it.

You have web search, page fetching, and subagents. Use all three. The output is
not a summary — it is a folder of **dossiers containing quotes you actually
read**, from pages you actually opened, which everything downstream cites.

## Your job is recall, not judgment

Do not filter for relevance. Do not rank. Do not skip a paper because it looks
unpromising. `build-vault` screens next, with its own criteria and a recorded
reason either way — duplicating that decision here hides it from the audit
trail and, worse, makes the corpus depend on a snap judgment nobody logged.

The failure mode you are guarding against is a survey that misses the obvious
prior work. Err toward breadth.

## Steps

### 1. Read the brief

`brief.md` gives you the claims, the search vocabulary, the seminal-works
checklist, the scope limits and the date window. If there is no brief, run
`frame-idea` first — surveying an unframed idea produces a pile of papers nobody
can screen.

Check `sources/` for what the vault already holds. Do not re-read those.

### 2. Fan out

Spawn subagents (the Task tool) — one per sub-topic, per named seminal work to
locate, per adjacent literature. Give each a narrow assignment and the same
instruction set: search, open pages, report only what you read. Parallel breadth
is the point; a single agent running queries in sequence is a slow version of
the fan-out this replaces.

Aim for roughly 40–60 distinct papers before dedup, unless the brief says
otherwise. Bound it: stop a thread when it stops returning work you have not
already seen, and **write down where you stopped**.

Search properly:

- **Vary the vocabulary.** The field's current term, the older name, the term
  the adjacent subfield uses. One phrasing finds one community.
- **Go to the sources directly** — `arxiv.org` listings, `openreview.net` venue
  pages, `semanticscholar.org`, `paperswithcode.com`, authors' own pages. A
  generic web search alone will find you blog posts about the paper rather than
  the paper.
- **Snowball.** Open the reference lists of the strongest hits and follow the
  citations that recur across several of them — recurrence is the signal. Then
  look at what cites them, which is how you find the work that supersedes the
  paper you started from.
- **Check the checklist explicitly.** Search for each named seminal work *by
  title* before finishing. Hoping it turns up is not checking.

### 3. Write a dossier per paper

One file per paper in `raw/`, named for its citation key
(`raw/arxiv-2312.00752.md`). This is the immutable record — nothing later edits
it, and every note in the vault is rebuildable from this folder.

Use `references/dossier-template.md`. What matters:

- **Quotes are copied, character for character**, from the page in front of you.
  Never paraphrase into a quote. Never reconstruct a sentence from memory of the
  paper. Never quote a paper you did not open.
- **Every quote gets a locator** — the section heading as printed, or the table
  or figure number, plus a page number when the source has pages — and the URL
  it came from.
- **Quote what carries information**: the method as the authors describe it, the
  experimental setup, headline numbers *with their conditions*, and the
  limitations paragraph most readings skip. Skip boilerplate.
- **`citations:` only if a page stated it.** That number feeds ranking later; an
  invented one is worse than a missing one. Omit the field.
- **If the full text is unreachable**, set `read: abstract-only` or
  `read: unreachable`, work from what you can genuinely read, and say what that
  leaves unsupported. An abstract-only dossier that says so is useful. One
  dressed up as a full reading corrupts the vault and every claim built on it.

### 4. Report what you could not reach

Write `raw/_coverage.md`: the queries you ran, the sites you consulted, where
each thread bottomed out, what was paywalled or blocked, and — checked against
`brief.md` — **which named seminal works never surfaced**.

A gap that is written down is a finding. A gap that is papered over is a
fabrication with extra steps.

### 5. Hand off

Append to `log.md`, update `README.md`, and tell the user: how many papers, the
titles, what was unreachable, and any seminal work that did not appear. Then run
`build-vault` to turn the dossiers into linked notes.

## Getting this wrong

**Constructing an identifier.** An arXiv id or DOI assembled from a pattern
rather than read off a page is the most common fabrication in this whole system,
and it is invisible until someone clicks the link. Copy identifiers; never
derive them.

**Summarising instead of quoting.** If the `## Evidence` section of a dossier
contains your prose rather than the paper's, the dossier is worthless — every
downstream citation would point at something you wrote.

**Stopping at page one.** Three queries and eight papers is not a survey. If the
fan-out came back thin, say so explicitly rather than presenting it as complete.

**Reading only what agrees.** Actively search for work that contradicts the
brief's claims. A `contradicted_by` edge in the vault is the most valuable thing
you can bring back — it is usually the experiment worth running.
