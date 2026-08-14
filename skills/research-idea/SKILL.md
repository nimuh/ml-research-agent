---
name: research-idea
description: Take a research idea and run the whole loop on it — frame it as a falsifiable question, survey the literature with deep research, build an Obsidian knowledge vault, design and run pre-registered experiments, and write a memo backed only by what the vault and the runs actually show. Use when the user gives you an idea, hypothesis, or research question to investigate, or asks to continue/resume an existing research vault.
argument-hint: "<your research idea>"
user-invocable: true
---

# research-idea

You are running a research project end to end. The user gave you an idea; you
will turn it into a question that can be *wrong*, find out what is already
known, test what is not, and report what the evidence supports — including when
the answer is "no" or "we could not tell."

This skill is the conductor. Each phase below is a separate skill with its own
instructions; invoke it rather than improvising the work here.

## The one rule

**Every claim you end up making must trace to a quote in the vault or a run id
in the vault.** Not to your memory of a paper, not to what is probably true.
If you cannot point at the line, you do not write the sentence. Everything else
in this system exists to make that rule survivable.

## The loop

```
FRAME ──▶ SURVEY ──▶ BUILD ──▶ MAP ──▶ DESIGN ──▶ RUN ──▶ ANALYZE ──▶ MEMO
                                 │                          │
                                 └──── red-team at ─────────┘
                                    each ▲ gate      inconclusive
                                                     loops to DESIGN
```

| # | Phase | Skill | Done when |
|---|-------|-------|-----------|
| 1 | Frame | `frame-idea` | The idea is ≥1 claim with a stated falsifier and measurable success criteria |
| 2 | Survey | `survey-literature` | Deep research has read the field; named seminal works were found or their absence reported |
| 3 | Build | `build-vault` | Every source read has a note whose every assertion carries a quote |
| 4 | Map | `map-vault` | Concepts, claims and the gap matrix are linked; the open question is visible |
| 5 | Design | `design-experiment` | A spec exists with controls, baselines, seeds, and a decision rule fixed *before* running |
| 6 | Run | `run-experiment` | Every arm × seed completed or failed with a recorded, categorised reason |
| 7 | Analyze | `analyze-results` | The pre-registered rule was evaluated by script, not by argument |
| 8 | Memo | `write-memo` | Every sentence cites a source note or a run id |

`red-team` runs after Build, after Map, after Design, and after Analyze. It is
prompted to refute, not to approve. Take its blockers seriously: a blocking
finding downgrades a verdict to inconclusive whatever the arithmetic said.

## Steps

### 0. Set up or resume the vault

The vault is a plain folder of Markdown that opens directly in Obsidian. It is
also the entire memory of this project — you are stateless between sessions,
the vault is not.

1. Slugify the idea into a short kebab-case name (`mamba-long-context-genomics`).
   If the user named a folder, or you are inside a vault already (a directory
   containing `_meta/schema.md`), use that instead and **resume** rather than
   scaffold.
2. Create the structure:

```
<slug>/
  README.md              entry point: the question, the state, links out
  brief.md               the falsifiable brief
  log.md                 append-only journal of what ran and what it decided
  TODO.md                open gaps
  sources/               one note per paper
  concepts/              methods, datasets, metrics, claims as nodes
  experiments/           one folder per experiment
  reports/               memos
  raw/                   dossiers, verbatim, never hand-edited
  _meta/schema.md        the frontmatter and link contract
  .obsidian/             config, so the graph looks right on first open
```

3. Copy `references/schema.md` from this skill to `<slug>/_meta/schema.md`, and
   `references/obsidian/` to `<slug>/.obsidian/`. Every other skill reads the
   contract from the vault, which is what keeps the vault portable and the
   skills decoupled.
4. Write `README.md` from `references/readme-template.md`.

To resume: read `log.md` and `README.md`, work out which phase last completed,
and continue from the next one. Never redo a completed phase silently — if you
think it needs redoing, say why and ask.

### 1..8 Run the phases

Invoke each skill in order. After each one:

- Append one entry to `log.md`: the phase, the date, what it produced, what it
  cost in wall-clock, and any truncation or gap it recorded.
- Update the **Status** table in `README.md`.
- Stop and report to the user at the checkpoints below.

### Checkpoints — stop and ask

Three places where you hand control back rather than pressing on:

1. **After Survey + Build.** Show the user how many sources the vault holds, the
   titles, what the search could not reach, and any seminal work that never
   surfaced. Ask: *is this the right literature?* A vault built on the wrong
   corpus fails silently forever afterwards.
2. **Before Run.** Show the specs, the arm × seed count, the estimated wall-clock
   and any cost, and the decision rule that has been pre-registered. Ask for
   approval before spending compute.
3. **Before Memo**, only if the verdict is inconclusive twice on the same
   hypothesis. Two inconclusive rounds means the design is not answering the
   question; more compute will not fix that, and the user should choose.

Otherwise keep going without asking permission at every step.

## Things that will go wrong, and what to do

**The literature already answers the question.** Good — that is a result, and a
cheap one. Say so plainly, cite it, and offer the user the sharper question the
survey exposed. Do not invent novelty to justify running an experiment.

**No runnable reference implementation exists.** Document the gap in the vault
and design around it. "Cloning is not running": third-party code only ever
executes through `run-experiment`, which sandboxes it.

**A source is paywalled or unreachable.** Record it as unreachable in the note's
`read:` field and in `TODO.md`. An abstract-only note that says it is
abstract-only is useful. One dressed up as a full reading corrupts everything
downstream.

**The experiment is inconclusive.** That is a real outcome. Report the effect,
the spread, and what it would take to tell — then either propose the cheapest
follow-up with the best information gain, or stop. Do not rerun until it looks
positive.

**You are tempted to write a sentence you cannot cite.** Delete the sentence.
