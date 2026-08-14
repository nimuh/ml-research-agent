# Design

The contract between the skills. Each `SKILL.md` is its own spec; this document
says how they fit together and which of their rules are load-bearing.

## 1. What this is

Eleven Claude Code skills that take a research idea and produce an Obsidian
vault containing what is known, what was tested, and what the evidence supports.

The system it replaced was a 9,000-line Python package: an LLM client with model
tiers and a budget tracker, six literature source adapters, a hybrid vector/BM25
index, an agent framework with structured-output repair loops, and a ten-phase
orchestrator driving it all. Almost all of that was scaffolding for capabilities
Claude Code already has — web research, file manipulation, subagents,
long-context reading — and the scaffolding was where the failures lived. The
last recorded run of the old system spent most of its budget absorbing HTTP 429s
from OpenReview and halted at a human gate having produced nothing.

What survived the conversion is what was never about calling a model: a schema
for the notes, a phase order with checkpoints, and four scripts doing arithmetic
and hashing.

## 2. The loop

```
FRAME ──▶ SURVEY ──▶ BUILD ──▶ MAP ──▶ DESIGN ──▶ RUN ──▶ ANALYZE ──▶ MEMO
                        │                 ▲          │
                        └── red-team ─────┘          │
                            at 4 gates    inconclusive
```

| Phase | Skill | Exit criterion |
|---|---|---|
| Frame | `frame-idea` | ≥1 claim with a stated falsifier and a measurable threshold |
| Survey | `survey-literature` | Named seminal works located, or their absence recorded |
| Build | `build-vault` | Every note's assertions carry a quote; every dossier screened |
| Map | `map-vault` | Claims sorted; the gap named as a testable comparison |
| Design | `design-experiment` | Decision rule fixed and hashed before any run exists |
| Run | `run-experiment` | Every arm × seed completed or failed with a category |
| Analyze | `analyze-results` | The rule applied by script; the verdict is what it returned |
| Memo | `write-memo` | Every sentence cites a source note or a run id |

`red-team` runs after Build, Map, Design and Analyze. A blocker stops the phase;
on a result it forces the verdict to `inconclusive` regardless of the numbers.

Three checkpoints hand control back to the user: after Build (*is this the right
literature?*), before Run (*approve the compute*), and before Memo only when a
hypothesis has come back inconclusive twice.

## 3. State lives in the vault

Skills are stateless. A session ends and everything it knew is gone — so the
vault is the memory, and it is designed to be read by a person, by Obsidian, and
by the next session, with no format privileging any of the three.

`log.md` is append-only and records what each phase did. `README.md` carries the
status table. Resuming means reading those two and continuing from the next
phase. `_meta/schema.md` travels *inside* each vault rather than living here,
which is what lets a vault be moved, shared, or picked up by a skill that has
never seen this repository.

The full schema is `skills/research-idea/references/schema.md`.

## 4. Invariants

These constrain what a skill may do. Violating one is a design regression, not a
style preference — each is here because of a specific way this kind of system
produces confident, wrong output.

**Traceability.** Every assertion traces to a verbatim quote with a locator, or
to a run id. `## Evidence` is mandatory in source notes; a number appearing in
no quote is a lint blocker. The Memo may only assert what a note or a run backs.

**Files are truth, indexes are derived.** Markdown and wikilinks, no database.
`map.md`, `glossary.md`, `TODO.md` and everything in `reports/` are rebuildable;
if a rebuild disagrees with what is there, what is there is stale.

**Retrieval is not relevance.** `survey-literature` returns everything it found
and never screens; `build-vault` decides inclusion and records a reason both
ways. Keeping them apart is what lets recall and precision be tuned separately,
and what keeps the corpus boundary auditable.

**Pre-registration is enforced, not requested.** The decision rule is written
before any run exists and hashed by `spec_hash.py`. `run-experiment` verifies
the hash before executing and refuses if it fails. Changing a design means a new
experiment id, never an edit and a re-hash.

**A run is `(spec_hash, code_hash, env_hash, arm, seed)`.** The arm belongs in
the key because one spec runs several arms at the same seed; without it the
baseline and the treatment collide and every experiment looks irreproducible
against itself.

**Cheap before expensive.** `smoke → small → main`. Smoke checks the harness and
answers nothing; a failed smoke stops the ladder. Reference numbers from the
literature are reproduced at smoke scale, because a broken harness caught there
costs minutes instead of a conclusion.

**Three seeds minimum.** A single-seed delta is inconclusive by construction.
The seed floor is checked before anything else in `decide.py`, so the
equivalence check cannot smuggle a verdict past a sample of two.

**A clean null is `refuted`.** When the confidence interval sits entirely below
the pre-registered threshold, the effect is ruled out and that is the verdict.
A significance test alone cannot separate "we ruled it out" from "we could not
tell," and collapsing them is what makes negative results feel like failures.

**Judge against the worst baseline.** With several baselines, the verdict is
decided on the one the treatment does worst against — otherwise the order they
were listed in decides the outcome, and adding a weak baseline manufactures a
win.

**Disagreement is data.** A `contradicted_by` edge is a finding. Contested
claims are never resolved by deleting a side.

**Bounded, and logged when it bites.** Source caps, turn ceilings, passage
limits. Every truncation is written to the log. A silently truncated corpus
looks exactly like a complete one.

**Cloning is not running.** Third-party code executes only inside the workspace
`run-experiment` builds, never at fetch time. Licences are checked before
adaptation.

## 5. The deterministic core

Four scripts, each replacing something a language model does unreliably:

| Script | Replaces |
|---|---|
| `spec_hash.py` | Trusting that a decision rule was not edited |
| `stats.py` | Eyeballing three numbers and forming a view |
| `decide.py` | Interpreting a threshold after seeing the result |
| `lint_vault.py` | Checking 200 files by hand and missing the same ones twice |

Stdlib only, so a skill can call them with bare `python3`. Student's t is
implemented in `stats.py` rather than imported, for the same reason.

`spec_hash.py` and `lint_vault.py` **duplicate** the payload computation, because
an installed skill has to stand alone. `tests/test_scripts.py` hashes one fixture
through both and compares the whole payload — that test exists because an earlier
version compared only part of it, the two parsers silently disagreed, and the
linter reported every honest spec as tampered. A check that fires on correct
input is worse than no check.

## 6. Testing

`python3 tests/test_scripts.py` — 40 tests, stdlib, offline.

Four groups, and the third is the one that matters:

- **Hashing** — the same spec hashes stably; editing the *rule* changes it;
  editing prose, `status`, or whitespace does not. A spec whose every edit looks
  like tampering trains everyone to ignore the check.
- **Statistics** — a clear effect, pure noise that has different means, the seed
  floor, failed runs excluded from the mean but counted, and Student's t checked
  against published table values.
- **Adversarial** — the cases the system exists to catch: a number with no quote
  behind it, a note claiming to have read more than its dossier did, a spec
  edited after registration, a claim marked settled while still contradicted, a
  significant effect below the pre-registered threshold reported as a
  refutation rather than a win.
- **The skills themselves** — every `SKILL.md` parses, `name` matches its
  directory, every description carries a trigger clause, every referenced script
  exists and runs.

## 7. Layout

```
skills/<name>/
  SKILL.md         frontmatter + instructions; the whole spec
  scripts/         stdlib Python, only where determinism is required
  references/      templates the skill writes into a vault
install.sh         symlinks skills/ into ~/.claude/skills
tests/             stdlib unittest over the scripts and the skill files
docs/DESIGN.md     this file
```

## 8. Changing a skill

Its `SKILL.md` is the spec — change it deliberately, and check whether §4 needs
the same edit. Rules of thumb that have earned their place:

- **Say when to use it, in the words a user would use.** A description that only
  says what the skill *is* never gets triggered. A test enforces this.
- **Say why, not just what.** Every instruction that looks arbitrary will be
  reasoned around by a capable model unless the reason is on the page.
- **Name the failure.** Each skill ends with the ways it goes wrong. Those
  sections do more work than the instructions above them.
- **Do not add a script for something Claude does well.** The four that exist do
  arithmetic, hashing and counting. Prose extraction, judgment and search do not
  belong in Python — that is the mistake this repository was built out of.
