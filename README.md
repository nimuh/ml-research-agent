# ml-research-agent

A set of Claude Code skills that take a research idea, find out what is already
known, test what is not, and write up what the evidence actually supports —
leaving behind an Obsidian vault you can open, read and navigate as a graph.

```
/research-idea does curriculum ordering help small-model math reasoning?
```

That runs the whole loop. Each step is also a skill you can invoke on its own.

## Install

```bash
git clone <this repo> && cd ml-research-agent
./install.sh          # symlinks skills/ into ~/.claude/skills
```

Then start a new Claude Code session — skills are discovered at startup.

Nothing else to install. No API key, no package, no virtualenv: the skills use
Claude Code's own web search and file tools, and the four Python scripts they
call are stdlib-only. `./install.sh --uninstall` removes exactly the links it
made and touches nothing else.

## What you get

A folder that is a research project and an Obsidian vault at the same time:

```
curriculum-ordering/
  README.md              the question, and where it stands
  brief.md               claims, falsifiers, success criteria
  sources/               one note per paper, every claim carrying a quote
  concepts/              methods, datasets, metrics and claims as graph nodes
  experiments/           pre-registered specs, run records, results
  reports/               the memo
  raw/                   verbatim dossiers — nothing edits these
  log.md · TODO.md       what ran; what is still missing
```

Open the folder in Obsidian (`File → Open vault`) and it works immediately —
wikilinks, backlinks, and a graph coloured by note type. There is no export
step because there is no database: **the Markdown files are the truth**, and
every index, map and report is derived from them.

## The eleven skills

| Skill | What it does |
|---|---|
| `research-idea` | Runs the whole loop, scaffolds the vault, stops at the checkpoints |
| `frame-idea` | Turns an idea into claims with explicit falsifiers |
| `survey-literature` | Deep research across the web; writes verbatim dossiers |
| `build-vault` | Screens sources, writes linked notes, extracts the graph |
| `map-vault` | Settled / contested / gaps, glossary, coverage matrix |
| `lint-vault` | Broken links, unsupported numbers, tampered pre-registrations |
| `design-experiment` | Pre-registers a spec with a decision rule, and hashes it |
| `run-experiment` | Implements it, runs the smoke rung first, records every run |
| `analyze-results` | Seed statistics and the pre-registered verdict, by script |
| `red-team` | Attacks the artifact: leaky evals, unfair baselines, bad citations |
| `write-memo` | The report, where every sentence cites a note or a run |

## What makes it different from asking for a literature review

Five constraints, each aimed at a specific way this kind of system goes wrong.

**Every claim traces to a quote.** Source notes carry an `## Evidence` section
of verbatim passages with locators, copied from dossiers that are never edited.
A number in a note that appears in no quote is a lint blocker. You can follow
any sentence in the memo back to the line on the page.

**The decision rule is written before the experiment runs — and hashed.**
`spec_hash.py` covers the hypothesis, the arms, the controls and the rule, so a
threshold adjusted after the numbers came in is *detectable*. `lint-vault`
reports it as a blocker. This is what makes a negative result a finding instead
of a failure.

**A clean negative is reported as a negative.** `decide.py` returns `refuted`
when the confidence interval sits entirely below the pre-registered threshold —
not `inconclusive`. Without that, an experiment that genuinely rules an effect
out reads identically to one too noisy to tell.

**Statistics are computed, not estimated.** Paired-by-seed differences,
confidence intervals and a t-test in ~200 lines of stdlib Python. Three seeds
minimum; a single-seed delta is inconclusive by construction.

**Disagreement is preserved.** When two papers contradict each other the vault
records both sides and marks the claim contested. That edge is usually the
experiment worth running, and averaging it away is the most expensive mistake a
survey can make.

## The deterministic core

Four scripts, because these are the things a language model does unreliably and
confidently:

| Script | Why it is not prose |
|---|---|
| `design-experiment/scripts/spec_hash.py` | A pre-registration is only worth something if editing it is detectable |
| `analyze-results/scripts/stats.py` | Reading a mean off three numbers is where a model is least trustworthy |
| `analyze-results/scripts/decide.py` | Applying a pre-registered rule must not be a judgment call |
| `lint-vault/scripts/lint_vault.py` | Counting links across 200 files, and comparing a hash to the bytes it covers |

```bash
python3 tests/test_scripts.py     # 40 tests, stdlib, no network
```

## Using the steps separately

Each skill works on its own, on any vault:

```
/survey-literature   state space models for genomic sequence modelling
/lint-vault          ./curriculum-ordering
/red-team            ./curriculum-ordering/experiments/exp-001/spec.md
/write-memo          ./curriculum-ordering
```

`lint-vault` and `red-team` are useful against a vault you built by hand, or one
that a previous session left half-finished.

## Design

`docs/DESIGN.md` is the contract: the phase machine, the vault schema, and the
invariants that constrain what any skill is allowed to do. Read it before
changing a skill — several of the rules that look like style are load-bearing.
