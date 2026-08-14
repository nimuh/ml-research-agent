---
name: run-experiment
description: Implement and execute a pre-registered experiment — write the code, run the smoke rung first, execute every arm and seed in an isolated workspace, and record a run file per execution with its reproducibility key and failure category. Use when the user wants to implement, run, or execute an experiment, benchmark, or ablation that has already been designed.
user-invocable: true
---

# run-experiment

Write the code, run the ladder, record every run.

You are allowed to write and execute code here — that is the point of this
phase. What you are not allowed to do is change the experiment while running it.

## Before anything runs

```bash
python3 ../design-experiment/scripts/spec_hash.py experiments/<id>/spec.md --verify
```

If it fails, **stop**. The spec changed after it was pre-registered, and running
it now would produce a result against a rule that was adjusted. Restore the spec
or register the change as a new experiment id.

If the spec has no hash at all, it was never pre-registered. Go back to
`design-experiment`.

## Steps

### 1. Build the workspace

`experiments/<id>/workspace/` — self-contained, and containing everything needed
to rebuild the run without this session:

```
workspace/
  run.py            or whatever the entrypoint is
  requirements.txt  pinned
  manifest.md       entrypoint, commands, hashes, divergences, assumptions
```

Prefer adapting a vetted reference implementation over writing from scratch;
`sources/` records which papers ship code. If you adapt one:

- **Check the licence first.** MIT, Apache-2.0, BSD and MPL-2.0 are fine to
  adapt. Anything else, or no licence at all, means ask before copying a line.
- **Clone, do not run.** Third-party code executes only inside the sandboxed
  workspace in step 3, never at fetch time and never against your own machine's
  environment.
- **Record every divergence** from the reference in `manifest.md`. A silent
  divergence is the reason a reproduction fails mysteriously three weeks later.

### 2. Smoke first, and honestly

Run the smoke rung: minutes, tiny, correctness only. It answers nothing about
the hypothesis and is not allowed to be reported as if it did.

What smoke is checking:

- The thing runs to completion and writes the metrics file.
- Both arms differ in the way the spec says and in no other way — diff the
  configs and look.
- The baseline reproduces the reference number from the spec within tolerance.
  If it does not, **your harness is wrong**, not the literature. Fix it before
  climbing. This check is the cheapest bug-catch in the whole system.

**A failed smoke stops the ladder.** Do not proceed to `small` hoping scale
fixes it.

### 3. Execute every arm × seed

Run the full grid: every arm, every seed in the spec. Isolate execution — a
container if one is available, otherwise a dedicated virtualenv and working
directory, with no network unless the spec needs to download a dataset and says
so.

Write one file per run to `experiments/<id>/runs/<arm>-seed<n>.md`, from
`references/run-record.md`. The frontmatter carries the reproducibility key:

```yaml
spec_hash: 8f3ad2...
code_hash: 91bb0c...     # sha256 of the workspace source files
env_hash: 4d21fe...      # sha256 of the pinned requirements + python version
arm: curriculum
seed: 0
```

**All five fields.** The arm is part of the identity because one spec runs
several arms at the same seed; drop it and the baseline and treatment collide,
and every experiment looks irreproducible against itself.

### 4. Categorise failures, and do not paper over them

A run that fails gets a record too, with `status:` set to what actually
happened:

| Status | What it means | What to do |
|---|---|---|
| `oom` | Out of memory | Reduce batch size *for every arm equally*, or the comparison is broken |
| `nan` | Loss diverged to NaN | Usually LR or precision; a bug, not a retry |
| `hang` | Exceeded the wall-clock ceiling | Kill it and record it; do not extend the ceiling mid-experiment |
| `diverged` | Ran, but training did not converge | Record it — a treatment that will not train is a finding |
| `error` | Crashed | Record the traceback's last lines |
| `skipped` | Not run | Say why |

Retrying an OOM with the same batch size is not a strategy. And if you change a
setting to make a run succeed, **change it for every arm** and record it — a
treatment that got a smaller batch than the baseline is a different experiment.

### 5. Report, then analyze

Update `spec.md`'s `status:` to `complete`, append to `log.md`, and tell the user
how many runs succeeded, failed and by what category. Then run `analyze-results`.

Do not interpret the numbers here. Reading metrics before the decision rule is
applied is how a rule gets quietly reinterpreted, and this phase's job is to
produce trustworthy runs, not conclusions.

## Getting this wrong

**Tuning while running.** Adjusting anything after seeing a seed's result
invalidates the experiment. If something must change, stop, register a new spec,
and start over.

**Unequal treatment.** Every setting that is not the independent variable is the
same across arms. Every one. This includes the things you did not think of as
settings: data order, worker count, precision, checkpoint frequency.

**Losing the failures.** A grid reported as "8 runs, mean 0.71" when four
crashed is a fabrication. Every cell in the grid has a record.

**Reporting smoke as a result.** Smoke exists to check the harness. Its numbers
go in the log, never in the memo.
