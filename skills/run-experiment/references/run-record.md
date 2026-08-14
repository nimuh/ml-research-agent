---
type: run
id: exp-001/curriculum-seed0
spec_hash: 8f3ad2c1b4e50917
code_hash: 91bb0c7fa2d31845
env_hash: 4d21fe08c6b7a239
arm: curriculum
seed: 0
status: ok                     # ok | oom | nan | hang | diverged | error | skipped
metrics: {accuracy: 0.7031}
runtime_seconds: 41
started: 2026-08-14T11:02:00Z
added: 2026-08-14
tags: [run]
---

# exp-001 · curriculum · seed 0

Spec: [[spec]] · Scale: smoke

## Metrics

| Metric | Value |
|---|---|
| accuracy | 0.7031 |

## Command

```
python3 run.py --arm curriculum --seed 0 --out runs/curriculum-seed0
```

## Notes

{{Anything that would change how the number is read: a warning, a fallback that
fired, a dataset that downloaded a different version. If nothing, say "nothing
unusual" rather than leaving it blank — a blank section is ambiguous between
"clean" and "not checked".}}

{{On failure, replace Metrics with:}}

## Failure

Category: `oom`

```
{{last ~15 lines of stderr}}
```

What was tried: {{...}}. What it would take: {{...}}.
