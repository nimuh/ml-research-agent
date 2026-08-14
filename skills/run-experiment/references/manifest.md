# Workspace manifest — exp-001

*Sufficient on its own to rebuild these runs without the session that made them.*

| | |
|---|---|
| Spec | `../spec.md` (spec_hash `8f3ad2c1b4e50917`) |
| Entrypoint | `python3 run.py --arm {arm} --seed {seed} --out {out}` |
| Smoke | `python3 run.py --arm {arm} --seed {seed} --out {out} --smoke` |
| Metrics file | `{out}/metrics.json` |
| code_hash | `91bb0c7fa2d31845` — sha256 over the source files below |
| env_hash | `4d21fe08c6b7a239` — sha256 over `requirements.txt` + python version |
| Python | 3.12.4 |

## Files

- `run.py` — both arms; the only difference between them is `--arm`
- `requirements.txt` — pinned, exact versions

## Adapted from

{{Repo URL, pinned commit, licence. Or "written from scratch".}}

## Divergences from the reference

{{Every place this differs from the implementation it was adapted from. A silent
divergence is why a reproduction fails mysteriously weeks later.}}

- {{...}}

## Assumptions

{{Things taken as given that the spec did not state — a default the reference
used, a preprocessing step inherited without checking.}}

- {{...}}
