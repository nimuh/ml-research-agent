# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status: implemented

M0–M5 are built; the ten-phase loop runs end to end and `mra auto "<idea>"` drives it. Each module's docstring is still its spec, and `docs/PLAN.md` is still the contract between modules — **read it before changing anything structural**; it defines the data model, the phase machine, the agent roster, and the safety/cost controls. §9 records how the plan's open questions were settled and which decisions departed from the original text.

When changing a module, honor its docstring rather than redesigning around it. If the docstring is wrong, change it deliberately and check whether `docs/PLAN.md` needs the same edit.

The whole test suite runs **offline** — no API key, no network. `llm.client.FakeLLMClient` scripts model responses (return a pydantic model instance and it becomes a forced tool call), and literature sources are exercised through cassette fixtures. If a change makes a test need the network, the change is wrong.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # or [literature] / [knowledge] / [experiments] / [claude-code] / [dev]
cp .env.example .env             # ANTHROPIC_API_KEY required; others optional

pytest                           # testpaths=tests, -q by default
pytest tests/unit/test_x.py::test_name    # single test
ruff check src tests             # line-length 100, rules E,F,I,UP,B,SIM
ruff format src tests
mypy                             # strict, pydantic plugin, files=src/ml_research_agent
```

### A green test suite does not prove the install works

`pytest` sets `pythonpath = ["src"]`, so the suite imports the source tree directly and passes even when the installed package is unimportable. Verify the entrypoint separately with `mra --help` — a green suite is not evidence that it works.

**Known macOS breakage of the editable install.** The `.pth` file pip writes for `pip install -e .` acquires the `com.apple.provenance` extended attribute. `site.addpackage` opens `.pth` files through `io.open_code()` and silently returns on `OSError`, so the file is skipped with no error at all and `import ml_research_agent` fails everywhere except pytest. The attribute is protected — `xattr -c` will not remove it, and a hand-written replacement `.pth` acquires it too, so rewriting the file only works until the next process touches it.

Two workarounds that do hold:

```bash
PYTHONPATH=src .venv/bin/mra --help   # per-invocation; export it for a session
pip install .                          # non-editable; copies into site-packages, no .pth
```

The non-editable install loses live source edits, so prefer `PYTHONPATH=src` while developing.

`mra` is the console entrypoint (`ml_research_agent.cli:main`): `idea`, `survey`, `design`, `run`, `report`, `auto`, `status`, and `kb search|show|stats|health`. Pass `-y` to auto-approve human gates in non-interactive runs.

## Configuration

`configs/default.yaml` documents every knob. Precedence: CLI flag > `MRA_*` env > `configs/local.yaml` (gitignored) > `configs/default.yaml`. Nested keys use a double underscore in env: `MRA_BUDGET__USD_PER_PROJECT=50`. Config objects are immutable and passed explicitly — no module-level globals.

Two providers, selected by `llm.provider`. `anthropic` is the metered API and needs `ANTHROPIC_API_KEY`. `claude_code` drives the logged-in Claude Code CLI through the Claude Agent SDK, so a Claude Code subscription needs no key at all — set it in `configs/local.yaml` and install `[claude-code]`. `llm/claude_code.py` documents what the adapter has to do to make an agent harness behave like a single-shot completion endpoint; read it before changing anything there.

Model tiers are configured, not hardcoded: `fast` (haiku) for extraction/screening, `standard` (sonnet) for notes and code work, `deep` (opus) for framing, design, and critique. `orchestrator/router.py` is the only place that maps a task to a tier.

## Architecture

**Dependency rule (strictly one-directional):** platform (`llm/`, `tools/`, `observability/`, `utils/`) ← capability layers (`literature/`, `knowledge/`, `code/`, `experiments/`) ← `agents/` ← `orchestrator/` ← `cli.py`. Capability layers must not import each other; the orchestrator composes them. `cli.py` holds no business logic.

**Control flow:** `orchestrator/director.py` runs a ten-phase machine (FRAME → SURVEY → CURATE → GROUND → SYNTHESIZE → DESIGN → IMPLEMENT → RUN → ANALYZE → REPORT), with ANALYZE looping back to DESIGN on an inconclusive verdict. Each phase declares preconditions and exit criteria scored by the Critic; failing exit criteria loops back rather than proceeding.

**Agents never talk to each other.** All coordination flows through the orchestrator and `orchestrator/state.py` (`ProjectState`: append-only JSONL event log + derived snapshot). Agents are stateless — memory lives in `ProjectState` and the KB. Every agent shares `agents/base.py`'s loop (prompt → tool calls → structured output → validate → repair-on-invalid) and carries an explicit tool allow-list.

### Invariants that constrain implementation choices

These are load-bearing; violating one is a design regression, not a style preference.

- **Traceability.** Every domain object in `types.py` carries `provenance`. The Writer may only assert what a KB citation or a run id backs. Notes without passage-level provenance are rejected.
- **Files are truth, indexes are derived.** The KB is Markdown + SQLite + blobs under `workspace/kb/`; vector/BM25 indexes must be rebuildable from scratch. Ingest is idempotent by content hash.
- **Pre-registration.** An `ExperimentSpec` without a stated decision rule (what result would refute the hypothesis) is rejected at DESIGN when `require_preregistered_decision_rule: true`.
- **Reproducibility contract.** A run is identified by `(spec_hash, code_hash, env_hash, arm, seed)` — the arm is part of the key because one spec runs several arms at the same seed. Seeds default to 3; single-seed deltas are reported as inconclusive by construction.
- **Cheap before expensive.** The `smoke → small → main` scale ladder is enforced by the orchestrator, not by agent judgment.
- **Cloning is not running.** Third-party repo code only ever executes inside the sandbox (`experiments/sandbox.py`); `code/fetch.py` must not execute anything at fetch time. License gating (`code/license.py`) precedes any adaptation.
- **Bounded fan-out, logged truncation.** `max_candidates`, `max_kb_papers`, `snowball_depth`, `max_parallel_runs` — every cap is enforced and every truncation is logged, never silent.
- **Budgets are hard stops.** Exceeding a ceiling raises `BudgetExceeded`; never silently truncate or degrade.
- **Single retrieval entrypoint.** All agent retrieval goes through `knowledge/search.py` so there is one place to instrument, cache, and evaluate.
- **Prompts are data.** Versioned assets under `prompts/` and `llm/prompts/`, diffable and testable against golden cases — not string literals in agent code.
- **Retrieval ≠ relevance.** `literature/` returns candidates and never judges relevance; screening is the Curator's job, so recall and precision tune independently.
- **The Critic is adversarial by construction** — prompted to refute, not approve. It runs post-CURATE, post-SYNTHESIZE, post-DESIGN, post-ANALYZE.

## Testing strategy

Six categories, per `docs/PLAN.md` §7: unit (pure logic, no LLM calls — dedup, ranking, chunking, spec hashing, decision-rule evaluation, budget accounting, stats); golden-prompt tests per agent prompt version; cassette-style HTTP fixtures so literature tests never hit live APIs; one end-to-end tiny run (5 papers, 1 repo, 30-second smoke experiment); a reproducibility test (same spec + seed → same `RunRecord` metrics); and an adversarial suite of deliberately leaky evals, unfair baselines, and single-seed noise the Critic must catch. The LLM response cache (`llm/cache.py`, keyed by model + prompt hash + params) is what makes prompt tests fast and deterministic.

When creating and running test scripts, have separate agents for writing tests and reviewing test code. When with every major change, make sure the tests pass.

## workspace/

Runtime state only — gitignored and regenerable (`workspace/**` ignored except `.gitkeep`). Never commit its contents, and never treat anything there as a source of truth that can't be rebuilt. Its `kb/{raw,wiki,reports}/` layout deliberately mirrors the `kb-*` skills so those operate on it directly.
