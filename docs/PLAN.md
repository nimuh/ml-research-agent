# ml-research-agent — System Plan

**Status:** implemented. M0–M5 are built and the ten-phase loop runs end to end; each module's docstring remains its spec, and this document remains the contract between them. The open questions in §9 are settled — see the answers recorded there.

---

## 1. What this system is

You give it a research idea in one sentence. It:

1. **Frames** the idea into a falsifiable research brief with explicit success criteria.
2. **Surveys** the literature across arXiv / Semantic Scholar / OpenReview / Papers-with-Code / GitHub, and snowballs citations until new results stop being informative.
3. **Builds a knowledge base** of papers *and* runnable codebases — structured notes, a hybrid search index, and a knowledge graph linking methods, datasets, metrics, and claims.
4. **Runs a team of agents** that design pre-registered experiments, implement them (preferring adaptation of a vetted reference implementation over greenfield code), execute them in a sandbox, analyze results, and red-team the conclusions.
5. **Reports** what the evidence actually supports — where a negative result is as legible and as trustworthy as a positive one.

### The design commitments that shape everything else

| Commitment | Consequence in the architecture |
|---|---|
| **Every claim is traceable** | Each domain object carries `provenance`. The Writer may only assert what a KB citation or a run id backs. |
| **Falsify before you run** | `ExperimentSpec` requires a *pre-registered decision rule* — what result would refute the hypothesis — written before compute is spent. |
| **Files are truth, indexes are derived** | The KB is Markdown + SQLite + blobs on disk. Vector/BM25 indexes are rebuildable from scratch at any time. |
| **Cheap before expensive** | A scale ladder (`smoke → small → main`) and a model-tier router. Nothing large launches until the small version passes. |
| **Resumable and auditable** | `ProjectState` is an append-only event log plus a derived snapshot. Any phase can be replayed. |
| **Agents don't talk to each other** | All coordination flows through the orchestrator and shared state. No emergent multi-agent chatter, no untraceable context. |

---

## 2. Architecture at a glance

```
                        ┌──────────────────────────────────────┐
   "test this idea" ──▶ │   CLI  (mra idea/survey/design/run)  │
                        └───────────────────┬──────────────────┘
                                            ▼
                     ┌──────────────────────────────────────────┐
                     │  ORCHESTRATOR — ResearchDirector          │
                     │  phase machine · router · budgets ·       │
                     │  checkpoints · human gates                │
                     │  ── owns ProjectState (event log) ──      │
                     └───┬──────────┬──────────┬──────────┬──────┘
                         │          │          │          │
              dispatches │          │          │          │
                         ▼          ▼          ▼          ▼
                   ┌──────────────────────────────────────────┐
                   │  AGENT TEAM (typed in → typed out)        │
                   │  Framer · Scout · Curator · CodeAnalyst · │
                   │  Synthesizer · ExperimentPlanner ·        │
                   │  Implementer · Runner · Analyst ·         │
                   │  Critic · Writer                          │
                   └───┬──────────┬──────────┬──────────┬──────┘
                       ▼          ▼          ▼          ▼
        ┌────────────────┐ ┌─────────────┐ ┌──────────┐ ┌───────────────┐
        │  literature/   │ │ knowledge/  │ │  code/   │ │ experiments/  │
        │ query·sources· │ │ ingest·chunk│ │ discover │ │ spec·design·  │
        │ dedupe·rank·   │ │ ·embed·index│ │ ·fetch·  │ │ workspace·    │
        │ snowball·parse │ │ ·search·    │ │ analyze· │ │ sandbox·      │
        │                │ │ notes·graph │ │ recipe   │ │ execute·      │
        │                │ │ ·health     │ │ ·env·lic │ │ tracking·     │
        │                │ │             │ │          │ │ analysis      │
        └────────────────┘ └─────────────┘ └──────────┘ └───────────────┘
                       ▲          ▲          ▲          ▲
                       └──────────┴────┬─────┴──────────┘
                                       │
        ┌──────────────────────────────┴───────────────────────────────┐
        │  PLATFORM:  llm/ (client·structured·prompts·budget·cache)     │
        │             tools/ (registry·shell·python_exec·fs·http)       │
        │             observability/ (logging·tracing·cost)             │
        │             reporting/ · utils/                               │
        └───────────────────────────────────────────────────────────────┘

        Persisted:  workspace/kb/{raw,wiki,reports}   workspace/runs/
                    workspace/repos/                  workspace/cache/
```

**Dependency rule:** platform ← capability layers (literature, knowledge, code, experiments) ← agents ← orchestrator ← CLI. Arrows never point back up. Capability layers do not import each other; the orchestrator composes them.

---

## 3. The research loop

Ten phases. Each declares preconditions, dispatched agents, and exit criteria scored by the Critic. Failing exit criteria loops back rather than proceeding.

| # | Phase | Agents | Produces | Exit criterion |
|---|---|---|---|---|
| 1 | **FRAME** | Framer | `ResearchBrief` | Idea restated as ≥1 falsifiable claim with measurable success criteria and stated scope limits |
| 2 | **SURVEY** | Scout | ≤300 ranked candidates | Snowball yield below threshold; the known-seminal papers for the area appear unprompted |
| 3 | **CURATE** | Curator, Critic | KB notes for ≤60 papers | Every note has provenance; inclusion/exclusion reasons recorded |
| 4 | **GROUND** | CodeAnalyst | `Recipe` per usable repo | ≥1 reference implementation reproduces a published number at smoke scale, or the gap is documented |
| 5 | **SYNTHESIZE** | Synthesizer, Critic | Settled / contested / gap map; baseline + benchmark set | The idea's novelty claim survives the Critic, or the brief is revised |
| 6 | **DESIGN** | ExperimentPlanner, Critic | `Hypothesis` + `ExperimentSpec` ladder | Every spec has controls, baselines, seeds, cost estimate, and a pre-registered decision rule |
| 7 | **IMPLEMENT** | Implementer | Workspace + code snapshot | Smoke run completes; baseline reproduces within tolerance |
| 8 | **RUN** | RunnerAgent | `RunRecord`s + artifacts | All seeds complete or fail with a recorded, categorized reason |
| 9 | **ANALYZE** | Analyst, Critic | `Result` + `Verdict` | Decision rule evaluated; seed variance reported; Critic finds no leakage or unfair-baseline defect |
| 10 | **REPORT** | Writer | Research memo | Every claim cites a KB source or run id |

**Iteration:** ANALYZE returns `supported | refuted | inconclusive`. `inconclusive` → `ablation.py` proposes the follow-ups with the best information gain per dollar and control returns to DESIGN. Two consecutive inconclusive rounds on the same hypothesis escalates to a human gate rather than burning more compute.

**Human gates** (`orchestrator/gates.py`), configurable: after SURVEY (is the KB the right literature?), before RUN (approve the specs and the bill), before REPORT (off by default).

---

## 4. Data model (`types.py`)

Everything is a pydantic model with a stable id and a `provenance` field.

```
Idea ──▶ ResearchBrief ──▶ Question ──▶ Hypothesis ──▶ ExperimentSpec ──▶ RunRecord ──▶ Result ──▶ Verdict
                 │                          ▲                                              │
                 └──▶ Paper ──▶ Note ──▶ Claim ──▶ Evidence ◀───────────────────────────────┘
                        └──▶ CodeRepo ──▶ Recipe
```

Key objects:

- **`Paper`** — ids (arXiv/DOI/S2), metadata, venue, citation counts, parsed sections, linked repos, source URLs.
- **`Note`** — the KB's atomic unit: task, method, datasets, baselines, headline metrics, compute used, stated limitations, relevance-to-brief, confidence, and the passage locators backing each field.
- **`Claim`** / **`Evidence`** — a claim is an assertion ("X improves Y under Z"); evidence links it to a paper passage *or* a run id. Claims can be `contradicts` one another — that edge is a first-class research signal, not an error.
- **`CodeRepo`** / **`Recipe`** — repo pinned to a commit, plus the runnable distillation: exact commands, config knobs, expected artifacts, reference numbers, known gotchas.
- **`ExperimentSpec`** — independent/dependent variables, controls, dataset + split, baselines, metrics, seeds, budget ceiling, stopping rule, **decision rule**. Hashable: `spec_hash` identifies a reproducible experiment.
- **`RunRecord`** — `(spec_hash, code_hash, env_hash, arm, seed)` → metrics, artifacts, logs, cost, status. That tuple is the reproducibility contract; the arm belongs in it because one spec runs several arms at the same seed.

---

## 5. Layer-by-layer

### 5.1 `orchestrator/` — the control plane

- **`director.py`** — the phase machine of §3. Holds no domain logic itself; it sequences, checks preconditions, and evaluates exit criteria.
- **`state.py`** — `ProjectState`: append-only JSONL event log + derived snapshot. Every agent output, tool call, decision, and gate response is an event. Enables replay, audit, and diffed resume.
- **`plan.py`** — decomposes a brief into questions → hypotheses → an experiment *ladder* ordered cheapest-and-most-diagnostic first. Re-plans when results invalidate an assumption.
- **`router.py`** — task → (agent, model tier, tool subset, budget). This is where cost policy lives: `fast` for screening and extraction, `standard` for note writing and code work, `deep` for framing, design, and critique.
- **`checkpoint.py`** — snapshot/resume; a resumed run re-executes only steps whose inputs changed.
- **`gates.py`** — renders a decision packet and blocks until approved / auto-approved / timed out.

### 5.2 `agents/` — the team

`BaseAgent` gives every role the same loop: prompt → tool calls → structured output → validate → repair-on-invalid → return. Agents are **stateless** — memory lives in `ProjectState` and the KB — and each has an explicit tool allow-list.

| Agent | Input → Output | The failure it exists to prevent |
|---|---|---|
| **Framer** | `Idea` → `ResearchBrief` | Researching a vague idea nobody could ever falsify |
| **Scout** | `ResearchBrief` → ranked candidates | Missing the obvious prior work |
| **Curator** | candidates → `Note`s | A KB full of abstracts nobody extracted structure from |
| **CodeAnalyst** | `Paper` + repo → `Recipe` | "There's a repo" that turns out to be unrunnable |
| **Synthesizer** | KB → settled/contested/gap map | Testing something already settled, or reinventing a baseline |
| **ExperimentPlanner** | `Hypothesis` → `ExperimentSpec` | Uncontrolled experiments that can't answer anything |
| **Implementer** | spec + recipe → workspace | Greenfield code with silent divergences from the reference |
| **RunnerAgent** | workspace → `RunRecord`s | Silent hangs, NaNs, OOM, wasted GPU-hours |
| **Analyst** | runs → `Result` + `Verdict` | Reading noise across 1 seed as a result |
| **Critic** | any artifact → findings + gate score | Leaky evals, unfair baselines, cherry-picked seeds, unsupported claims |
| **Writer** | state + KB → report | Confident prose the evidence doesn't support |

The **Critic is adversarial by construction** — prompted to refute, not to approve — and runs at four points (post-CURATE, post-SYNTHESIZE, post-DESIGN, post-ANALYZE). It is the main defense against a system that is fluent enough to be convincing while wrong.

### 5.3 `literature/` — acquisition

Query planning → parallel source fan-out → dedup → cheap ranking → snowball → full-text fetch/parse. It returns candidates; **it never decides relevance** — that's the Curator's judgment call, kept separate so retrieval recall and screening precision can be tuned independently.

- Each source implements one `Source` protocol (`search`, `get`) with rate limiting, on-disk caching, and retries — adding a source is one file.
- **Dedup** by DOI/arXiv id → normalized title → author-year fuzzy match; merges to the richest record while keeping all source links.
- **Ranking** is deliberately cheap (semantic similarity to brief + citation velocity + recency + venue + code availability) — it's a filter before expensive LLM screening, not a judgment.
- **Snowballing** expands backward (references) and forward (citations) from seeds, breadth-limited and score-gated, stopping when marginal yield drops below threshold.
- **Parsing** is section-aware. Chunking a paper into flat text destroys the locators needed to cite precisely, which breaks the traceability commitment.

### 5.4 `knowledge/` — the KB

Layout intentionally mirrors the existing `kb-*` skills so they operate on it directly:

```
workspace/kb/
  raw/       PDFs, LaTeX, HTML, repo snapshots  (immutable, content-hash named)
  wiki/      compiled notes + concept pages     (Markdown + YAML front matter)
  reports/   generated reports
  kb.sqlite  metadata, relations, graph edges
  index/     vector + BM25 indexes (derived, rebuildable)
```

- **Ingest** is idempotent by content hash: `raw doc → parse → chunk → embed → index → note stub`. Safe to re-run over a growing `raw/`.
- **Search** is hybrid (vector + BM25 + metadata filters) with reciprocal-rank fusion and optional reranking. `knowledge/search.py` is the **single retrieval entrypoint** for all agents — one place to instrument, cache, and evaluate.
- **Notes** carry front matter: `id, type, sources, claims, tags, confidence, added_at, supersedes`. New evidence supersedes rather than overwrites, so the KB has history.
- **Graph** (`papers, methods, datasets, metrics, repos, claims` × `cites, uses, contradicts, reproduces`) powers gap-finding: a method-by-dataset matrix with holes is a candidate experiment; a `contradicts` edge is a candidate replication.
- **Health** surfaces orphan notes, uncited claims, staleness, and coverage gaps as TODOs that feed the next survey round.

### 5.5 `code/` — codebase knowledge

The layer that separates *describable* ideas from *testable* ones.

`discover` (paper → implementations, ranked by fidelity, activity, license, published reproductions) → `fetch` (shallow clone pinned to a commit, into a content-addressed cache; **no repo code executes at fetch time**) → `analyze` (entrypoints, config system, data pipeline, model defs, train/eval loops, dependency graph, hardware assumptions) → `recipe` (the runnable distillation) → `env` (pinned interpreter/CUDA, lockfile or container spec) → `license` (gate before any adaptation).

**Rule: third-party code only ever executes inside the sandbox.** Cloning is not running.

### 5.6 `experiments/` — the part that makes it research

`hypothesis → spec → design → workspace → sandbox → execute → tracking → metrics → analysis → ablation`

- **Pre-registration.** `require_preregistered_decision_rule: true` means a spec without a stated falsifier is rejected at DESIGN. This is the single highest-value constraint in the system: it makes a negative result a *finding* rather than a *failure*, and it removes the incentive to rationalize whatever came out.
- **Scale ladder.** `smoke` (minutes, correctness only) → `small` (a reduced but honest setting) → `main`. The ladder is enforced by the orchestrator, not by agent judgment.
- **Reproducibility contract.** A run is identified by `(spec_hash, code_hash, env_hash, arm, seed)`. The workspace manifest is sufficient to rebuild the run without the original session.
- **Seeds are mandatory** (default 3). The Analyst reports spread and paired comparisons against the baseline; single-seed deltas are reported as inconclusive by construction.
- **Failure signatures** (OOM, NaN, hang, divergence, silent no-op) are detected and categorized by the runner, each with its own retry policy — retrying an OOM with the same batch size is not a strategy.
- **`ablation.py`** closes the loop: given a verdict, propose the follow-ups ranked by expected information gain per dollar.

### 5.7 Platform

- **`llm/`** — one client, tiered models, structured outputs via schema-constrained generation with a repair loop, versioned prompt assets (prompts are *data*: diffable, testable against golden cases), a hard token/cost budget that raises `BudgetExceeded` instead of silently truncating, and a deterministic response cache keyed by `(model, prompt hash, params)` that makes replays and tests cheap.
- **`tools/`** — every agent-callable tool is typed, permissioned, logged, and individually disableable. `fs.py` is workspace-scoped such that path traversal is impossible by construction; `shell.py` has allow/deny lists and timeouts; `http.py` enforces domain policy and rate limits.
- **`observability/`** — JSONL structured logs with run/agent/phase correlation ids, span tracing over agent turns and runs, and a cost ledger across tokens, compute hours, and API calls. *If a run can't be explained after the fact, the system is broken.*
- **`reporting/`** — Markdown-first assembly from `ProjectState` + KB + results, standard figures (learning curves, baseline-vs-variant with error bars, ablation grids, cost breakdowns), and the decision-packet template used at human gates.

---

## 6. Safety and cost

| Risk | Control |
|---|---|
| Untrusted repo code | Sandbox by default (`subprocess` → `docker` → remote); network `deny` with an allow-list for dataset/model downloads; workspace-scoped FS; wall-clock/GPU/token ceilings; kill switch |
| Runaway spend | Per-project USD ceiling, per-phase token ceiling, cost estimate required before RUN, human gate before compute |
| Runaway scope | `max_candidates`, `max_kb_papers`, `snowball_depth`, `max_parallel_runs` — every fan-out is bounded and the truncation is *logged*, never silent |
| Fabricated citations | Notes rejected without passage-level provenance; Writer restricted to KB-backed claims; Critic verifies a sample against `raw/` |
| Licensing | License extraction gates repo adaptation and dataset use before any code is copied |
| Secrets | `.env` only, never in `ProjectState`, never in logs, redacted in traces |

---

## 7. Testing strategy

- **Unit** — pure logic without LLM calls: dedup, ranking, chunking, spec hashing, decision-rule evaluation, budget accounting, statistics.
- **Golden prompts** — each agent's prompt version tested against fixture inputs with schema + rubric assertions; the LLM cache makes this fast and deterministic.
- **Recorded-source fixtures** — cassette-style HTTP fixtures so literature tests never hit live APIs.
- **End-to-end tiny** — one narrow idea, 5 papers, 1 repo, a 30-second smoke experiment; runs in CI and exercises every phase.
- **Reproducibility test** — the same spec at the same seed must produce the same `RunRecord` metrics; this test failing means the reproducibility contract is broken.
- **Adversarial suite** — deliberately leaky eval, unfair baseline, and single-seed noise cases the Critic must catch. These are regression tests for the system's *judgment*, and the most important tests here.

---

## 8. Build order

| Milestone | Scope | Done when |
|---|---|---|
| **M0 — Skeleton** | `types`, `config`, `llm/client`, `structured`, `observability`, CLI shell | `mra idea "..."` returns a validated `ResearchBrief` |
| **M1 — Literature + KB** | arXiv + Semantic Scholar sources, dedup, rank, parse, ingest, hybrid search, Curator notes | `mra survey` builds a 30-paper KB; `mra kb search` returns cited passages |
| **M2 — Grounding** | GitHub/PwC discovery, repo fetch, static analysis, Recipe, license gate | A reference implementation runs at smoke scale in the sandbox |
| **M3 — Experiments** | Hypothesis, spec + decision rule, workspace, sandbox, execute, tracking | A pre-registered smoke experiment runs end to end and produces a `RunRecord` |
| **M4 — The loop** | Analyst, Critic, ablation, re-planning, gates, checkpoints | An inconclusive result automatically proposes and runs the right follow-up |
| **M5 — Report** | Synthesizer, Writer, figures, templates | A full memo with only KB/run-backed claims, produced from one CLI command |

Ship M0–M1 before anything else — the KB is what every later phase reads from, and a weak KB fails silently downstream.

---

## 9. Open questions — how they were settled

1. **Vector store** — **Settled: a local numpy-backed index**, not LanceDB. LanceDB has no wheels for the Python version this is developed against, and the "files are truth, indexes are derived" invariant makes the vector store a swappable detail: `knowledge/index.py` persists a `.npy` matrix plus an id list and rebuilds itself from SQLite and the wiki alone (`HybridIndex.rebuild`, covered by a test asserting a rebuilt index returns identical results). Swapping in LanceDB or pgvector later means reimplementing one class.
2. **PDF parsing** — **Settled: PyMuPDF, LaTeX preferred.** `literature/fetch.py` requests arXiv LaTeX source first and falls back to PDF. Parsing is section-aware in both cases because flat text destroys the locators traceability depends on.
3. **Compute target** — **Settled: local by default, remote is a real backend or an error.** `sandbox.backend` selects `subprocess` (default), `docker` (the only one where network denial is genuinely enforced), or `remote`. `RemoteSandbox` raises rather than falling back to local execution — a remote policy that quietly becomes a local one is the exact failure the sandbox exists to prevent.
4. **Screening cost** — **Settled: cheap rank, then `fast`-tier screen.** `literature/rank.py` filters with lexical similarity, citation velocity, recency, venue and code availability (no LLM, no embeddings), and only the survivors reach the `fast`-tier Screener. Ranking never judges relevance, so recall and precision tune independently.
5. **Human gates by default** — **Settled: on for post-SURVEY and pre-RUN, off before REPORT.** Full autonomy is `gates.auto_approve: true`. The pre-RUN packet shows the bill and the run count before any compute burns.
6. **Ambition ceiling** — **Settled: small-scale, high-signal.** The scale ladder is enforced by the orchestrator: nothing reaches `main` before `smoke` passes, a failed smoke stops the ladder, and every spec carries a cost ceiling checked against the project budget before it runs. Full-scale training is reachable only by explicitly configuring the top rung and approving the bill.

### Decisions made during implementation

- **The reproducibility tuple gained the arm.** The plan originally stated `(spec_hash, code_hash, env_hash, seed)`, but a spec runs several arms at the same seed; without the arm, baseline and treatment share a key and every experiment looks irreproducible against itself. `RunRecord.repro_key` is therefore a 5-tuple.
- **`Hypothesis.status` has an `escalate` terminal state**, so the two-inconclusive-rounds rule in §3 has somewhere to land instead of looping.
- **A blocking Critic finding downgrades a verdict to `inconclusive`**, whatever the arithmetic said. A defensible-looking number with a leaky eval behind it is not a result.
- **Overriding `paths.workspace` re-roots every derived path.** Otherwise a second project or a test silently writes into the first one's knowledge base.
- **A clean null result can be `refuted`, not just `inconclusive`.** A p-value gate alone cannot deliver §1's promise that a negative result is as legible as a positive one: an effect indistinguishable from zero fails significance, so a well-run experiment that genuinely rules the effect out reads identically to one too noisy to tell. `analysis.py` therefore adds an equivalence check — when the effect's confidence interval sits entirely below the pre-registered threshold, the prediction is refuted. Too few seeds is still inconclusive; the check never smuggles a verdict past the seed floor.
- **One definition of "the gate passed".** `gate_passed()` is the only one, and `CritiqueReport.passed` is computed from it. The Critic's own `would_you_stake_your_reputation` is deliberately not a term — `validate_output` already requires that declining to stand behind an artifact be backed by a finding, which moves the score; counting it twice would let a model veto a phase on sentiment.
- **The decision rule compares against the baseline the treatment does *worst* against**, not the first one declared. Otherwise the order a planner happened to list baselines in could decide a verdict, and adding a weak baseline could manufacture a win.
- **`network: deny` is enforced only on the container backend.** On `subprocess` it is proxy environment variables — a cooperative hint a raw socket ignores. `Sandbox.enforces_network_policy` says which it is, and a run under an advisory policy logs a warning rather than implying a boundary that isn't there.
- **Three documented controls read by nothing.** `code.run_untrusted_code`, `sandbox.network_allowlist`, and `dry_run` were each specified in `configs/default.yaml` and consulted nowhere — guards that were not merely wrong but unreachable. `Sandbox.run(..., untrusted=True)` now gates third-party repo execution on the first; the allow-list populates `no_proxy` so denial exempts exactly those hosts; and `execute_run` returns a `SKIPPED` record under `dry_run` instead of launching the experiment.
- **Network denial was inert, not merely advisory.** `no_proxy` was set to `"*"`, which exempts *every* host from the blackhole proxy — so even a cooperative HTTP client reached the network freely. It now carries the allow-list, which is what that setting is for.

---

## 10. Repository layout

```
ml-research-agent/
├── pyproject.toml            # hatchling, src-layout, `mra` entrypoint
├── configs/default.yaml      # all knobs, documented
├── docs/PLAN.md              # this document
├── prompts/                  # versioned prompt assets
├── scripts/                  # dev/eval utilities
├── tests/{unit,integration,fixtures}/
├── workspace/                # runtime state (gitignored, regenerable)
│   ├── kb/{raw,wiki,reports}/
│   ├── runs/ repos/ cache/
└── src/ml_research_agent/
    ├── cli.py config.py types.py errors.py
    ├── orchestrator/   director state plan router checkpoint gates
    ├── agents/         base registry framer scout curator code_analyst
    │                   synthesizer experiment_planner implementer
    │                   runner_agent analyst critic writer
    ├── literature/     query fetch parse dedupe rank snowball
    │                   sources/{arxiv,semantic_scholar,openreview,
    │                            paperswithcode,github_src,web}
    ├── knowledge/      schema store ingest chunk embed index search
    │                   notes graph health
    ├── code/           discover fetch analyze recipe env license
    ├── experiments/    hypothesis spec design workspace sandbox execute
    │                   tracking metrics analysis ablation
    ├── llm/            client structured budget cache prompts/
    ├── tools/          registry shell python_exec fs http
    ├── reporting/      report figures templates/
    ├── observability/  logging tracing cost
    └── utils/          io hashing cache concurrency
```

Every module listed above exists with a docstring describing its responsibility and planned API. Read the docstrings as the per-module spec; this document is the contract between them.
