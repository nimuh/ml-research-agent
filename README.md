# ml-research-agent

An agentic system for end-to-end ML research. Give it an idea in one sentence; it surveys the
literature, builds a knowledge base of papers **and runnable codebases**, then runs a team of agents
that design pre-registered experiments, execute them in a sandbox, analyze the results, and
red-team the conclusions.

```
mra idea "Does curriculum ordering help small-model math reasoning?"
mra survey   <idea-id>     # literature search -> knowledge base
mra design   <idea-id>     # hypotheses -> pre-registered experiment specs
mra run      <spec-id>     # sandboxed execution -> run records
mra report   <idea-id>     # a memo where every claim cites a source or a run id
```

**Status: implemented (M0–M5).** Every layer in the plan is built and the full loop runs end to
end: `FRAME → SURVEY → CURATE → GROUND → SYNTHESIZE → DESIGN → IMPLEMENT → RUN → ANALYZE → REPORT`,
with `ANALYZE` looping back to `DESIGN` on an inconclusive verdict. `mra auto "<idea>"` drives the
whole thing.

The test suite runs offline — no API key, no network — via a scripted LLM client and cassette-style
HTTP fixtures, and includes an adversarial suite of deliberately defective artifacts (unfalsifiable
decision rules, single-seed deltas, fabricated citations, silent divergence from a reference
implementation) that the system is required to catch.

```bash
pytest                      # unit + integration, offline
ruff check src tests
mypy                        # strict
```

> **Read [`docs/PLAN.md`](docs/PLAN.md) first.** It is the full system plan: architecture, the
> ten-phase research loop, the agent roster and their contracts, the data model, safety and cost
> controls, testing strategy, and build order.

## Layout

| Path | What lives there |
|---|---|
| `src/ml_research_agent/orchestrator/` | Phase machine, project state, routing, budgets, human gates |
| `src/ml_research_agent/agents/` | Framer, Scout, Curator, CodeAnalyst, Planner, Implementer, Runner, Analyst, Critic, Writer |
| `src/ml_research_agent/literature/` | Multi-source retrieval, dedup, ranking, citation snowballing, PDF parsing |
| `src/ml_research_agent/knowledge/` | Ingest, chunking, embeddings, hybrid search, notes, knowledge graph |
| `src/ml_research_agent/code/` | Repo discovery, static analysis, runnable "recipes", license gating |
| `src/ml_research_agent/experiments/` | Specs with pre-registered decision rules, sandboxing, tracking, analysis |
| `src/ml_research_agent/llm/` `tools/` `observability/` `reporting/` | Platform layer |
| `configs/default.yaml` | Every knob, documented |
| `workspace/` | Runtime state — knowledge base, runs, repo cache (gitignored, regenerable) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env      # add ANTHROPIC_API_KEY
```
