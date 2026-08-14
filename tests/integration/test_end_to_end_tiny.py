"""One narrow idea, 5 papers, no repo, a seconds-long smoke experiment.

Exercises every phase FRAME -> REPORT with a scripted LLM and a real sandboxed
run. No network, no API key. This is the test that catches a phase boundary
breaking; the unit tests catch the logic inside a phase.
"""

from __future__ import annotations

import json
import re

import pytest

from ml_research_agent.agents.analyst import VerdictDraft
from ml_research_agent.agents.code_analyst import RecipeDraft
from ml_research_agent.agents.critic import CritiqueDraft
from ml_research_agent.agents.curator import NoteDraft, ScreeningBatch
from ml_research_agent.agents.experiment_planner import SpecDraft
from ml_research_agent.agents.framer import BriefDraft
from ml_research_agent.agents.implementer import GeneratedFile, ImplementationDraft
from ml_research_agent.agents.scout import QueryExpansion
from ml_research_agent.agents.synthesizer import SupportedStatement, SynthesisDraft
from ml_research_agent.agents.writer import ReportDraft
from ml_research_agent.config import Config
from ml_research_agent.llm.client import FakeLLMClient
from ml_research_agent.orchestrator.director import ResearchDirector
from ml_research_agent.types import (
    Author,
    Citation,
    DatasetRef,
    DecisionRule,
    FalsifiableClaim,
    Finding,
    Idea,
    MetricDef,
    Paper,
    Phase,
    Provenance,
    RankedPaper,
    ReportSection,
    ScreeningDecision,
    ScreeningDecisionKind,
    SuccessCriterion,
    Variable,
    VerdictStatus,
)

# A real experiment script. It is deterministic given the seed, which is what
# makes the reproducibility assertion meaningful rather than decorative.
TRAIN_PY = """
import argparse, json, os, random

parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="baseline")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=".")
parser.add_argument("--smoke", action="store_true")
args, _ = parser.parse_known_args()

random.seed(args.seed)
# "curriculum" genuinely beats "random" here, by more than the seed jitter.
base = 0.60 if args.arm == "random_order" else 0.70
accuracy = base + random.uniform(-0.005, 0.005)

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "metrics.json"), "w") as handle:
    json.dump({"accuracy": accuracy}, handle)
print(f"arm={args.arm} seed={args.seed} accuracy={accuracy:.4f}")
"""


def papers() -> list[RankedPaper]:
    """Five synthetic candidates, standing in for the literature fan-out."""
    return [
        RankedPaper(
            paper=Paper(
                title=f"Curriculum Learning Study {i}",
                abstract=(
                    f"We study ordering effects on small models, part {i}. "
                    "We report accuracy on a toy split."
                ),
                authors=[Author(name=f"Author {i}")],
                year=2020 + i,
                arxiv_id=f"24{i:02d}.0000{i}",
                citation_count=100 * i,
            ),
            score=1.0 - i * 0.1,
            rank=i,
        )
        for i in range(1, 6)
    ]


def _offered_citation_keys(messages) -> list[str]:
    """Pull the `- [key] label` lines the Writer prompt lists as permissible."""
    text = " ".join(str(m.get("content", "")) for m in messages)
    # Anchored on the "(kind)" suffix so verdict bullets like "- [SUPPORTED]"
    # are not mistaken for citation keys.
    return re.findall(r"- \[([^\]]+)\] [^\n]*\((?:paper|run|repo|note)\)", text) or ["none-offered"]


def scripted(**request) -> object:
    """Dispatch on the output schema the agent asked for.

    Keyed on a field unique to each draft model, so the script does not depend
    on call ordering -- a test that breaks when a phase gains one extra LLM call
    is testing the wrong thing.
    """
    params = request["params"]
    if not params.get("tool_choice"):
        # The evidence-gathering turn, not the commit turn: answering in prose
        # ends the tool loop, which is what an agent that needs no tools does.
        return "I have what I need; producing the result now."

    tools = params.get("tools") or []
    properties = set((tools[0].get("input_schema", {}) if tools else {}).get("properties", {}))

    if "problem_statement" in properties and "claims" in properties:
        return BriefDraft(
            title="Curriculum ordering for small-model math reasoning",
            problem_statement="Whether example ordering changes final accuracy at matched compute.",
            claims=[
                FalsifiableClaim(
                    statement="Curriculum ordering improves accuracy at matched compute",
                    falsifier="The gain is within seed-to-seed noise",
                    confidence=0.5,
                )
            ],
            success_criteria=[
                SuccessCriterion(metric="accuracy", comparator=">", threshold=0.02, dataset="toy")
            ],
            scope_limits=["small models only", "single toy dataset"],
            search_terms=["curriculum learning", "example ordering", "training dynamics"],
        )
    if "seminal_works" in properties:
        return QueryExpansion(
            search_terms=["curriculum learning", "example ordering", "data scheduling"],
            seminal_works=["Curriculum Learning Study 1"],
        )
    if "decisions" in properties:
        return ScreeningBatch(
            decisions=[
                ScreeningDecision(
                    paper_key=p.paper.key,
                    decision=ScreeningDecisionKind.INCLUDE
                    if i < 3
                    else ScreeningDecisionKind.EXCLUDE,
                    reason="directly on the ordering question"
                    if i < 3
                    else "adjacent but not informative",
                    relevance=0.9 if i < 3 else 0.2,
                )
                for i, p in enumerate(papers())
            ]
        )
    if "relevance_to_brief" in properties:
        return NoteDraft(
            summary="Studies ordering effects and reports accuracy on a toy split.",
            task="math reasoning",
            method="curriculum ordering",
            datasets=["toy"],
            baselines=["random order"],
            relevance_to_brief="Tests the same ordering mechanism the brief asks about.",
            confidence=0.6,
            citations=[Provenance(source="abstract", locator="abstract")],
        )
    if "runnable" in properties:
        return RecipeDraft(summary="n/a", smoke=[], runnable=False, blocking_issues=["no repo"])
    if "novelty_survives" in properties:
        return SynthesisDraft(
            settled=[
                SupportedStatement(
                    statement="Ordering matters in some regimes",
                    support=[Provenance(source="arxiv:2401.00001")],
                )
            ],
            baselines=["random_order"],
            benchmarks=["toy"],
            novelty_assessment="Open at small scale; prior work does not settle it.",
            novelty_survives=True,
        )
    if "decision_rule" in properties:
        return SpecDraft(
            title="Curriculum vs random ordering at smoke scale",
            independent_variables=[
                Variable(name="ordering", values=["random_order", "curriculum"])
            ],
            dependent_variables=[MetricDef(name="accuracy", higher_is_better=True)],
            controls=["matched compute", "identical data budget", "same seeds"],
            dataset=DatasetRef(name="toy", split="test"),
            baselines=["random_order"],
            treatment="curriculum",
            seeds=[0, 1, 2],
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.02, min_seeds=3, max_p_value=0.05
            ),
            estimated_cost_usd=0.1,
            max_runtime_minutes=2,
        )
    if "entrypoint" in properties:
        return ImplementationDraft(
            files=[GeneratedFile(path="train.py", content=TRAIN_PY, purpose="both arms")],
            entrypoint="python3 train.py --arm {arm} --seed {seed} --out {out}",
            smoke_command="python3 train.py --arm {arm} --seed {seed} --out {out} --smoke",
            metrics_path="metrics.json",
        )
    if "seed_variance_note" in properties:
        return VerdictDraft(
            reasoning="Curriculum ordering beat random ordering by about 10 points on every seed.",
            seed_variance_note="Spread within each arm is under 0.01, far smaller than the gap.",
            threats_to_validity=["single toy dataset", "smoke scale only"],
            confidence=0.7,
        )
    if "would_you_stake_your_reputation" in properties:
        return CritiqueDraft(
            findings=[], summary="No blocking defect found.", would_you_stake_your_reputation=True
        )
    if "headline_finding" in properties:
        # Cite a key actually offered in the prompt. The Writer rejects anything
        # else, which is the fabricated-citation guard doing its job.
        offered = _offered_citation_keys(request["messages"])
        return ReportDraft(
            title="Curriculum ordering at smoke scale",
            abstract="We tested whether ordering helps and found a consistent gain.",
            headline_finding="Curriculum ordering improved accuracy by ~10 points across 3 seeds.",
            sections=[
                ReportSection(
                    title="Results",
                    body="The treatment beat the baseline on every seed.",
                    citations=[Citation(key=offered[0], label="evidence")],
                    order=1,
                ),
                ReportSection(
                    title="Threats to validity", body="Toy dataset, smoke scale.", order=2
                ),
            ],
        )
    raise AssertionError(f"unscripted request for schema with properties {sorted(properties)[:8]}")


@pytest.fixture
def director(tmp_path, monkeypatch):
    config = Config(
        paths={"workspace": tmp_path / "ws"},
        gates={"auto_approve": True},
        llm={"cache_responses": False},
        offline=True,
        experiments={"max_runtime_minutes": 3, "default_seeds": 3},
    ).ensure_paths()

    # Stand in for the literature fan-out; the source adapters have their own
    # cassette tests and must never be hit here.
    monkeypatch.setattr(
        "ml_research_agent.literature.search_literature",
        lambda brief, cfg, **kw: (papers(), {"sources": ["fake"], "candidates": 5}),
    )
    # No full text: ingestion falls back to abstracts, which is the honest
    # behaviour for a paywalled paper and exercises that path.
    monkeypatch.setattr(
        "ml_research_agent.literature.fetch.ContentFetcher.fetch_paper", lambda self, paper: None
    )

    llm = FakeLLMClient(scripted, config=config)
    director = ResearchDirector(config, llm=llm)
    yield director
    director.close()


def _citable_run_id(director):
    return next(r.id for r in director.state.snapshot.runs if r.succeeded)


@pytest.mark.integration
def test_full_loop_produces_an_evidence_backed_memo(director, monkeypatch):
    state = director.run(Idea(text="Does curriculum ordering help small-model math reasoning?"))
    snap = state.snapshot

    # FRAME
    assert snap.brief is not None
    assert snap.brief.claims[0].falsifier

    # SURVEY + CURATE
    assert len(snap.paper_keys) == 5
    assert len(snap.note_ids) == 3, "only the three included papers should become notes"

    # SYNTHESIZE + DESIGN
    assert snap.synthesis is not None and snap.synthesis.baselines == ["random_order"]
    assert snap.specs, "DESIGN must pre-register at least one spec"
    spec = snap.specs[0]
    assert spec.decision_rule.statement, (
        "a spec without a pre-registered rule must not survive DESIGN"
    )

    # RUN -- real subprocess execution inside the sandbox
    assert snap.runs, "RUN must produce run records"
    assert all(r.succeeded for r in snap.runs), [
        r.failure_detail for r in snap.runs if not r.succeeded
    ]
    assert {r.arm for r in snap.runs} == {"random_order", "curriculum"}
    assert {r.seed for r in snap.runs} == {0, 1, 2}

    # ANALYZE -- the rule is applied, not argued
    assert snap.verdicts
    verdict = snap.verdicts[-1]
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.decision_rule_statement
    assert verdict.threats_to_validity

    # REPORT
    assert snap.report is not None
    assert snap.report.path and snap.report.path.endswith(".md")


@pytest.mark.integration
def test_every_run_carries_the_reproducibility_tuple(director):
    director.run(Idea(text="Does curriculum ordering help?"), until=Phase.RUN)
    runs = director.state.snapshot.runs
    assert runs
    for run in runs:
        spec_hash, code_hash, _env_hash, arm, _seed = run.repro_key
        assert spec_hash and code_hash, "a run without spec and code hashes is not reproducible"
        assert arm, "the arm is part of the run's identity"
        assert run.workspace_path


@pytest.mark.integration
def test_same_spec_and_seed_reproduce_the_same_metrics(director):
    """The reproducibility contract, executed rather than asserted."""
    from ml_research_agent.experiments.execute import execute_run
    from ml_research_agent.experiments.workspace import ExperimentWorkspace

    director.run(Idea(text="Does curriculum ordering help?"), until=Phase.IMPLEMENT)
    spec = director.state.snapshot.specs[0]
    workspace = ExperimentWorkspace.create(director.config, spec)

    first = execute_run(spec, workspace, arm="curriculum", seed=0, config=director.config)
    second = execute_run(spec, workspace, arm="curriculum", seed=0, config=director.config)

    assert first.succeeded and second.succeeded
    assert first.repro_key == second.repro_key
    assert first.metric("accuracy") == second.metric("accuracy")


@pytest.mark.integration
def test_the_workspace_manifest_alone_describes_the_run(director):
    director.run(Idea(text="Does curriculum ordering help?"), until=Phase.IMPLEMENT)
    spec = director.state.snapshot.specs[0]

    from ml_research_agent.experiments.workspace import ExperimentWorkspace

    manifest = ExperimentWorkspace.create(director.config, spec).manifest()
    assert manifest["spec_hash"] == spec.spec_hash
    assert manifest["code_hash"]
    assert manifest["entrypoint"] and manifest["smoke"]
    assert "train.py" in manifest["files"]


@pytest.mark.integration
def test_state_replays_to_the_same_snapshot(director):
    from ml_research_agent.orchestrator.state import ProjectState

    state = director.run(Idea(text="Does curriculum ordering help?"), until=Phase.ANALYZE)
    before = json.loads(
        json.dumps(state.snapshot.model_dump(mode="json"), sort_keys=True, default=str)
    )

    reloaded = ProjectState.load(director.config, state.project_id)
    after = json.loads(
        json.dumps(reloaded.snapshot.model_dump(mode="json"), sort_keys=True, default=str)
    )
    assert before == after


def _scripted_with_critique(draft: CritiqueDraft):
    """The end-to-end script, with the Critic's verdict swapped out."""

    def respond(**request) -> object:
        params = request["params"]
        tools = params.get("tools") or []
        properties = set((tools[0].get("input_schema", {}) if tools else {}).get("properties", {}))
        if params.get("tool_choice") and "would_you_stake_your_reputation" in properties:
            return draft
        return scripted(**request)

    return respond


@pytest.mark.integration
def test_a_critique_of_accumulated_majors_stops_the_loop(tmp_path, monkeypatch):
    """The Critic's score must gate the phase machine, not just its blockers.

    Three major findings and no blocker: a gate keyed on `critique.blockers`
    alone reads this as clean and runs the experiments anyway. Driving the real
    loop is what pins that — asserting on the shape of the gate expression would
    pass for any regression phrased differently from the one literal it forbids.
    """
    config = Config(
        paths={"workspace": tmp_path / "ws"},
        gates={"auto_approve": True},
        llm={"cache_responses": False},
        offline=True,
        experiments={"max_runtime_minutes": 3},
    ).ensure_paths()
    monkeypatch.setattr(
        "ml_research_agent.literature.search_literature",
        lambda brief, cfg, **kw: (papers(), {"sources": ["fake"]}),
    )
    monkeypatch.setattr(
        "ml_research_agent.literature.fetch.ContentFetcher.fetch_paper", lambda self, paper: None
    )

    majors = CritiqueDraft(
        findings=[
            Finding(
                severity="major",
                category="baseline",
                statement=f"the baseline is undertuned in respect {i}",
                suggested_fix="match the tuning budget across arms",
            )
            for i in range(3)
        ],
        summary="Several substantial defects, none individually fatal.",
        would_you_stake_your_reputation=True,
    )
    llm = FakeLLMClient(_scripted_with_critique(majors), config=config)
    director = ResearchDirector(config, llm=llm)

    state = director.run(Idea(text="Does curriculum ordering help?"))
    snap = state.snapshot
    director.close()

    critique = snap.critiques[-1]
    assert not critique.blockers, "the scenario under test is majors-without-blockers"
    assert critique.score < 0.5

    assert not snap.runs, "a failed critique gate must not spend compute on experiments"
    assert snap.report is None
    assert Phase.RUN not in snap.completed_phases
