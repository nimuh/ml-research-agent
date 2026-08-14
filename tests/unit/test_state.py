"""ProjectState: the snapshot must be a pure fold over the append-only log."""

from __future__ import annotations

import pytest

from ml_research_agent.config import Config
from ml_research_agent.orchestrator.state import EventType, ProjectState, Synthesis
from ml_research_agent.types import (
    DatasetRef,
    DecisionRule,
    ExperimentSpec,
    FalsifiableClaim,
    Idea,
    MetricDef,
    Phase,
    ResearchBrief,
    RunRecord,
    RunStatus,
)


@pytest.fixture
def config(tmp_path):
    return Config(paths={"workspace": tmp_path / "ws"}).ensure_paths()


@pytest.fixture
def brief():
    return ResearchBrief(
        idea_id="idea_1",
        title="Curriculum ordering",
        problem_statement="Does ordering help?",
        claims=[
            FalsifiableClaim(statement="ordering helps", falsifier="no gain beyond seed noise")
        ],
    )


def make_spec(**overrides) -> ExperimentSpec:
    defaults = dict(
        hypothesis_id="hyp_1",
        title="smoke",
        dependent_variables=[MetricDef(name="accuracy")],
        dataset=DatasetRef(name="toy"),
        baselines=["baseline"],
        treatment="treatment",
        decision_rule=DecisionRule(metric="accuracy", comparator=">", threshold=0.01),
    )
    return ExperimentSpec(**{**defaults, **overrides})


def test_snapshot_is_a_fold_over_the_log(config, brief):
    state = ProjectState.create(config, idea=Idea(text="test idea"))
    state.set_brief(brief)
    state.record_survey(["arxiv:1", "arxiv:2"])
    state.add_note("note_1", paper_key="arxiv:1")

    reloaded = ProjectState.load(config, state.project_id)

    assert reloaded.snapshot.brief is not None
    assert reloaded.snapshot.brief.title == brief.title
    assert reloaded.snapshot.paper_keys == ["arxiv:1", "arxiv:2"]
    assert reloaded.snapshot.note_ids == ["note_1"]


def test_replay_is_deterministic(config, brief):
    state = ProjectState.create(config, idea=Idea(text="x"))
    state.set_brief(brief)
    state.add_spec(make_spec())
    state.start_phase(Phase.DESIGN)
    state.complete_phase(Phase.DESIGN)

    before = state.snapshot.model_dump(mode="json")
    after = state.replay().model_dump(mode="json")
    assert before == after


def test_events_are_append_only(config):
    state = ProjectState.create(config, idea=Idea(text="x"))
    state.add_todo("first")
    state.add_todo("second")

    lines = state.log_path.read_text(encoding="utf-8").strip().splitlines()
    seqs = [e.seq for e in state.events()]
    assert seqs == sorted(seqs)
    assert len(lines) == len(seqs)


def test_duplicate_survey_keys_are_not_double_counted(config):
    state = ProjectState.create(config, idea=Idea(text="x"))
    state.record_survey(["arxiv:1"])
    state.record_survey(["arxiv:1", "arxiv:2"])
    assert state.snapshot.paper_keys == ["arxiv:1", "arxiv:2"]


def test_rerecording_a_run_replaces_rather_than_duplicates(config):
    state = ProjectState.create(config, idea=Idea(text="x"))
    spec = make_spec()
    run = RunRecord(spec_id=spec.id, spec_hash=spec.spec_hash, seed=0, cost_usd=1.0)
    state.record_run(run)
    state.record_run(run.model_copy(update={"status": RunStatus.COMPLETED}))

    assert len(state.snapshot.runs) == 1
    assert state.snapshot.runs[0].status is RunStatus.COMPLETED


def test_truncation_is_recorded_not_silent(config):
    state = ProjectState.create(config, idea=Idea(text="x"))
    state.record_truncation("survey candidates", kept=60, dropped=240, limit=60)

    assert state.snapshot.truncations
    truncation = state.snapshot.truncations[0]
    assert truncation["dropped"] == 240
    assert state.events(type=EventType.TRUNCATED)


def test_corrupt_log_line_does_not_brick_the_project(config, brief):
    state = ProjectState.create(config, idea=Idea(text="x"))
    state.set_brief(brief)
    with open(state.log_path, "a", encoding="utf-8") as handle:
        handle.write("{not json at all\n")
    state.add_todo("still works")

    reloaded = ProjectState.load(config, state.project_id)
    assert reloaded.snapshot.brief is not None
    assert "still works" in reloaded.snapshot.todos


def test_synthesis_round_trips(config):
    state = ProjectState.create(config, idea=Idea(text="x"))
    state.set_synthesis(Synthesis(settled=["a"], contested=["b"], gaps=["c"], baselines=["sgd"]))

    reloaded = ProjectState.load(config, state.project_id)
    assert reloaded.snapshot.synthesis is not None
    assert reloaded.snapshot.synthesis.baselines == ["sgd"]


def test_list_projects_finds_created_projects(config):
    a = ProjectState.create(config, idea=Idea(text="a"))
    b = ProjectState.create(config, idea=Idea(text="b"))
    assert {a.project_id, b.project_id} <= set(ProjectState.list_projects(config))
