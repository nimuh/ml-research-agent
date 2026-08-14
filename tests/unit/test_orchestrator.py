"""Plan, ladder, router, gates and checkpoints -- the control plane's logic,
tested without any LLM call."""

from __future__ import annotations

import pytest

from ml_research_agent.config import Config, ExperimentsConfig
from ml_research_agent.errors import ConfigError, GateRejected
from ml_research_agent.orchestrator.checkpoint import CheckpointStore
from ml_research_agent.orchestrator.director import _already_analyzed, _already_ran
from ml_research_agent.orchestrator.gates import (
    DecisionPacket,
    GateKeeper,
    auto_approve_responder,
    deny_responder,
    run_packet,
    survey_packet,
)
from ml_research_agent.orchestrator.plan import (
    build_ladder,
    build_plan,
    hypotheses_from_brief,
    order_specs,
    questions_from_brief,
    replan,
)
from ml_research_agent.orchestrator.router import Router
from ml_research_agent.orchestrator.state import ProjectSnapshot, ProjectState
from ml_research_agent.types import (
    DatasetRef,
    DecisionRule,
    ExperimentSpec,
    FalsifiableClaim,
    FollowUp,
    Idea,
    MetricDef,
    ModelTier,
    Phase,
    ResearchBrief,
    Result,
    RunRecord,
    RunStatus,
    Scale,
    Verdict,
    VerdictStatus,
)


@pytest.fixture
def config(tmp_path):
    return Config(paths={"workspace": tmp_path / "ws"}).ensure_paths()


@pytest.fixture
def brief():
    return ResearchBrief(
        idea_id="idea_1",
        title="Curriculum ordering",
        problem_statement="Does ordering help small-model math reasoning?",
        claims=[
            FalsifiableClaim(
                statement="ordering improves accuracy",
                falsifier="gain within seed noise",
                confidence=0.6,
            ),
            FalsifiableClaim(
                statement="ordering reduces steps to converge",
                falsifier="no step reduction",
                confidence=0.3,
            ),
        ],
    )


def make_spec(scale=Scale.SMOKE, **overrides) -> ExperimentSpec:
    defaults = dict(
        hypothesis_id="hyp",
        title=f"spec-{scale.value}",
        scale=scale,
        dependent_variables=[MetricDef(name="accuracy")],
        dataset=DatasetRef(name="toy"),
        baselines=["baseline"],
        treatment="treatment",
        decision_rule=DecisionRule(metric="accuracy", comparator=">", threshold=0.01),
    )
    return ExperimentSpec(**{**defaults, **overrides})


class TestPlanDecomposition:
    def test_one_question_per_claim_plus_a_reproduction_question(self, brief):
        questions = questions_from_brief(brief)
        assert len(questions) == len(brief.claims) + 1
        assert any(q.kind == "reproduction" for q in questions)

    def test_hypotheses_carry_the_falsifier_through_unchanged(self, brief):
        hypotheses = hypotheses_from_brief(brief)
        assert [h.falsifier for h in hypotheses] == [c.falsifier for c in brief.claims]

    def test_decomposition_is_deterministic_in_content(self, brief):
        first = [q.text for q in questions_from_brief(brief)]
        second = [q.text for q in questions_from_brief(brief)]
        assert first == second


class TestScaleLadder:
    def test_every_hypothesis_starts_at_smoke(self, brief):
        ladder = build_ladder(hypotheses_from_brief(brief), ExperimentsConfig())
        assert ladder[0].scale is Scale.SMOKE

    def test_all_smoke_rungs_precede_any_small_rung(self, brief):
        ladder = build_ladder(hypotheses_from_brief(brief), ExperimentsConfig())
        scales = [r.scale for r in ladder]
        last_smoke = max(i for i, s in enumerate(scales) if s is Scale.SMOKE)
        first_small = min(i for i, s in enumerate(scales) if s is Scale.SMALL)
        assert last_smoke < first_small

    def test_specs_order_cheapest_first(self):
        specs = [make_spec(Scale.MAIN), make_spec(Scale.SMOKE), make_spec(Scale.SMALL)]
        assert [s.scale for s in order_specs(specs)] == [Scale.SMOKE, Scale.SMALL, Scale.MAIN]


class TestReplan:
    def _verdict(self, plan, status):
        return Verdict(
            hypothesis_id=plan.hypotheses[0].id,
            result_id="result_1",
            status=status,
            reasoning="because",
        )

    def test_refuted_cancels_the_remaining_ladder(self, brief):
        plan = build_plan(brief, ExperimentsConfig())
        target = plan.hypotheses[0].id
        replan(plan, self._verdict(plan, VerdictStatus.REFUTED), [], ExperimentsConfig())
        assert plan.hypotheses[0].status == "refuted"
        assert all(r.status == "cancelled" for r in plan.rungs_for(target))

    def test_inconclusive_schedules_the_best_followup(self, brief):
        plan = build_plan(brief, ExperimentsConfig())
        before = len(plan.ladder)
        followups = [
            FollowUp(
                title="cheap diagnostic",
                rationale="r",
                expected_information_gain=0.9,
                estimated_cost_usd=1.0,
            ),
            FollowUp(
                title="expensive scale-up",
                rationale="r",
                expected_information_gain=0.95,
                estimated_cost_usd=100.0,
            ),
        ]
        replan(
            plan, self._verdict(plan, VerdictStatus.INCONCLUSIVE), followups, ExperimentsConfig()
        )
        assert len(plan.ladder) == before + 1
        assert plan.ladder[0].estimated_usd == 1.0  # gain per dollar, not gain

    def test_two_inconclusive_rounds_escalate_instead_of_looping(self, brief):
        config = ExperimentsConfig(max_inconclusive_rounds=2)
        plan = build_plan(brief, config)
        for _ in range(2):
            replan(plan, self._verdict(plan, VerdictStatus.INCONCLUSIVE), [], config)
        assert plan.hypotheses[0].status == "escalate"


class TestRouter:
    def test_cost_policy_assigns_deep_to_judgment_and_fast_to_extraction(self, config):
        router = Router(config)
        assert router.route("frame").tier is ModelTier.DEEP
        assert router.route("critique").tier is ModelTier.DEEP
        assert router.route("design").tier is ModelTier.DEEP
        assert router.route("screen").tier is ModelTier.FAST
        assert router.route("extract_note").tier is ModelTier.STANDARD

    def test_concurrency_never_exceeds_the_global_ceiling(self, tmp_path):
        # Both directions: the global ceiling binds when it is the smaller of
        # the two, and the route's own limit binds when it is. Testing only the
        # first would pass even if the route's value were ignored entirely.
        clamped = Config(paths={"workspace": tmp_path}, llm={"max_concurrency": 2})
        assert Router(clamped).concurrency_for("screen") == 2

        roomy = Config(paths={"workspace": tmp_path}, llm={"max_concurrency": 1000})
        route = Router(roomy).route("screen")
        assert Router(roomy).concurrency_for("screen") == route.max_concurrency
        assert route.max_concurrency < 1000, "the route must carry its own limit"

    def test_task_budget_is_clamped_to_the_project_budget(self, tmp_path):
        tight = Config(paths={"workspace": tmp_path}, budget={"usd_per_project": 1.0})
        assert Router(tight).budget_for("implement") == 1.0

        roomy = Config(paths={"workspace": tmp_path}, budget={"usd_per_project": 10_000.0})
        route = Router(roomy).route("implement")
        assert Router(roomy).budget_for("implement") == route.budget_usd
        assert route.budget_usd < 10_000.0, "the route must carry its own ceiling"

    def test_unknown_task_is_an_error_not_a_default(self, config):
        with pytest.raises(ConfigError):
            Router(config).route("do_something_clever")

    def test_critic_runs_at_the_four_declared_gates(self, config):
        router = Router(config)
        expected = {Phase.CURATE, Phase.SYNTHESIZE, Phase.DESIGN, Phase.ANALYZE}
        assert {p for p in Phase if router.critiques_after(p)} == expected


class TestGates:
    def _packet(self):
        return DecisionPacket(name="before_run", phase=Phase.RUN, question="ok?", summary="s")

    def test_disabled_gate_passes_without_asking(self, tmp_path):
        config = Config(paths={"workspace": tmp_path}, gates={"before_run": False})
        keeper = GateKeeper(config, responder=deny_responder)
        assert keeper.ask("before_run", self._packet()) is True

    def test_enabled_gate_consults_the_responder(self, tmp_path):
        config = Config(paths={"workspace": tmp_path}, gates={"before_run": True})
        assert (
            GateKeeper(config, responder=deny_responder).ask("before_run", self._packet()) is False
        )
        assert (
            GateKeeper(config, responder=auto_approve_responder).ask("before_run", self._packet())
            is True
        )

    def test_auto_approve_is_a_config_flip(self, tmp_path):
        config = Config(
            paths={"workspace": tmp_path}, gates={"before_run": True, "auto_approve": True}
        )
        assert (
            GateKeeper(config, responder=deny_responder).ask("before_run", self._packet()) is True
        )

    def test_require_raises_on_rejection(self, tmp_path):
        config = Config(paths={"workspace": tmp_path}, gates={"before_run": True})
        with pytest.raises(GateRejected):
            GateKeeper(config, responder=deny_responder).require("before_run", self._packet())

    def test_survey_packet_surfaces_missing_seminal_work_as_a_warning(self):
        packet = survey_packet(
            paper_count=100,
            kb_count=60,
            missing_seminal=["Attention Is All You Need"],
            cost_usd=1.0,
            top_titles=["a", "b"],
        )
        assert any("Attention" in w for w in packet.warnings)
        assert "60" in packet.render()

    def test_run_packet_shows_the_bill_before_the_spend(self):
        packet = run_packet(
            specs=["s1"], total_runs=9, estimated_usd=12.5, cost_usd=3.0, warnings=[]
        )
        rendered = packet.render()
        assert "$12.50" in rendered and "$3.00" in rendered


class TestBudgetIsAHardStop:
    """A ceiling that degrades into a soft phase failure is not a ceiling.

    ``run_phase`` catches ``MRAError`` and turns it into ``ok=False``. Budget
    exhaustion must escape that handler, so the ordering of the two except
    clauses is load-bearing and gets its own test.
    """

    def _director(self, config, boom):
        from ml_research_agent.llm.client import FakeLLMClient
        from ml_research_agent.orchestrator.director import ResearchDirector

        director = ResearchDirector(config, llm=FakeLLMClient([], config=config))
        director.state = ProjectState.create(config, idea=Idea(text="x"))
        monkeypatched = director.PHASE_METHODS | {Phase.FRAME: "_boom"}
        director.PHASE_METHODS = monkeypatched
        director._boom = boom  # type: ignore[attr-defined]
        return director

    def test_budget_exhaustion_stops_the_run_instead_of_failing_the_phase(self, config):
        from ml_research_agent.errors import BudgetExceeded

        def boom():
            raise BudgetExceeded(
                "project USD ceiling reached", scope="project", limit=1.0, spent=1.5
            )

        director = self._director(config, boom)
        with pytest.raises(BudgetExceeded):
            director.run_phase(Phase.FRAME)
        director.close()

    def test_an_ordinary_phase_error_is_a_recorded_failure_not_a_crash(self, config):
        from ml_research_agent.errors import SourceError

        def boom():
            raise SourceError("arxiv is down")

        director = self._director(config, boom)
        outcome = director.run_phase(Phase.FRAME)
        assert outcome.ok is False
        assert "arxiv is down" in outcome.summary
        director.close()


class TestTruncationIsNeverSilent:
    """PLAN §6: every fan-out cap is enforced *and* the truncation is logged.

    ``ProjectState.record_truncation`` has its own test; this one drives the
    real SURVEY call site, which is where a cap can quietly start dropping
    candidates without anything in the audit trail saying so.
    """

    def test_exceeding_max_candidates_is_recorded_by_the_survey_phase(self, tmp_path, monkeypatch):
        from ml_research_agent.agents.scout import Scout, SurveyReport
        from ml_research_agent.llm.client import FakeLLMClient
        from ml_research_agent.orchestrator.director import ResearchDirector
        from ml_research_agent.types import Paper, RankedPaper

        config = Config(
            paths={"workspace": tmp_path / "ws"},
            literature={"max_candidates": 3},
            gates={"auto_approve": True},
        ).ensure_paths()

        ranked = [
            RankedPaper(paper=Paper(title=f"P{i}", arxiv_id=f"24{i:02d}.1"), score=1.0 - i / 10)
            for i in range(10)
        ]
        monkeypatch.setattr(
            Scout,
            "survey",
            lambda self, payload, ctx, cfg: SurveyReport(ranked=ranked, stats={"candidates": 10}),
        )

        director = ResearchDirector(config, llm=FakeLLMClient([], config=config))
        director.state = ProjectState.create(config, idea=Idea(text="x"))
        director.state.set_brief(
            ResearchBrief(
                idea_id="idea_1",
                title="t",
                problem_statement="p",
                claims=[FalsifiableClaim(statement="s", falsifier="f")],
            )
        )

        outcome = director.run_survey()
        director.close()

        assert outcome.ok
        assert len(director.state.snapshot.paper_keys) == 3, "the cap must actually be enforced"
        truncations = director.state.snapshot.truncations
        assert truncations, "a dropped candidate that is not logged is a silent cap"
        assert truncations[0]["dropped"] == 7
        assert truncations[0]["limit"] == 3


class TestCheckpoints:
    def test_unchanged_inputs_make_a_phase_skippable(self, config):
        state = ProjectState.create(config, idea=Idea(text="x"))
        store = CheckpointStore(config.paths.state_dir / "_cp")
        store.save(state, Phase.SURVEY, inputs={"query": "a"})

        assert store.is_valid(state.project_id, Phase.SURVEY, inputs={"query": "a"})
        assert not store.is_valid(state.project_id, Phase.SURVEY, inputs={"query": "b"})

    def test_invalidation_cascades_downstream(self, config):
        state = ProjectState.create(config, idea=Idea(text="x"))
        store = CheckpointStore(config.paths.state_dir / "_cp")
        for phase in (Phase.FRAME, Phase.SURVEY, Phase.CURATE, Phase.DESIGN):
            store.save(state, phase, inputs={"k": phase.value})

        store.invalidate(state.project_id, Phase.CURATE)
        remaining = store.phases_completed(state.project_id)
        assert Phase.FRAME in remaining and Phase.SURVEY in remaining
        assert Phase.CURATE not in remaining and Phase.DESIGN not in remaining

    def test_snapshot_round_trips_through_a_checkpoint(self, config, brief):
        state = ProjectState.create(config, idea=Idea(text="x"))
        state.set_brief(brief)
        store = CheckpointStore(config.paths.state_dir / "_cp")
        store.save(state, Phase.FRAME, inputs="x")

        loaded = store.load(state.project_id, Phase.FRAME)
        assert loaded is not None
        _, snapshot = loaded
        assert snapshot.brief is not None and snapshot.brief.title == brief.title


class TestLoopBackDoesNotRepeatWork:
    """An ANALYZE -> DESIGN loop-back must not re-spend the budget on work already done."""

    def _snapshot(self, *, ran: bool, analyzed: bool) -> ProjectSnapshot:
        spec = make_spec()
        snapshot = ProjectSnapshot(project_id="p", specs=[spec])
        if ran:
            snapshot.runs.append(
                RunRecord(
                    spec_id=spec.id,
                    spec_hash=spec.spec_hash,
                    seed=0,
                    arm="treatment",
                    status=RunStatus.COMPLETED,
                )
            )
        if analyzed:
            result = Result(spec_id=spec.id, spec_hash=spec.spec_hash)
            snapshot.results.append(result)
            snapshot.verdicts.append(
                Verdict(
                    hypothesis_id=spec.hypothesis_id,
                    result_id=result.id,
                    status=VerdictStatus.INCONCLUSIVE,
                    reasoning="r",
                )
            )
        return snapshot

    def test_a_spec_with_a_successful_run_is_not_re_executed(self):
        snapshot = self._snapshot(ran=True, analyzed=False)
        assert _already_ran(snapshot.specs[0].id, snapshot)

    def test_a_spec_with_no_successful_run_is_still_pending(self):
        snapshot = self._snapshot(ran=False, analyzed=False)
        assert not _already_ran(snapshot.specs[0].id, snapshot)

    def test_a_failed_run_does_not_count_as_executed(self):
        spec = make_spec()
        snapshot = ProjectSnapshot(
            project_id="p",
            specs=[spec],
            runs=[
                RunRecord(
                    spec_id=spec.id,
                    spec_hash=spec.spec_hash,
                    seed=0,
                    status=RunStatus.FAILED,
                )
            ],
        )
        assert not _already_ran(spec.id, snapshot)

    def test_a_spec_with_a_verdict_is_not_re_analyzed(self):
        snapshot = self._snapshot(ran=True, analyzed=True)
        assert _already_analyzed(snapshot.specs[0].id, snapshot)

    def test_a_result_without_a_verdict_is_still_unanalyzed(self):
        snapshot = self._snapshot(ran=True, analyzed=False)
        assert not _already_analyzed(snapshot.specs[0].id, snapshot)
