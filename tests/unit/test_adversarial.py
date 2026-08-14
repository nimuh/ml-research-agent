"""The adversarial suite: deliberately defective artifacts the system must catch.

These are regression tests for the system's *judgment*. Each case is a way a
fluent-but-wrong result gets through in real research -- a leaky eval, an unfair
baseline, a single-seed delta, a fabricated citation, an unfalsifiable rule --
and each is checked against the guard that is supposed to stop it.

The guards tested here are deterministic code, not model judgment: a defence
that only works when the model is having a good day is not a defence.
"""

from __future__ import annotations

import pytest

from ml_research_agent.agents.analyst import AnalystInput, VerdictDraft
from ml_research_agent.agents.critic import Critic, CritiqueDraft, CritiqueInput, gate_passed
from ml_research_agent.agents.curator import Curator, NoteDraft, NoteInput
from ml_research_agent.agents.experiment_planner import ExperimentPlanner, PlannerInput, SpecDraft
from ml_research_agent.agents.framer import BriefDraft, Framer, FramerInput
from ml_research_agent.agents.implementer import (
    GeneratedFile,
    ImplementationDraft,
    Implementer,
    ImplementerInput,
)
from ml_research_agent.agents.writer import ReportDraft, Writer, WriterInput
from ml_research_agent.experiments.analysis import analyze_runs, evaluate_decision_rule
from ml_research_agent.experiments.design import power_check
from ml_research_agent.experiments.hypothesis import validate_hypothesis
from ml_research_agent.types import (
    Citation,
    DatasetRef,
    DecisionRule,
    ExperimentSpec,
    FalsifiableClaim,
    Finding,
    Hypothesis,
    Idea,
    Metric,
    MetricDef,
    MetricReport,
    Paper,
    Passage,
    Phase,
    Provenance,
    Recipe,
    RecipeStep,
    ReportSection,
    ResearchBrief,
    Result,
    RunRecord,
    RunStatus,
    SuccessCriterion,
    Variable,
    VerdictStatus,
)

CTX = None  # every guard under test is pure; no agent context is needed


@pytest.fixture
def brief():
    return ResearchBrief(
        idea_id="i",
        title="Curriculum ordering",
        problem_statement="Does ordering help?",
        claims=[FalsifiableClaim(statement="ordering helps", falsifier="gain within seed noise")],
        scope_limits=["small models only"],
        search_terms=["curriculum", "ordering", "reasoning"],
    )


@pytest.fixture
def hypothesis(brief):
    return Hypothesis(
        brief_id=brief.id,
        statement="ordering improves accuracy",
        rationale="easier examples first stabilize early training",
        prediction="+2 points accuracy",
        falsifier="gain is within seed noise",
    )


def spec(**overrides) -> ExperimentSpec:
    defaults = dict(
        hypothesis_id="hyp",
        title="t",
        dependent_variables=[MetricDef(name="accuracy")],
        dataset=DatasetRef(name="toy"),
        baselines=["baseline"],
        treatment="treatment",
        controls=["matched compute"],
        seeds=[0, 1, 2],
        decision_rule=DecisionRule(metric="accuracy", comparator=">", threshold=0.01, min_seeds=3),
    )
    return ExperimentSpec(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# 1. Unfalsifiable framing
# ---------------------------------------------------------------------------


class TestUnfalsifiableFraming:
    def _draft(self, **overrides) -> BriefDraft:
        defaults = dict(
            title="t",
            problem_statement="p",
            claims=[
                FalsifiableClaim(
                    statement="X helps Y", falsifier="no measurable gain over baseline"
                )
            ],
            success_criteria=[SuccessCriterion(metric="accuracy", comparator=">", threshold=0.8)],
            scope_limits=["small scale only"],
            search_terms=["a", "b", "c"],
        )
        return BriefDraft(**{**defaults, **overrides})

    def test_a_well_formed_brief_passes(self, brief):
        assert Framer().validate_output(self._draft(), FramerInput(idea=Idea(text="x")), CTX) == []

    def test_a_falsifier_that_restates_the_claim_is_caught(self):
        draft = self._draft(claims=[FalsifiableClaim(statement="X helps Y", falsifier="X helps Y")])
        problems = Framer().validate_output(draft, FramerInput(idea=Idea(text="x")), CTX)
        assert any("restates" in p for p in problems)

    def test_an_empty_falsifier_is_caught(self):
        claim = FalsifiableClaim(statement="X helps Y", falsifier="x")
        object.__setattr__(claim, "falsifier", "   ")
        problems = Framer().validate_output(
            self._draft(claims=[claim]), FramerInput(idea=Idea(text="x")), CTX
        )
        assert any("no falsifier" in p for p in problems)

    def test_a_brief_with_no_measurable_criterion_is_caught(self):
        problems = Framer().validate_output(
            self._draft(success_criteria=[]), FramerInput(idea=Idea(text="x")), CTX
        )
        assert any("measurable" in p for p in problems)

    def test_unbounded_scope_is_caught(self):
        problems = Framer().validate_output(
            self._draft(scope_limits=[]), FramerInput(idea=Idea(text="x")), CTX
        )
        assert any("scope" in p for p in problems)

    @pytest.mark.parametrize(
        "falsifier",
        ["if it does not work", "if the results are bad", "if performance does not improve"],
    )
    def test_vacuous_falsifiers_are_rejected_at_the_hypothesis_level(self, brief, falsifier):
        h = Hypothesis(
            brief_id=brief.id, statement="s", rationale="r", prediction="p", falsifier=falsifier
        )
        assert any("vacuous" in problem for problem in validate_hypothesis(h))


# ---------------------------------------------------------------------------
# 2. Unfalsifiable / unfair experiment design
# ---------------------------------------------------------------------------


class TestDefectiveDesign:
    def _draft(self, **overrides) -> SpecDraft:
        defaults = dict(
            title="t",
            independent_variables=[Variable(name="ordering", values=["random", "curriculum"])],
            dependent_variables=[MetricDef(name="accuracy")],
            controls=["matched compute", "identical data budget"],
            dataset=DatasetRef(name="toy"),
            baselines=["random ordering"],
            treatment="curriculum ordering",
            seeds=[0, 1, 2],
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.02, min_seeds=3
            ),
            estimated_cost_usd=1.0,
        )
        return SpecDraft(**{**defaults, **overrides})

    def _payload(self, hypothesis) -> PlannerInput:
        return PlannerInput(hypothesis=hypothesis, budget_usd=5.0)

    def test_a_sound_design_passes(self, hypothesis):
        assert (
            ExperimentPlanner().validate_output(self._draft(), self._payload(hypothesis), CTX) == []
        )

    def test_a_rule_on_an_unmeasured_metric_is_caught(self, hypothesis):
        draft = self._draft(
            decision_rule=DecisionRule(metric="f1", comparator=">", threshold=0.02, min_seeds=3)
        )
        problems = ExperimentPlanner().validate_output(draft, self._payload(hypothesis), CTX)
        assert any("not among the measured metrics" in p for p in problems)

    def test_an_unfalsifiable_zero_threshold_rule_is_caught(self, hypothesis):
        draft = self._draft(
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.0, min_effect_size=0.0, min_seeds=3
            )
        )
        problems = ExperimentPlanner().validate_output(draft, self._payload(hypothesis), CTX)
        assert any("unfalsifiable" in p for p in problems)

    def test_a_single_seed_rule_is_caught(self, hypothesis):
        draft = self._draft(
            seeds=[0],
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.02, min_seeds=1
            ),
        )
        problems = ExperimentPlanner().validate_output(draft, self._payload(hypothesis), CTX)
        assert any("noise" in p for p in problems)

    def test_fewer_seeds_than_the_rule_demands_is_caught(self, hypothesis):
        draft = self._draft(
            seeds=[0, 1],
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.02, min_seeds=5
            ),
        )
        problems = ExperimentPlanner().validate_output(draft, self._payload(hypothesis), CTX)
        assert any("requires 5" in p for p in problems)

    def test_treatment_compared_against_itself_is_caught(self, hypothesis):
        draft = self._draft(baselines=["curriculum ordering"], treatment="curriculum ordering")
        problems = ExperimentPlanner().validate_output(draft, self._payload(hypothesis), CTX)
        assert any("vacuous" in p for p in problems)

    def test_a_design_over_budget_is_caught(self, hypothesis):
        draft = self._draft(estimated_cost_usd=500.0)
        problems = ExperimentPlanner().validate_output(draft, self._payload(hypothesis), CTX)
        assert any("exceeds" in p for p in problems)

    def test_underpowered_design_is_warned_about_before_it_runs(self):
        warnings = power_check(
            spec(
                seeds=[0, 1, 2],
                decision_rule=DecisionRule(
                    metric="accuracy",
                    comparator=">",
                    threshold=0.01,
                    min_effect_size=0.05,
                    min_seeds=3,
                ),
            )
        )
        assert any("can detect" in w for w in warnings)

    def test_no_minimum_effect_size_is_warned_about(self):
        assert any("minimum effect size" in w for w in power_check(spec()))


# ---------------------------------------------------------------------------
# 3. Single-seed noise read as a result
# ---------------------------------------------------------------------------


class TestSingleSeedNoise:
    def _runs(self, spec_, values: dict[str, list[float]]) -> list[RunRecord]:
        return [
            RunRecord(
                spec_id=spec_.id,
                spec_hash=spec_.spec_hash,
                seed=seed,
                arm=arm,
                status=RunStatus.COMPLETED,
                metrics=[Metric(name="accuracy", value=value, arm=arm)],
            )
            for arm, vals in values.items()
            for seed, value in enumerate(vals)
        ]

    def test_a_huge_single_seed_delta_is_still_inconclusive(self):
        s = spec(seeds=[0])
        runs = self._runs(s, {"baseline": [0.10], "treatment": [0.99]})
        status, _ = evaluate_decision_rule(s, analyze_runs(s, runs))
        assert status is VerdictStatus.INCONCLUSIVE

    def test_noise_larger_than_the_effect_is_reported_on_the_result(self):
        s = spec()
        runs = self._runs(s, {"baseline": [0.30, 0.60, 0.90], "treatment": [0.32, 0.62, 0.92]})
        result = analyze_runs(s, runs)
        assert any("overlap" in note for note in result.notes)

    def test_dropping_failed_runs_is_never_silent(self):
        s = spec()
        runs = self._runs(s, {"baseline": [0.5, 0.5, 0.5], "treatment": [0.9, 0.9]})
        runs.append(
            RunRecord(
                spec_id=s.id,
                spec_hash=s.spec_hash,
                seed=2,
                arm="treatment",
                status=RunStatus.FAILED,
            )
        )
        result = analyze_runs(s, runs)
        assert result.failed_runs == 1
        assert any("failed" in n for n in result.notes)

    def test_analyst_must_report_spread_and_threats(self):
        s = spec()
        payload = AnalystInput(
            spec=s,
            runs=self._runs(s, {"baseline": [0.5], "treatment": [0.6]}),
            result=Result(spec_id=s.id, spec_hash=s.spec_hash),
            rule_status=VerdictStatus.INCONCLUSIVE,
        )
        from ml_research_agent.agents.analyst import Analyst

        bad = VerdictDraft(reasoning="it worked", seed_variance_note="", threats_to_validity=["x"])
        problems = Analyst().validate_output(bad, payload, CTX)
        assert any("seed-variance" in p for p in problems)


# ---------------------------------------------------------------------------
# 4. Fabricated citations and unsupported prose
# ---------------------------------------------------------------------------


class TestFabricatedCitations:
    def _payload(self, brief) -> WriterInput:
        return WriterInput(
            brief=brief,
            available_citations=[Citation(key="arxiv:1234.5678", label="Real Paper", kind="paper")],
        )

    def _draft(self, citations: list[Citation]) -> ReportDraft:
        return ReportDraft(
            title="t",
            abstract="a",
            headline_finding="the treatment did not beat the baseline",
            sections=[
                ReportSection(title="Results", body="b", citations=citations),
                ReportSection(title="Threats to validity", body="b"),
            ],
        )

    def test_a_report_citing_only_real_sources_passes(self, brief):
        draft = self._draft([Citation(key="arxiv:1234.5678", label="Real Paper")])
        assert Writer().validate_output(draft, self._payload(brief), CTX) == []

    def test_an_invented_citation_key_is_caught(self, brief):
        draft = self._draft([Citation(key="arxiv:9999.0000", label="Invented Paper")])
        problems = Writer().validate_output(draft, self._payload(brief), CTX)
        assert any("fabricated" in p for p in problems)

    def test_a_results_section_with_no_citations_is_caught(self, brief):
        problems = Writer().validate_output(self._draft([]), self._payload(brief), CTX)
        assert any("no citations" in p for p in problems)

    def test_a_report_without_threats_to_validity_is_caught(self, brief):
        draft = ReportDraft(
            title="t",
            abstract="a",
            headline_finding="h",
            sections=[
                ReportSection(
                    title="Results",
                    body="b",
                    citations=[Citation(key="arxiv:1234.5678", label="Real Paper")],
                )
            ],
        )
        problems = Writer().validate_output(draft, self._payload(brief), CTX)
        assert any("threats to validity" in p for p in problems)


# ---------------------------------------------------------------------------
# 5. Notes that assert what no passage says
# ---------------------------------------------------------------------------


class TestUnsupportedNotes:
    def _payload(self, brief) -> NoteInput:
        return NoteInput(
            brief=brief,
            paper=Paper(title="A Paper", arxiv_id="1234.5678"),
            passages=[
                Passage(
                    id="psg_1", doc_id="doc_1", text="we report 91.2 accuracy", section="Results"
                )
            ],
        )

    def test_a_note_citing_a_real_passage_passes(self, brief):
        draft = NoteDraft(
            summary="s",
            relevance_to_brief="directly tests the same claim",
            citations=[Provenance(source="psg_1", locator="Results")],
        )
        assert Curator().validate_output(draft, self._payload(brief), CTX) == []

    def test_a_note_citing_a_passage_that_does_not_exist_is_caught(self, brief):
        draft = NoteDraft(
            summary="s",
            relevance_to_brief="r",
            citations=[Provenance(source="psg_hallucinated")],
        )
        problems = Curator().validate_output(draft, self._payload(brief), CTX)
        assert any("unknown passage" in p for p in problems)

    def test_a_metric_with_a_fabricated_source_is_caught(self, brief):
        draft = NoteDraft(
            summary="s",
            relevance_to_brief="r",
            metrics=[
                MetricReport(
                    name="accuracy", value=99.9, provenance=Provenance(source="psg_invented")
                )
            ],
            citations=[Provenance(source="psg_1")],
        )
        problems = Curator().validate_output(draft, self._payload(brief), CTX)
        assert any("unknown passage" in p for p in problems)

    def test_the_type_system_itself_refuses_a_note_without_provenance(self):
        from ml_research_agent.types import Note

        with pytest.raises(ValueError, match="provenance"):
            Note(type="paper", title="t", summary="s")


# ---------------------------------------------------------------------------
# 6. Silent divergence from the reference implementation
# ---------------------------------------------------------------------------


class TestSilentDivergence:
    def _payload(self) -> ImplementerInput:
        return ImplementerInput(
            spec=spec(),
            recipe=Recipe(repo_id="repo_1", smoke=[RecipeStep(command="python train.py --smoke")]),
            workspace_path="/tmp/ws",
        )

    def _draft(self, **overrides) -> ImplementationDraft:
        defaults = dict(
            files=[
                GeneratedFile(path="train.py", content="# baseline and treatment arms\nprint('x')")
            ],
            entrypoint="python train.py --arm baseline --seed 0",
            smoke_command="python train.py --smoke",
        )
        return ImplementationDraft(**{**defaults, **overrides})

    def test_adaptation_with_no_stated_divergence_is_caught(self):
        draft = self._draft(adapted_from_recipe=True, divergences=[])
        problems = Implementer().validate_output(draft, self._payload(), CTX)
        assert any("silent divergence" in p for p in problems)

    def test_adaptation_with_stated_divergences_passes(self):
        draft = self._draft(
            adapted_from_recipe=True, divergences=["reduced steps to 100 for smoke scale"]
        )
        assert Implementer().validate_output(draft, self._payload(), CTX) == []

    def test_a_path_escaping_the_workspace_is_caught(self):
        draft = self._draft(files=[GeneratedFile(path="../../etc/cron.d/evil", content="x")])
        problems = Implementer().validate_output(draft, self._payload(), CTX)
        assert any("escapes the workspace" in p for p in problems)

    def test_an_arm_with_no_implementation_is_caught(self):
        draft = self._draft(
            files=[GeneratedFile(path="train.py", content="print('only one path')")]
        )
        problems = Implementer().validate_output(draft, self._payload(), CTX)
        assert any("no code path implements" in p for p in problems)

    def test_no_smoke_command_is_caught(self):
        draft = self._draft(smoke_command=" ")
        problems = Implementer().validate_output(draft, self._payload(), CTX)
        assert any("smoke" in p for p in problems)


# ---------------------------------------------------------------------------
# 7. The Critic gate itself
# ---------------------------------------------------------------------------


class TestCriticGate:
    def _payload(self) -> CritiqueInput:
        return CritiqueInput(phase=Phase.DESIGN, target_id="spec_1", artifact={"k": "v"})

    def test_a_blocker_fails_the_gate_regardless_of_confidence(self):
        draft = CritiqueDraft(
            findings=[
                Finding(
                    severity="blocker",
                    category="leakage",
                    statement="test set is in training data",
                    suggested_fix="resplit by document",
                )
            ],
            summary="leaky eval",
            would_you_stake_your_reputation=True,
        )
        report = Critic().to_report(draft, self._payload())
        assert not report.passed
        assert not gate_passed(report)

    def test_a_clean_artifact_passes(self):
        draft = CritiqueDraft(findings=[], summary="held up", would_you_stake_your_reputation=True)
        report = Critic().to_report(draft, self._payload())
        assert report.passed and report.score == 1.0

    def test_accumulated_majors_can_fail_the_gate(self):
        draft = CritiqueDraft(
            findings=[
                Finding(
                    severity="major",
                    category="baseline",
                    statement=f"issue {i}",
                    suggested_fix="fix",
                )
                for i in range(3)
            ],
            summary="several problems",
            would_you_stake_your_reputation=True,
        )
        report = Critic().to_report(draft, self._payload())
        assert report.score < 0.5 and not report.passed

    def test_doubt_must_be_substantiated_with_a_finding(self):
        # Declining to stand behind the artifact while listing nothing wrong is
        # rejected: the score is computed from findings, so an objection that
        # names no finding cannot move the gate and is just a shrug on record.
        #
        # NOTE: this checks the *draft validator*, not the phase gate. What
        # `would_you_stake_your_reputation=False` does downstream is inconsistent
        # today -- `CritiqueReport.passed` honours it, `gate_passed()` ignores
        # it, and the director uses neither. See the review notes.
        draft = CritiqueDraft(findings=[], summary="s", would_you_stake_your_reputation=False)
        problems = Critic().validate_output(draft, self._payload(), CTX)
        assert any("no finding" in p for p in problems)

    def test_a_blocker_without_a_fix_is_rejected(self):
        draft = CritiqueDraft(
            findings=[
                Finding(severity="blocker", category="c", statement="broken", suggested_fix="")
            ],
            summary="s",
            would_you_stake_your_reputation=False,
        )
        problems = Critic().validate_output(draft, self._payload(), CTX)
        assert any("proposes no fix" in p for p in problems)


# ---------------------------------------------------------------------------
# 8. The guards are wired, not merely present
# ---------------------------------------------------------------------------


class TestGuardsAreWiredIntoTheAgentLoop:
    """Every case above calls ``validate_output`` directly.

    That leaves one way for the whole suite to pass while the system is
    defenceless: ``BaseAgent.run`` not calling the validator at all. These
    tests drive the real loop with a scripted client so a deleted call site
    fails here rather than in production.
    """

    def _ctx(self, responses):
        from ml_research_agent.agents.base import AgentContext
        from ml_research_agent.config import Config
        from ml_research_agent.llm.client import FakeLLMClient
        from ml_research_agent.llm.prompts import PromptLibrary

        config = Config()
        return AgentContext(
            config=config,
            llm=FakeLLMClient(responses, config=config),
            prompts=PromptLibrary(),
        )

    def _draft(self, **overrides) -> BriefDraft:
        defaults = dict(
            title="t",
            problem_statement="p",
            claims=[FalsifiableClaim(statement="X helps Y", falsifier="no gain over baseline")],
            success_criteria=[SuccessCriterion(metric="accuracy", comparator=">", threshold=0.8)],
            scope_limits=["small scale only"],
            search_terms=["a", "b", "c"],
        )
        return BriefDraft(**{**defaults, **overrides})

    def test_an_output_that_fails_validation_is_sent_back_for_repair(self):
        # Unbounded scope on the first attempt, corrected on the second.
        ctx = self._ctx([self._draft(scope_limits=[]), self._draft()])

        result = Framer().run(FramerInput(idea=Idea(text="does X help?")), ctx)

        assert result.scope_limits == ["small scale only"]
        assert len(ctx.llm.calls) == 2, "an invalid output must trigger exactly one repair round"

    def test_an_output_that_stays_invalid_raises_rather_than_being_returned(self):
        from ml_research_agent.errors import AgentFailure

        ctx = self._ctx([self._draft(scope_limits=[]), self._draft(scope_limits=[])])

        with pytest.raises(AgentFailure) as excinfo:
            Framer().run(FramerInput(idea=Idea(text="does X help?")), ctx)

        assert excinfo.value.context["agent"] == "framer"
        assert excinfo.value.context["violations"]

    def test_the_repair_turn_names_the_violation(self):
        ctx = self._ctx([self._draft(scope_limits=[]), self._draft()])

        Framer().run(FramerInput(idea=Idea(text="does X help?")), ctx)

        repair_prompt = str(ctx.llm.calls[1]["messages"][-1]["content"])
        assert "scope" in repair_prompt, "a repair that does not name the defect cannot converge"


class TestTheGateActuallyGatesTheDirector:
    """`gate_passed` must be the rule the phase machine applies, not a spare part.

    A `CritiqueReport` scores findings by severity, but a gate wired to blockers
    alone silently discards that score -- a pile of major defects would then read
    as a clean pass, which is exactly the "fluent enough to be convincing while
    wrong" failure the Critic exists to prevent.
    """

    def _report(self, findings: list[Finding], *, stake: bool = True) -> object:
        payload = CritiqueInput(phase=Phase.DESIGN, target_id="spec_1", artifact={})
        draft = CritiqueDraft(findings=findings, summary="s", would_you_stake_your_reputation=stake)
        return Critic().to_report(draft, payload)

    def _major(self, n: int) -> list[Finding]:
        return [
            Finding(
                severity="major", category="baseline", statement=f"defect {i}", suggested_fix="fix"
            )
            for i in range(n)
        ]

    def test_accumulated_majors_fail_the_gate(self):
        report = self._report(self._major(3))
        assert not report.blockers, "no blocker -- a blocker-only gate would pass this"
        assert report.score < 0.5
        assert not gate_passed(report)

    def test_a_couple_of_majors_still_pass(self):
        # The floor must not be so tight that ordinary critique halts the loop.
        report = self._report(self._major(1))
        assert gate_passed(report)

    def test_a_single_blocker_fails_the_gate(self):
        report = self._report(
            [
                Finding(
                    severity="blocker",
                    category="leakage",
                    statement="leaky",
                    suggested_fix="resplit",
                )
            ]
        )
        assert not gate_passed(report)

    def test_minor_findings_do_not_halt_the_loop(self):
        report = self._report(
            [Finding(severity="minor", category="style", statement=f"nit {i}") for i in range(3)]
        )
        assert gate_passed(report)

    def test_the_floor_is_configurable(self):
        report = self._report(self._major(1))
        assert gate_passed(report, min_score=0.5)
        assert not gate_passed(report, min_score=0.95)
