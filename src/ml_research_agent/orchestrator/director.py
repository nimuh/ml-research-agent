"""ResearchDirector: top-level controller.

Phases: FRAME -> SURVEY -> CURATE -> GROUND -> DESIGN -> IMPLEMENT -> RUN ->
ANALYZE -> (loop back to DESIGN) -> REPORT. Each phase declares entry
preconditions, the agents it dispatches, and exit criteria the critic scores.
Resumable: every transition writes a checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..agents.base import AgentContext
from ..agents.critic import gate_passed
from ..agents.registry import create
from ..config import Config
from ..errors import BudgetExceeded, GateRejected, MRAError, PhasePreconditionFailed
from ..llm.budget import BudgetTracker
from ..llm.cache import ResponseCache
from ..llm.client import LLMClient
from ..llm.prompts import PromptLibrary
from ..observability.cost import CostLedger
from ..observability.logging import StructuredLogger, configure_logging, get_logger
from ..observability.tracing import Tracer
from ..tools.registry import ToolRegistry, default_registry
from ..types import (
    Citation,
    CritiqueReport,
    ExperimentSpec,
    Idea,
    Phase,
    Provenance,
    ResearchBrief,
    Scale,
    VerdictStatus,
)
from .checkpoint import CheckpointStore
from .gates import GateKeeper, report_packet, run_packet, survey_packet
from .plan import ResearchPlan, build_plan, order_specs, replan
from .router import Router
from .state import EventType, ProjectSnapshot, ProjectState, Synthesis


@dataclass
class PhaseOutcome:
    """What a phase produced, and whether the loop may advance past it."""

    phase: Phase
    ok: bool
    summary: str = ""
    critique: CritiqueReport | None = None
    loop_back_to: Phase | None = None
    data: dict[str, Any] = field(default_factory=dict)


class ResearchDirector:
    """Sequences phases, checks preconditions, scores exits. Holds no domain logic.

    Everything domain-specific lives in the capability layers and the agents;
    this class decides *when* things run, enforces the ladders and ceilings,
    and writes a checkpoint at every transition.
    """

    def __init__(
        self,
        config: Config,
        *,
        state: ProjectState | None = None,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        retriever: Any | None = None,
        gatekeeper: GateKeeper | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.config = config.ensure_paths()
        configure_logging(config.observability, log_dir=config.paths.logs)
        self.logger = logger or get_logger("director")
        # The toggle controls persistence only -- in-memory accounting always
        # runs, because the budget's hard stop depends on it.
        self.ledger = CostLedger(
            config.paths.workspace / "cost.jsonl" if config.observability.cost_ledger else None
        )
        self.budget = BudgetTracker(config.budget, ledger=self.ledger)
        self.tracer = Tracer(enabled=config.observability.trace, logger=self.logger)
        self.cache = ResponseCache(config.paths.cache / "llm", enabled=config.llm.cache_responses)
        self.llm = llm or LLMClient(
            config,
            cache=self.cache,
            budget=self.budget,
            ledger=self.ledger,
            tracer=self.tracer,
            logger=self.logger,
        )
        self.prompts = PromptLibrary()
        self.tools = (
            tools
            if tools is not None
            else default_registry(config, logger=self.logger, tracer=self.tracer)
        )
        self.retriever = retriever
        self.router = Router(config)
        self.gates = GateKeeper(config)
        if gatekeeper is not None:
            self.gates = gatekeeper
        self.checkpoints = CheckpointStore(config.paths.state_dir / "_checkpoints")
        self.state = state or ProjectState.create(config)
        self.plan: ResearchPlan | None = None

    # -- context ------------------------------------------------------------

    def context(self, phase: Phase) -> AgentContext:
        return AgentContext(
            config=self.config,
            llm=self.llm,
            prompts=self.prompts,
            tools=self.tools,
            retriever=self.retriever,
            logger=self.logger.bind(phase=phase.value, project=self.state.project_id),
            tracer=self.tracer,
            phase=phase,
            project_id=self.state.project_id,
        )

    # -- the machine ---------------------------------------------------------

    PHASE_METHODS: dict[Phase, str] = {
        Phase.FRAME: "run_frame",
        Phase.SURVEY: "run_survey",
        Phase.CURATE: "run_curate",
        Phase.GROUND: "run_ground",
        Phase.SYNTHESIZE: "run_synthesize",
        Phase.DESIGN: "run_design",
        Phase.IMPLEMENT: "run_implement",
        Phase.RUN: "run_execute",
        Phase.ANALYZE: "run_analyze",
        Phase.REPORT: "run_report",
    }

    def run(
        self, idea: Idea | str, *, until: Phase = Phase.REPORT, max_loops: int = 4
    ) -> ProjectState:
        """Drive the loop from an idea to a report.

        ANALYZE returning ``inconclusive`` sends control back to DESIGN rather
        than forward -- that backwards edge is the whole reason this is a
        machine and not a pipeline.
        """
        if isinstance(idea, str):
            idea = Idea(text=idea)
        if self.state.snapshot.idea is None:
            self.state.set_idea(idea)

        order = [p for p in Phase]
        index = 0
        loops = 0
        while index < len(order):
            phase = order[index]
            if _phase_rank(phase) > _phase_rank(until):
                break
            outcome = self.run_phase(phase)
            if not outcome.ok:
                if outcome.loop_back_to is not None and loops < max_loops:
                    loops += 1
                    self.logger.info(
                        "loop_back",
                        frm=phase.value,
                        to=outcome.loop_back_to.value,
                        loop=loops,
                        reason=outcome.summary,
                    )
                    index = order.index(outcome.loop_back_to)
                    continue
                self.state.fail_phase(phase, outcome.summary)
                self.logger.error("phase_halted", phase=phase.value, reason=outcome.summary)
                break
            if outcome.loop_back_to is not None and loops < max_loops:
                loops += 1
                index = order.index(outcome.loop_back_to)
                continue
            index += 1

        self.state.save()
        return self.state

    def run_phase(self, phase: Phase) -> PhaseOutcome:
        """Preconditions -> work -> exit criteria -> checkpoint."""
        self.check_preconditions(phase)
        self.state.start_phase(phase)
        method: Callable[[], PhaseOutcome] = getattr(self, self.PHASE_METHODS[phase])
        with self.tracer.span(f"phase:{phase.value}"):
            try:
                outcome = method()
            except BudgetExceeded:
                raise
            except GateRejected as exc:
                self.state.fail_phase(phase, str(exc))
                return PhaseOutcome(phase=phase, ok=False, summary=str(exc))
            except MRAError as exc:
                self.logger.error("phase_error", phase=phase.value, error=str(exc))
                return PhaseOutcome(phase=phase, ok=False, summary=str(exc))

        if outcome.ok:
            self.state.complete_phase(phase, note=outcome.summary)
            self.checkpoints.save(self.state, phase, inputs=self._phase_inputs(phase))
        self.state.save()
        return outcome

    def check_preconditions(self, phase: Phase) -> None:
        """A phase entered without its inputs fails loudly, not subtly."""
        snap = self.state.snapshot
        requirement: dict[Phase, tuple[bool, str]] = {
            Phase.FRAME: (snap.idea is not None, "no idea to frame"),
            Phase.SURVEY: (snap.brief is not None, "SURVEY needs a brief"),
            Phase.CURATE: (bool(snap.paper_keys), "CURATE needs surveyed candidates"),
            Phase.GROUND: (bool(snap.note_ids), "GROUND needs curated notes"),
            Phase.SYNTHESIZE: (bool(snap.note_ids), "SYNTHESIZE needs a populated KB"),
            Phase.DESIGN: (snap.synthesis is not None, "DESIGN needs a synthesis"),
            Phase.IMPLEMENT: (bool(snap.specs), "IMPLEMENT needs at least one spec"),
            Phase.RUN: (bool(snap.specs), "RUN needs specs"),
            Phase.ANALYZE: (bool(snap.runs), "ANALYZE needs runs"),
            Phase.REPORT: (snap.brief is not None, "REPORT needs a brief"),
        }
        ok, message = requirement.get(phase, (True, ""))
        if not ok:
            raise PhasePreconditionFailed(message, phase=phase.value)

    def _phase_inputs(self, phase: Phase) -> Any:
        """What a phase's result depends on -- the resume-invalidation key."""
        snap = self.state.snapshot
        match phase:
            case Phase.FRAME:
                return snap.idea.text if snap.idea else None
            case Phase.SURVEY:
                return snap.brief.query_text if snap.brief else None
            case Phase.CURATE:
                return sorted(snap.paper_keys)
            case Phase.GROUND:
                return sorted(snap.note_ids)
            case Phase.SYNTHESIZE:
                return sorted(snap.note_ids)
            case Phase.DESIGN:
                return snap.synthesis.model_dump(mode="json") if snap.synthesis else None
            case Phase.IMPLEMENT | Phase.RUN:
                return sorted(s.spec_hash for s in snap.specs)
            case Phase.ANALYZE:
                return sorted(r.id for r in snap.runs)
            case _:
                return sorted(v.id for v in snap.verdicts)

    # -- phases --------------------------------------------------------------

    def run_frame(self) -> PhaseOutcome:
        from ..agents.framer import Framer, FramerInput

        idea = self.state.snapshot.idea
        assert idea is not None
        framer = create("framer")
        assert isinstance(framer, Framer)
        draft = framer.run(FramerInput(idea=idea), self.context(Phase.FRAME))
        brief = framer.to_brief(draft, idea, model=self.router.model_for("frame"))
        self.state.set_brief(brief)

        self.plan = build_plan(brief, self.config.experiments)
        for question in self.plan.questions:
            self.state.add_question(question)
        return PhaseOutcome(phase=Phase.FRAME, ok=True, summary=brief.title, data={"brief": brief})

    def run_survey(self) -> PhaseOutcome:
        from ..agents.scout import Scout, ScoutInput

        brief = self._brief()
        scout = create("scout")
        assert isinstance(scout, Scout)
        report = scout.survey(
            ScoutInput(brief=brief, known_keys=self.state.snapshot.paper_keys),
            self.context(Phase.SURVEY),
            self.config,
        )
        limit = self.config.literature.max_candidates
        ranked = report.ranked[:limit]
        if len(report.ranked) > limit:
            self.state.record_truncation(
                "survey candidates", kept=limit, dropped=len(report.ranked) - limit, limit=limit
            )

        # Persist the candidates before recording them. State holds keys; the KB
        # holds the papers, so a resumed CURATE reads from disk rather than
        # re-running the whole fan-out.
        from ..knowledge.store import KnowledgeStore

        with KnowledgeStore(self.config) as store:
            for candidate in ranked:
                store.add_paper(candidate.paper)

        self.state.record_survey([r.paper.key for r in ranked], stats=report.stats)

        approved = self.gates.ask(
            "after_survey",
            survey_packet(
                paper_count=len(ranked),
                kb_count=min(len(ranked), self.config.literature.max_kb_papers),
                missing_seminal=report.missing_seminal,
                cost_usd=self.ledger.total_usd,
                top_titles=[r.paper.title for r in ranked[:15]],
            ),
        )
        self.state.record_gate("after_survey", approved)
        if not approved:
            return PhaseOutcome(phase=Phase.SURVEY, ok=False, summary="survey rejected at gate")

        return PhaseOutcome(
            phase=Phase.SURVEY,
            ok=True,
            summary=f"{len(ranked)} candidates",
            data={"ranked": ranked, "report": report},
        )

    def run_curate(self) -> PhaseOutcome:
        """Screen cheaply, ingest what survives, extract notes with provenance."""
        from ..agents.curator import (
            Curator,
            NoteInput,
            Screener,
            ScreeningInput,
            included,
        )
        from ..knowledge.embed import get_embedder
        from ..knowledge.index import HybridIndex
        from ..knowledge.ingest import Ingestor
        from ..knowledge.store import KnowledgeStore
        from ..literature.fetch import ContentFetcher
        from ..literature.parse import parse_document

        brief = self._brief()
        ranked = self._survey_candidates()
        if not ranked:
            return PhaseOutcome(phase=Phase.CURATE, ok=False, summary="no candidates to curate")

        screener = create("screener")
        assert isinstance(screener, Screener)
        batch = screener.run(
            ScreeningInput(brief=brief, candidates=[r.paper for r in ranked]),
            self.context(Phase.CURATE),
        )
        keys = included(batch.decisions)
        cap = self.config.literature.max_kb_papers
        if len(keys) > cap:
            self.state.record_truncation("kb papers", kept=cap, dropped=len(keys) - cap, limit=cap)
            keys = keys[:cap]

        papers = {r.paper.key: r.paper for r in ranked}
        store = KnowledgeStore(self.config)
        index = HybridIndex(
            self.config.paths.kb_index, get_embedder(self.config), self.config.knowledge
        )
        index.load()
        ingestor = Ingestor(self.config, store, index)
        fetcher = ContentFetcher(self.config, logger=self.logger)
        curator = create("curator")
        assert isinstance(curator, Curator)

        notes = 0
        for key in keys:
            paper = papers.get(key)
            if paper is None:
                continue
            sections = []
            try:
                document = fetcher.fetch_paper(paper)
                if document is not None:
                    sections = parse_document(document)
            except MRAError as exc:  # a paywalled PDF must not stop the phase
                self.logger.warning("fulltext_unavailable", paper=key, error=str(exc))

            result = ingestor.ingest_paper(paper, sections=sections)
            passages = store.list_passages(doc_id=result.doc_id)[:40]
            try:
                draft = curator.run(
                    NoteInput(brief=brief, paper=paper, passages=passages),
                    self.context(Phase.CURATE),
                )
            except MRAError as exc:
                self.logger.warning("note_extraction_failed", paper=key, error=str(exc))
                continue
            note = curator.to_note(draft, NoteInput(brief=brief, paper=paper, passages=passages))
            store.save_note(note)
            self.state.add_note(note.id, paper_key=key)
            notes += 1

        index.save()
        store.close()

        critique = self.critique(
            Phase.CURATE,
            target_id=brief.id,
            artifact={
                "included": keys,
                "excluded": [
                    d.model_dump(mode="json") for d in batch.decisions if d.paper_key not in keys
                ][:40],
                "note_count": notes,
            },
        )
        return PhaseOutcome(
            phase=Phase.CURATE,
            ok=gate_passed(critique),
            summary=f"{notes} notes from {len(keys)} papers",
            critique=critique,
            loop_back_to=None if gate_passed(critique) else Phase.SURVEY,
        )

    def run_ground(self) -> PhaseOutcome:
        """Turn papers into runnable references, or document why we cannot."""
        from ..agents.code_analyst import CodeAnalyst, CodeAnalystInput
        from ..code.analyze import analyze_repo
        from ..code.discover import discover_repos
        from ..code.fetch import fetch_repo
        from ..code.license import license_gate
        from ..knowledge.store import KnowledgeStore

        store = KnowledgeStore(self.config)
        papers = [p for p in store.list_papers() if p.key in set(self.state.snapshot.paper_keys)]
        analyst = create("code_analyst")
        assert isinstance(analyst, CodeAnalyst)

        grounded = 0
        blocked: list[str] = []
        for paper in papers[: self.config.literature.max_kb_papers]:
            repos = discover_repos(paper, self.config, logger=self.logger)
            for repo in repos[: self.config.code.max_repos_per_paper]:
                allowed, reason = license_gate(repo, self.config.code)
                if not allowed:
                    # The gate precedes adaptation, not the other way round.
                    blocked.append(f"{repo.url}: {reason}")
                    self.state.add_todo(f"license blocked {repo.url}: {reason}")
                    continue
                try:
                    fetched = fetch_repo(repo, self.config, logger=self.logger)
                    repo_map = analyze_repo(fetched, self.config)
                except MRAError as exc:
                    self.logger.warning("repo_unavailable", repo=repo.url, error=str(exc))
                    continue
                payload = CodeAnalystInput(
                    repo=fetched,
                    repo_map=repo_map,
                    paper_key=paper.key,
                    paper_title=paper.title,
                )
                draft = analyst.run(payload, self.context(Phase.GROUND))
                recipe = analyst.to_recipe(draft, payload)
                self.state.emit(EventType.REPO_GROUNDED, phase=Phase.GROUND, repo_id=fetched.id)
                self.state.add_recipe(recipe)
                grounded += 1

        store.close()
        summary = f"{grounded} recipes"
        if not grounded:
            # Not a failure: the gap is the finding, and DESIGN can still proceed.
            self.state.add_todo("no runnable reference implementation was found; document the gap")
            summary = "no runnable reference implementation; gap documented"
        return PhaseOutcome(
            phase=Phase.GROUND, ok=True, summary=summary, data={"blocked_licenses": blocked}
        )

    def run_synthesize(self) -> PhaseOutcome:
        from ..agents.synthesizer import SynthesisInput, Synthesizer
        from ..knowledge.graph import KnowledgeGraph
        from ..knowledge.store import KnowledgeStore

        brief = self._brief()
        store = KnowledgeStore(self.config)
        notes = [n for n in store.list_notes() if n.id in set(self.state.snapshot.note_ids)]
        graph = KnowledgeGraph(store)
        graph.build()
        payload = SynthesisInput(
            brief=brief,
            notes=notes or store.list_notes(),
            contradictions=graph.contradictions(),
            coverage_gaps=graph.gaps(),
        )
        synthesizer = create("synthesizer")
        assert isinstance(synthesizer, Synthesizer)
        draft = synthesizer.run(payload, self.context(Phase.SYNTHESIZE))
        store.close()

        self.state.set_synthesis(
            Synthesis(
                settled=[s.statement for s in draft.settled],
                contested=[s.statement for s in draft.contested],
                gaps=[s.statement for s in draft.gaps],
                baselines=draft.baselines,
                benchmarks=draft.benchmarks,
                novelty_assessment=draft.novelty_assessment,
                positioning=draft.positioning,
            )
        )
        critique = self.critique(
            Phase.SYNTHESIZE, target_id=brief.id, artifact=draft.model_dump(mode="json")
        )
        if not draft.novelty_survives:
            # The brief is answered already; revising it beats testing it again.
            self.state.add_todo(
                "novelty did not survive synthesis; revise the brief before designing"
            )
            return PhaseOutcome(
                phase=Phase.SYNTHESIZE,
                ok=False,
                summary="the literature already answers this; revise the brief",
                critique=critique,
                loop_back_to=Phase.FRAME,
            )
        return PhaseOutcome(
            phase=Phase.SYNTHESIZE,
            ok=gate_passed(critique),
            summary=draft.novelty_assessment[:200],
            critique=critique,
        )

    def run_design(self) -> PhaseOutcome:
        from ..agents.experiment_planner import ExperimentPlanner, PlannerInput

        brief = self._brief()
        snap = self.state.snapshot
        synthesis = snap.synthesis
        assert synthesis is not None
        if self.plan is None:
            self.plan = build_plan(brief, self.config.experiments)
            for hypothesis in self.plan.hypotheses:
                self.state.add_hypothesis(hypothesis)
        elif not snap.hypotheses:
            for hypothesis in self.plan.hypotheses:
                self.state.add_hypothesis(hypothesis)

        planner = create("experiment_planner")
        assert isinstance(planner, ExperimentPlanner)
        specs: list[ExperimentSpec] = []
        for hypothesis in self.state.snapshot.open_hypotheses():
            payload = PlannerInput(
                hypothesis=hypothesis,
                baselines=synthesis.baselines,
                benchmarks=synthesis.benchmarks,
                recipes=snap.recipes,
                budget_usd=self.router.budget_for("design"),
                default_seeds=self.config.experiments.default_seeds,
                scale=Scale.SMOKE,
            )
            draft = planner.run(payload, self.context(Phase.DESIGN))
            spec = planner.to_spec(draft, payload, config=self.config.experiments)
            specs.append(spec)
            self.state.add_spec(spec)

        if not specs:
            return PhaseOutcome(phase=Phase.DESIGN, ok=False, summary="no specs produced")

        critique = self.critique(
            Phase.DESIGN,
            target_id=specs[0].id,
            artifact={"specs": [s.model_dump(mode="json") for s in specs]},
        )
        return PhaseOutcome(
            phase=Phase.DESIGN,
            ok=gate_passed(critique),
            summary=f"{len(specs)} pre-registered specs",
            critique=critique,
            data={"specs": specs},
        )

    def run_implement(self) -> PhaseOutcome:
        from ..agents.implementer import Implementer, ImplementerInput
        from ..experiments.workspace import ExperimentWorkspace

        snap = self.state.snapshot
        implementer = create("implementer")
        assert isinstance(implementer, Implementer)
        built = 0
        for spec in order_specs(snap.specs):
            workspace = ExperimentWorkspace.create(self.config, spec)
            recipe = next((r for r in snap.recipes if r.id == spec.recipe_id), None) or (
                snap.recipes[0] if snap.recipes else None
            )
            payload = ImplementerInput(
                spec=spec,
                recipe=recipe,
                workspace_path=str(workspace.path),
                existing_files=workspace.list_files(),
            )
            draft = implementer.run(payload, self.context(Phase.IMPLEMENT))
            workspace.write_files({f.path: f.content for f in draft.files})
            workspace.set_commands(entrypoint=draft.entrypoint, smoke=draft.smoke_command)
            workspace.write_manifest(divergences=draft.divergences, assumptions=draft.assumptions)
            self.state.emit(
                EventType.WORKSPACE_READY,
                phase=Phase.IMPLEMENT,
                spec_id=spec.id,
                path=str(workspace.path),
                code_hash=workspace.code_hash,
                divergences=draft.divergences,
            )
            built += 1
        return PhaseOutcome(
            phase=Phase.IMPLEMENT, ok=built > 0, summary=f"{built} workspaces built"
        )

    def run_execute(self) -> PhaseOutcome:
        """Approve the bill, then run the ladder from the cheapest rung up."""
        from ..experiments.execute import execute_spec

        snap = self.state.snapshot
        # Only what has not already run. On an ANALYZE -> DESIGN loop-back the
        # earlier specs are still in state, and re-running them would spend the
        # budget re-deriving results we already hold.
        specs = order_specs([s for s in snap.specs if not _already_ran(s.id, snap)])
        if not specs:
            return PhaseOutcome(phase=Phase.RUN, ok=True, summary="all specs already executed")

        total_runs = sum(len(s.seeds) * max(len(s.arms), 1) for s in specs)
        estimate = sum(s.budget_usd for s in specs)

        approved = self.gates.ask(
            "before_run",
            run_packet(
                specs=[
                    f"{s.title} [{s.scale.value}] {len(s.seeds)} seeds x {len(s.arms)} arms"
                    for s in specs
                ],
                total_runs=total_runs,
                estimated_usd=estimate,
                cost_usd=self.ledger.total_usd,
                warnings=[
                    f"{s.title}: fewer seeds ({len(s.seeds)}) than the decision rule requires "
                    f"({s.decision_rule.min_seeds})"
                    for s in specs
                    if len(s.seeds) < s.decision_rule.min_seeds
                ],
            ),
        )
        self.state.record_gate("before_run", approved)
        if not approved:
            return PhaseOutcome(phase=Phase.RUN, ok=False, summary="run rejected at gate")

        completed = 0
        for spec in specs:
            records = execute_spec(spec, self.config, state=self.state, logger=self.logger)
            for record in records:
                self.state.record_run(record)
                completed += 1 if record.succeeded else 0
            if not any(r.succeeded for r in records) and spec.scale is Scale.SMOKE:
                # Cheap before expensive: a failed smoke stops the ladder here.
                self.logger.warning("smoke_failed", spec=spec.id)
                break
        return PhaseOutcome(
            phase=Phase.RUN,
            ok=completed > 0,
            summary=f"{completed} successful runs",
        )

    def run_analyze(self) -> PhaseOutcome:
        from ..agents.analyst import Analyst, AnalystInput
        from ..experiments.ablation import propose_followups
        from ..experiments.analysis import analyze_runs, evaluate_decision_rule

        snap = self.state.snapshot
        analyst = create("analyst")
        assert isinstance(analyst, Analyst)
        statuses: list[VerdictStatus] = []

        for spec in snap.specs:
            runs = snap.runs_for(spec.id)
            if not runs or _already_analyzed(spec.id, snap):
                # A spec keeps its runs across a DESIGN loop-back; re-analyzing
                # it would spend a deep-tier call to restate a verdict we hold.
                continue
            result = analyze_runs(spec, runs)
            self.state.add_result(result)
            status, detail = evaluate_decision_rule(spec, result)

            payload = AnalystInput(
                spec=spec, runs=runs, result=result, rule_status=status, rule_detail=detail
            )
            draft = analyst.run(payload, self.context(Phase.ANALYZE))
            verdict = analyst.to_verdict(draft, payload)

            critique = self.critique(
                Phase.ANALYZE,
                target_id=verdict.id,
                artifact={
                    "verdict": verdict.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                    "spec": spec.model_dump(mode="json"),
                },
            )
            # A blocker invalidates the conclusion by definition, so it downgrades
            # the verdict whatever the arithmetic said. Majors do not overturn a
            # measurement, but dropping them would lose the caveats a reader needs.
            surviving = [f for f in critique.findings if f.severity in ("blocker", "major")]
            if surviving:
                update: dict[str, Any] = {
                    "critique_id": critique.id,
                    "threats_to_validity": [
                        *verdict.threats_to_validity,
                        *(f"[{f.severity}] {f.statement}" for f in surviving),
                    ],
                }
                if critique.blockers:
                    update["status"] = VerdictStatus.INCONCLUSIVE
                verdict = verdict.model_copy(update=update)
            self.state.add_verdict(verdict)
            statuses.append(verdict.status)

            hypothesis = snap.hypothesis(spec.hypothesis_id)
            if hypothesis is not None:
                followups = propose_followups(spec, result, verdict, self.config)
                self.state.propose_followups(followups)
                if self.plan is not None:
                    replan(self.plan, verdict, followups, self.config.experiments)
                    updated = next((h for h in self.plan.hypotheses if h.id == hypothesis.id), None)
                    if updated is not None:
                        self.state.update_hypothesis(updated)

        if not statuses:
            return PhaseOutcome(phase=Phase.ANALYZE, ok=False, summary="nothing to analyze")

        inconclusive = [s for s in statuses if s is VerdictStatus.INCONCLUSIVE]
        loop_back = None
        if inconclusive and len(inconclusive) == len(statuses):
            escalated = any(h.status == "escalate" for h in self.state.snapshot.hypotheses)
            loop_back = None if escalated else Phase.DESIGN
        return PhaseOutcome(
            phase=Phase.ANALYZE,
            ok=True,
            summary=", ".join(s.value for s in statuses),
            loop_back_to=loop_back,
        )

    def run_report(self) -> PhaseOutcome:
        from ..agents.writer import Writer, WriterInput
        from ..knowledge.store import KnowledgeStore
        from ..reporting.report import write_report

        brief = self._brief()
        snap = self.state.snapshot

        approved = self.gates.ask(
            "before_report",
            report_packet(
                verdicts=[f"{v.status.value}: {v.reasoning[:120]}" for v in snap.verdicts],
                cost_usd=self.ledger.total_usd,
            ),
        )
        self.state.record_gate("before_report", approved)
        if not approved:
            return PhaseOutcome(phase=Phase.REPORT, ok=False, summary="report rejected at gate")

        store = KnowledgeStore(self.config)
        citations = [
            Citation(key=n.paper_key or n.id, label=n.title, kind="paper")
            for n in store.list_notes()
            if n.id in set(snap.note_ids)
        ]
        citations += [
            Citation(key=r.id, label=f"run {r.arm} seed {r.seed}", kind="run") for r in snap.runs
        ]
        store.close()
        if not citations:
            return PhaseOutcome(
                phase=Phase.REPORT,
                ok=False,
                summary="nothing citable; refusing to write unsupported prose",
            )

        writer = create("writer")
        assert isinstance(writer, Writer)
        payload = WriterInput(
            brief=brief,
            synthesis=snap.synthesis.model_dump(mode="json") if snap.synthesis else {},
            results=snap.results,
            verdicts=snap.verdicts,
            available_citations=citations,
            followups=[f.title for f in snap.followups],
            budget_note=(
                f"${self.ledger.total_usd:.2f} spent across {self.ledger.total_tokens} tokens"
            ),
        )
        draft = writer.run(payload, self.context(Phase.REPORT))
        report = writer.to_report(draft, payload)
        # The memo carries its own provenance: which project produced it and
        # what it cost. A report nobody can trace back is not auditable.
        path = write_report(
            report,
            self.config,
            project_id=self.state.project_id,
            cost_note=f"${self.ledger.total_usd:.2f} across {self.ledger.total_tokens} tokens",
        )
        report = report.model_copy(
            update={
                "path": str(path),
                "provenance": [Provenance(source=f"project:{self.state.project_id}")],
            }
        )
        self.state.set_report(report)
        return PhaseOutcome(phase=Phase.REPORT, ok=True, summary=str(path), data={"path": path})

    # -- shared helpers ------------------------------------------------------

    def critique(self, phase: Phase, *, target_id: str, artifact: dict[str, Any]) -> CritiqueReport:
        """Adversarial review at the four gates the plan names."""
        from ..agents.critic import Critic, CritiqueInput

        critic = create("critic")
        assert isinstance(critic, Critic)
        snap = self.state.snapshot
        payload = CritiqueInput(
            phase=phase,
            target_id=target_id,
            artifact=artifact,
            context={
                "brief": snap.brief.model_dump(mode="json") if snap.brief else {},
                "note_count": len(snap.note_ids),
                "paper_count": len(snap.paper_keys),
                "truncations": snap.truncations[-10:],
            },
        )
        draft = critic.run(payload, self.context(phase))
        report = critic.to_report(draft, payload)
        self.state.add_critique(report)
        return report

    def _brief(self) -> ResearchBrief:
        brief = self.state.snapshot.brief
        if brief is None:
            raise PhasePreconditionFailed("no brief in state")
        return brief

    def _survey_candidates(self) -> list[Any]:
        """Re-rank stored candidates rather than re-fetching them on resume."""
        from ..knowledge.store import KnowledgeStore
        from ..literature.rank import rank_candidates

        store = KnowledgeStore(self.config)
        keys = set(self.state.snapshot.paper_keys)
        papers = [p for p in store.list_papers() if p.key in keys]
        store.close()
        if not papers:
            return []
        return rank_candidates(papers, self._brief(), self.config.literature)

    def close(self) -> None:
        self.llm.close()
        self.state.save()


def _already_ran(spec_id: str, snapshot: ProjectSnapshot) -> bool:
    """True when this spec already has at least one successful run on record."""
    return any(r.spec_id == spec_id and r.succeeded for r in snapshot.runs)


def _already_analyzed(spec_id: str, snapshot: ProjectSnapshot) -> bool:
    """True when this spec's runs already produced a verdict."""
    result_ids = {r.id for r in snapshot.results if r.spec_id == spec_id}
    return any(v.result_id in result_ids for v in snapshot.verdicts)


def _phase_rank(phase: Phase) -> int:
    from ..types import PHASE_ORDER

    return PHASE_ORDER.index(phase)


__all__ = ["PhaseOutcome", "ResearchDirector"]
