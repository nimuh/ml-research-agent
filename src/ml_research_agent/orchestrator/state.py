"""ProjectState: the single serializable source of truth for one research project.

Holds brief, open questions, KB pointers, hypotheses, specs, runs, results,
open TODOs, and a transcript of decisions. Append-only event log + derived
snapshot so any step can be replayed or audited.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..errors import MRAError
from ..types import (
    CritiqueReport,
    ExperimentSpec,
    FollowUp,
    Hypothesis,
    Idea,
    Phase,
    Question,
    Recipe,
    Report,
    ResearchBrief,
    Result,
    RunRecord,
    Verdict,
    new_id,
    utcnow,
)
from ..utils.io import append_jsonl, ensure_dir, read_json, read_jsonl, write_json


class EventType(StrEnum):
    """Every mutation of project state is one of these, and only these."""

    PROJECT_CREATED = "project_created"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    IDEA_SET = "idea_set"
    BRIEF_SET = "brief_set"
    QUESTION_ADDED = "question_added"
    QUESTION_ANSWERED = "question_answered"
    PAPERS_SURVEYED = "papers_surveyed"
    NOTE_ADDED = "note_added"
    REPO_GROUNDED = "repo_grounded"
    RECIPE_ADDED = "recipe_added"
    SYNTHESIS_SET = "synthesis_set"
    HYPOTHESIS_ADDED = "hypothesis_added"
    HYPOTHESIS_UPDATED = "hypothesis_updated"
    SPEC_ADDED = "spec_added"
    WORKSPACE_READY = "workspace_ready"
    RUN_RECORDED = "run_recorded"
    RESULT_ADDED = "result_added"
    VERDICT_ADDED = "verdict_added"
    CRITIQUE_ADDED = "critique_added"
    FOLLOWUP_PROPOSED = "followup_proposed"
    REPORT_WRITTEN = "report_written"
    GATE_REQUESTED = "gate_requested"
    GATE_ANSWERED = "gate_answered"
    AGENT_CALLED = "agent_called"
    TOOL_CALLED = "tool_called"
    COST_RECORDED = "cost_recorded"
    TRUNCATED = "truncated"
    TODO_ADDED = "todo_added"
    NOTE_MESSAGE = "message"


class Event(BaseModel):
    """One immutable entry in the log. The log is the truth; state is a fold over it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: new_id("evt"))
    seq: int = 0
    at: datetime = Field(default_factory=utcnow)
    type: EventType
    phase: Phase | None = None
    agent: str | None = None
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class Synthesis(BaseModel):
    """The Synthesizer's map of the field, kept in state so DESIGN can read it."""

    model_config = ConfigDict(extra="forbid")

    settled: list[str] = Field(default_factory=list)
    contested: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    novelty_assessment: str = ""
    positioning: str = ""


class ProjectSnapshot(BaseModel):
    """Derived view. Never edited directly -- it is rebuilt by folding events."""

    model_config = ConfigDict(extra="forbid")

    project_id: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    phase: Phase = Phase.FRAME
    completed_phases: list[Phase] = Field(default_factory=list)

    idea: Idea | None = None
    brief: ResearchBrief | None = None
    questions: list[Question] = Field(default_factory=list)
    paper_keys: list[str] = Field(default_factory=list)
    note_ids: list[str] = Field(default_factory=list)
    repo_ids: list[str] = Field(default_factory=list)
    recipes: list[Recipe] = Field(default_factory=list)
    synthesis: Synthesis | None = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    specs: list[ExperimentSpec] = Field(default_factory=list)
    runs: list[RunRecord] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    critiques: list[CritiqueReport] = Field(default_factory=list)
    followups: list[FollowUp] = Field(default_factory=list)
    report: Report | None = None

    todos: list[str] = Field(default_factory=list)
    gate_responses: dict[str, bool] = Field(default_factory=dict)
    truncations: list[dict[str, Any]] = Field(default_factory=list)
    cost_usd: float = 0.0
    event_count: int = 0

    # -- convenience accessors the director and agents lean on ---------------

    def hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return next((h for h in self.hypotheses if h.id == hypothesis_id), None)

    def spec(self, spec_id: str) -> ExperimentSpec | None:
        return next((s for s in self.specs if s.id == spec_id), None)

    def runs_for(self, spec_id: str) -> list[RunRecord]:
        return [r for r in self.runs if r.spec_id == spec_id]

    def result_for(self, spec_id: str) -> Result | None:
        matches = [r for r in self.results if r.spec_id == spec_id]
        return matches[-1] if matches else None

    def verdict_for(self, hypothesis_id: str) -> Verdict | None:
        matches = [v for v in self.verdicts if v.hypothesis_id == hypothesis_id]
        return matches[-1] if matches else None

    def latest_critique(self, phase: Phase) -> CritiqueReport | None:
        matches = [c for c in self.critiques if c.phase is phase]
        return matches[-1] if matches else None

    def open_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.status in ("open", "testing", "inconclusive")]


class ProjectState:
    """Append-only event log plus the snapshot folded from it.

    Two rules make replay and audit work: nothing mutates the snapshot except
    :meth:`_apply`, and every mutation is written to the log *before* it is
    reflected in memory. A crash therefore loses at most the in-flight event.
    """

    def __init__(self, project_id: str, root: Path) -> None:
        self.project_id = project_id
        self.root = ensure_dir(root)
        self.log_path = self.root / "events.jsonl"
        self.snapshot_path = self.root / "snapshot.json"
        self.snapshot = ProjectSnapshot(project_id=project_id)
        self._seq = 0

    # -- lifecycle -----------------------------------------------------------

    @classmethod
    def create(
        cls, config: Config, *, project_id: str | None = None, idea: Idea | None = None
    ) -> Self:
        pid = project_id or new_id("proj")
        state = cls(pid, config.paths.state_dir / pid)
        state.emit(EventType.PROJECT_CREATED, message=f"project {pid} created")
        if idea is not None:
            state.set_idea(idea)
        return state

    @classmethod
    def load(cls, config: Config, project_id: str) -> Self:
        root = config.paths.state_dir / project_id
        if not root.exists():
            raise MRAError("no such project", project_id=project_id, path=str(root))
        state = cls(project_id, root)
        state.replay()
        return state

    @classmethod
    def list_projects(cls, config: Config) -> list[str]:
        root = config.paths.state_dir
        if not root.exists():
            return []
        return sorted(
            p.name for p in root.iterdir() if p.is_dir() and (p / "events.jsonl").exists()
        )

    def replay(self) -> ProjectSnapshot:
        """Rebuild the snapshot from the log alone. The audit guarantee, executable."""
        self.snapshot = ProjectSnapshot(project_id=self.project_id)
        self._seq = 0
        for record in read_jsonl(self.log_path):
            try:
                event = Event.model_validate(record)
            except Exception:  # a corrupt line must not brick a project
                continue
            self._seq = max(self._seq, event.seq)
            self._apply(event)
        return self.snapshot

    # -- writing -------------------------------------------------------------

    def append(self, event: Event) -> Event:
        self._seq += 1
        event = event.model_copy(update={"seq": self._seq})
        append_jsonl(self.log_path, event.model_dump(mode="json"))
        self._apply(event)
        return event

    def emit(
        self,
        type: EventType,
        *,
        phase: Phase | None = None,
        agent: str | None = None,
        message: str = "",
        **payload: Any,
    ) -> Event:
        return self.append(
            Event(
                type=type,
                phase=phase or self.snapshot.phase,
                agent=agent,
                message=message,
                payload=payload,
            )
        )

    def save(self) -> Path:
        """Persist the derived snapshot. Purely an optimization for fast reads."""
        write_json(self.snapshot_path, self.snapshot.model_dump(mode="json"))
        return self.snapshot_path

    # -- typed mutations (thin wrappers so callers never hand-roll payloads) --

    def set_idea(self, idea: Idea) -> None:
        self.emit(EventType.IDEA_SET, message=idea.text, idea=idea.model_dump(mode="json"))

    def set_brief(self, brief: ResearchBrief) -> None:
        self.emit(EventType.BRIEF_SET, message=brief.title, brief=brief.model_dump(mode="json"))

    def add_question(self, question: Question) -> None:
        self.emit(
            EventType.QUESTION_ADDED,
            message=question.text,
            question=question.model_dump(mode="json"),
        )

    def record_survey(self, paper_keys: list[str], stats: dict[str, Any] | None = None) -> None:
        self.emit(
            EventType.PAPERS_SURVEYED,
            phase=Phase.SURVEY,
            message=f"{len(paper_keys)} candidates",
            paper_keys=paper_keys,
            stats=stats or {},
        )

    def add_note(self, note_id: str, paper_key: str | None = None) -> None:
        self.emit(
            EventType.NOTE_ADDED,
            phase=Phase.CURATE,
            message=f"note for {paper_key or note_id}",
            note_id=note_id,
            paper_key=paper_key,
        )

    def add_recipe(self, recipe: Recipe) -> None:
        self.emit(
            EventType.RECIPE_ADDED,
            phase=Phase.GROUND,
            message=recipe.summary[:120],
            recipe=recipe.model_dump(mode="json"),
        )

    def set_synthesis(self, synthesis: Synthesis) -> None:
        self.emit(
            EventType.SYNTHESIS_SET,
            phase=Phase.SYNTHESIZE,
            message=(
                f"{len(synthesis.settled)} settled, {len(synthesis.contested)} contested, "
                f"{len(synthesis.gaps)} gaps"
            ),
            synthesis=synthesis.model_dump(mode="json"),
        )

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        self.emit(
            EventType.HYPOTHESIS_ADDED,
            phase=Phase.DESIGN,
            message=hypothesis.statement,
            hypothesis=hypothesis.model_dump(mode="json"),
        )

    def update_hypothesis(self, hypothesis: Hypothesis) -> None:
        self.emit(
            EventType.HYPOTHESIS_UPDATED,
            message=hypothesis.status,
            hypothesis=hypothesis.model_dump(mode="json"),
        )

    def add_spec(self, spec: ExperimentSpec) -> None:
        self.emit(
            EventType.SPEC_ADDED,
            phase=Phase.DESIGN,
            message=spec.title,
            spec=spec.model_dump(mode="json"),
            spec_hash=spec.spec_hash,
        )

    def record_run(self, run: RunRecord) -> None:
        self.emit(
            EventType.RUN_RECORDED,
            phase=Phase.RUN,
            message=f"{run.arm} seed={run.seed} {run.status}",
            run=run.model_dump(mode="json"),
        )

    def add_result(self, result: Result) -> None:
        self.emit(
            EventType.RESULT_ADDED,
            phase=Phase.ANALYZE,
            message=(
                f"{result.n_seeds} seeds, {len(result.comparisons)} comparisons, "
                f"{result.failed_runs} failed"
            ),
            result=result.model_dump(mode="json"),
        )

    def add_verdict(self, verdict: Verdict) -> None:
        self.emit(
            EventType.VERDICT_ADDED,
            phase=Phase.ANALYZE,
            message=verdict.status.value,
            verdict=verdict.model_dump(mode="json"),
        )

    def add_critique(self, critique: CritiqueReport) -> None:
        self.emit(
            EventType.CRITIQUE_ADDED,
            phase=critique.phase,
            agent="critic",
            message=critique.summary[:200],
            critique=critique.model_dump(mode="json"),
        )

    def propose_followups(self, followups: list[FollowUp]) -> None:
        self.emit(
            EventType.FOLLOWUP_PROPOSED,
            message=f"{len(followups)} proposed",
            followups=[f.model_dump(mode="json") for f in followups],
        )

    def set_report(self, report: Report) -> None:
        self.emit(
            EventType.REPORT_WRITTEN,
            phase=Phase.REPORT,
            message=report.path or report.title,
            report=report.model_dump(mode="json"),
        )

    def record_gate(self, name: str, approved: bool, *, note: str = "") -> None:
        self.emit(
            EventType.GATE_ANSWERED,
            message=note or f"{name}: {'approved' if approved else 'rejected'}",
            gate=name,
            approved=approved,
        )

    def record_truncation(self, what: str, *, kept: int, dropped: int, limit: int) -> None:
        """Bounded fan-out is only honest if the truncation is visible afterwards."""
        self.emit(
            EventType.TRUNCATED,
            message=f"{what}: kept {kept}, dropped {dropped} (limit {limit})",
            what=what,
            kept=kept,
            dropped=dropped,
            limit=limit,
        )

    def record_cost(
        self, usd: float, *, agent: str | None = None, phase: Phase | None = None
    ) -> None:
        self.emit(EventType.COST_RECORDED, agent=agent, phase=phase, usd=usd)

    def add_todo(self, todo: str) -> None:
        self.emit(EventType.TODO_ADDED, message=todo, todo=todo)

    def start_phase(self, phase: Phase) -> None:
        self.emit(EventType.PHASE_STARTED, phase=phase, message=f"entering {phase.value}")

    def complete_phase(self, phase: Phase, *, note: str = "") -> None:
        self.emit(EventType.PHASE_COMPLETED, phase=phase, message=note)

    def fail_phase(self, phase: Phase, reason: str) -> None:
        self.emit(EventType.PHASE_FAILED, phase=phase, message=reason)

    # -- the fold ------------------------------------------------------------

    def _apply(self, event: Event) -> None:  # noqa: C901 - a flat dispatch table is clearer than indirection
        snap = self.snapshot
        payload = event.payload
        snap.event_count += 1
        snap.updated_at = event.at

        match event.type:
            case EventType.PROJECT_CREATED:
                # Taken from the log, never from the clock: a replay that stamps
                # its own creation time is not a replay.
                snap.created_at = event.at
            case EventType.IDEA_SET:
                snap.idea = Idea.model_validate(payload["idea"])
            case EventType.BRIEF_SET:
                snap.brief = ResearchBrief.model_validate(payload["brief"])
            case EventType.QUESTION_ADDED:
                snap.questions.append(Question.model_validate(payload["question"]))
            case EventType.QUESTION_ANSWERED:
                for q in snap.questions:
                    if q.id == payload.get("question_id"):
                        q.status = "answered"
                        q.answer = payload.get("answer")
            case EventType.PAPERS_SURVEYED:
                for key in payload.get("paper_keys", []):
                    if key not in snap.paper_keys:
                        snap.paper_keys.append(key)
            case EventType.NOTE_ADDED:
                note_id = payload.get("note_id")
                if note_id and note_id not in snap.note_ids:
                    snap.note_ids.append(note_id)
            case EventType.REPO_GROUNDED:
                repo_id = payload.get("repo_id")
                if repo_id and repo_id not in snap.repo_ids:
                    snap.repo_ids.append(repo_id)
            case EventType.RECIPE_ADDED:
                snap.recipes.append(Recipe.model_validate(payload["recipe"]))
            case EventType.SYNTHESIS_SET:
                snap.synthesis = Synthesis.model_validate(payload["synthesis"])
            case EventType.HYPOTHESIS_ADDED:
                snap.hypotheses.append(Hypothesis.model_validate(payload["hypothesis"]))
            case EventType.HYPOTHESIS_UPDATED:
                updated = Hypothesis.model_validate(payload["hypothesis"])
                snap.hypotheses = [updated if h.id == updated.id else h for h in snap.hypotheses]
            case EventType.SPEC_ADDED:
                spec = ExperimentSpec.model_validate(payload["spec"])
                snap.specs = [s for s in snap.specs if s.id != spec.id] + [spec]
            case EventType.RUN_RECORDED:
                run = RunRecord.model_validate(payload["run"])
                snap.runs = [r for r in snap.runs if r.id != run.id] + [run]
                snap.cost_usd += run.cost_usd
            case EventType.RESULT_ADDED:
                snap.results.append(Result.model_validate(payload["result"]))
            case EventType.VERDICT_ADDED:
                snap.verdicts.append(Verdict.model_validate(payload["verdict"]))
            case EventType.CRITIQUE_ADDED:
                snap.critiques.append(CritiqueReport.model_validate(payload["critique"]))
            case EventType.FOLLOWUP_PROPOSED:
                snap.followups = [FollowUp.model_validate(f) for f in payload.get("followups", [])]
            case EventType.REPORT_WRITTEN:
                snap.report = Report.model_validate(payload["report"])
            case EventType.GATE_ANSWERED:
                snap.gate_responses[str(payload.get("gate"))] = bool(payload.get("approved"))
            case EventType.TRUNCATED:
                snap.truncations.append(payload)
            case EventType.COST_RECORDED:
                snap.cost_usd += float(payload.get("usd", 0.0))
            case EventType.TODO_ADDED:
                todo = str(payload.get("todo", ""))
                if todo and todo not in snap.todos:
                    snap.todos.append(todo)
            case EventType.PHASE_STARTED:
                if event.phase is not None:
                    snap.phase = event.phase
            case EventType.PHASE_COMPLETED:
                if event.phase is not None and event.phase not in snap.completed_phases:
                    snap.completed_phases.append(event.phase)
            case _:
                pass

    # -- reading -------------------------------------------------------------

    def events(self, *, type: EventType | None = None, phase: Phase | None = None) -> list[Event]:
        out: list[Event] = []
        for record in read_jsonl(self.log_path):
            try:
                event = Event.model_validate(record)
            except Exception:
                continue
            if type is not None and event.type is not type:
                continue
            if phase is not None and event.phase is not phase:
                continue
            out.append(event)
        return out

    def transcript(self, *, limit: int = 50) -> list[str]:
        """Human-readable decision trail, newest last."""
        return [
            f"[{e.at:%H:%M:%S}] {e.phase.value if e.phase else '-':<11} "
            f"{e.type.value:<18} {e.message}"
            for e in self.events()[-limit:]
        ]

    @property
    def cached_snapshot(self) -> ProjectSnapshot | None:
        data = read_json(self.snapshot_path)
        return ProjectSnapshot.model_validate(data) if data else None


__all__ = ["Event", "EventType", "ProjectSnapshot", "ProjectState", "Synthesis"]
