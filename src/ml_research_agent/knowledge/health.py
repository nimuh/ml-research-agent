"""KB health: orphan notes, uncited claims, stale entries, contradiction
detection, coverage gaps -> TODOs that feed the next survey round."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Literal

from ..config import Config
from ..types import Note, utcnow
from ..utils.hashing import hash_obj
from .graph import KnowledgeGraph
from .index import load_index_ids
from .notes import validate_note
from .store import KnowledgeStore

Severity = Literal["error", "warning", "info"]

# A note nobody revisited in half a year is not wrong, but the literature it
# summarizes has moved; that is a survey prompt, not a defect.
STALE_AFTER_DAYS = 180
MAX_GAP_ISSUES = 20


@dataclass
class HealthIssue:
    kind: str
    severity: Severity
    target: str
    message: str
    suggestion: str = ""


@dataclass
class HealthReport:
    issues: list[HealthIssue] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def errors(self) -> list[HealthIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[HealthIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_markdown(self) -> str:
        lines = ["# KB health", "", "## Stats", ""]
        lines += [f"- **{k}**: {v}" for k, v in sorted(self.stats.items())]
        for severity in ("error", "warning", "info"):
            group = [i for i in self.issues if i.severity == severity]
            if not group:
                continue
            lines += ["", f"## {severity.capitalize()}s ({len(group)})", ""]
            for issue in group:
                suffix = f" -- _{issue.suggestion}_" if issue.suggestion else ""
                lines.append(f"- `{issue.kind}` **{issue.target}**: {issue.message}{suffix}")
        if not self.issues:
            lines += ["", "No issues found."]
        return "\n".join(lines) + "\n"


class KnowledgeHealth:
    """Lints the KB and turns every hole into a TODO the next survey can pick up.

    Health checks exist because a KB degrades silently: a note whose citation
    does not resolve, a claim nothing backs, and a method-by-dataset cell nobody
    ever filled all look identical to a healthy KB from the inside.
    """

    def __init__(
        self, config: Config, store: KnowledgeStore, graph: KnowledgeGraph | None = None
    ) -> None:
        self.config = config
        self.store = store
        self.graph = graph or KnowledgeGraph(store)

    def check(self) -> HealthReport:
        issues: list[HealthIssue] = []
        notes = self.store.list_notes()
        documents = self.store.list_documents()
        paper_keys = {d.key for d in documents if d.kind == "paper"}

        issues += self._check_notes(notes, paper_keys)
        issues += self._check_claims()
        issues += self._check_documents()
        issues += self._check_index_drift()
        issues += self._check_contradictions()
        issues += self._check_gaps()

        stats = dict(self.store.stats())
        stats["issues"] = len(issues)
        stats["errors"] = sum(1 for i in issues if i.severity == "error")
        return HealthReport(issues=issues, stats=stats)

    # -- individual checks -------------------------------------------------

    def _check_notes(self, notes: list[Note], paper_keys: set[str]) -> list[HealthIssue]:
        issues: list[HealthIssue] = []
        cutoff = utcnow() - timedelta(days=STALE_AFTER_DAYS)
        for note in notes:
            for violation in validate_note(note, self.config.knowledge):
                issues.append(
                    HealthIssue(
                        kind="invalid_note",
                        severity="error",
                        target=note.id,
                        message=violation,
                        suggestion="re-curate this note from its passages",
                    )
                )
            if note.type == "paper" and note.paper_key and note.paper_key not in paper_keys:
                issues.append(
                    HealthIssue(
                        kind="orphan_note",
                        severity="warning",
                        target=note.id,
                        message=f"paper_key {note.paper_key} is not an ingested document",
                        suggestion="ingest the paper or correct the note's paper_key",
                    )
                )
            if note.type != "todo" and not note.sources and not note.paper_key:
                issues.append(
                    HealthIssue(
                        kind="orphan_note",
                        severity="warning",
                        target=note.id,
                        message="note is attached to no source",
                        suggestion="link the note to a paper, repo or run",
                    )
                )
            if note.added_at < cutoff and self.store.superseded_by(note.id) is None:
                issues.append(
                    HealthIssue(
                        kind="stale_note",
                        severity="info",
                        target=note.id,
                        message=f"not revisited since {note.added_at.date()}",
                        suggestion="re-check against literature published since",
                    )
                )
        return issues

    def _check_claims(self) -> list[HealthIssue]:
        issues: list[HealthIssue] = []
        for claim in self.store.list_claims():
            if not claim.evidence:
                issues.append(
                    HealthIssue(
                        kind="uncited_claim",
                        severity="error",
                        target=claim.id,
                        message="claim has no evidence; nothing in the KB backs it",
                        suggestion="attach a passage or a run id, or drop the claim",
                    )
                )
            elif not any(e.provenance.locator for e in claim.evidence):
                issues.append(
                    HealthIssue(
                        kind="uncited_claim",
                        severity="warning",
                        target=claim.id,
                        message="evidence names sources but no passage locators",
                        suggestion="narrow the evidence to a section or table",
                    )
                )
        return issues

    def _check_documents(self) -> list[HealthIssue]:
        issues: list[HealthIssue] = []
        for document in self.store.list_documents():
            if not self.store.list_passages(doc_id=document.id):
                issues.append(
                    HealthIssue(
                        kind="unchunked_document",
                        severity="warning",
                        target=document.key,
                        message="document has no passages; it cannot be retrieved or cited",
                        suggestion="re-ingest it, with parsed sections if available",
                    )
                )
        return issues

    def _check_index_drift(self) -> list[HealthIssue]:
        """Store and index must agree; either direction of drift is a bug."""
        issues: list[HealthIssue] = []
        index_dir = Path(self.config.paths.kb_index)
        if not index_dir.exists():
            return issues
        indexed = load_index_ids(index_dir)
        stored = {p.id for p in self.store.list_passages()}
        missing = stored - indexed
        extra = indexed - stored
        if missing:
            issues.append(
                HealthIssue(
                    kind="index_drift",
                    severity="warning",
                    target="index",
                    message=f"{len(missing)} stored passages are not indexed",
                    suggestion="run `mra kb reindex` (Ingestor.rebuild_indexes)",
                )
            )
        if extra:
            issues.append(
                HealthIssue(
                    kind="index_drift",
                    severity="warning",
                    target="index",
                    message=f"{len(extra)} indexed passages are missing from the store",
                    suggestion="rebuild the index from the store, which is truth",
                )
            )
        return issues

    def _check_contradictions(self) -> list[HealthIssue]:
        return [
            HealthIssue(
                kind="contradiction",
                severity="info",
                target=f"{a} vs {b}",
                message="two claims contradict each other",
                suggestion="candidate replication: design an experiment that separates them",
            )
            for a, b in self.graph.contradictions()
        ]

    def _check_gaps(self) -> list[HealthIssue]:
        gaps = self.graph.gaps(min_support=1)[:MAX_GAP_ISSUES]
        return [
            HealthIssue(
                kind="coverage_gap",
                severity="info",
                target=f"{gap['row']} x {gap['col']}",
                message=str(gap["reason"]),
                suggestion="candidate experiment, or a survey query that fills the cell",
            )
            for gap in gaps
        ]

    # -- todos -------------------------------------------------------------

    def write_todos(self, report: HealthReport) -> list[Path]:
        """Turn errors and warnings into TODO notes in the wiki.

        Ids are derived from ``(kind, target)`` so re-running health updates the
        same TODO instead of breeding a new one on every check.
        """
        paths: list[Path] = []
        for issue in report.issues:
            if issue.severity == "info":
                continue
            note = self.todo_note(issue)
            paths.append(self.store.save_note(note))
        return paths

    @staticmethod
    def todo_note(issue: HealthIssue) -> Note:
        return Note(
            id=f"note_todo_{hash_obj((issue.kind, issue.target), length=12)}",
            type="todo",
            title=f"TODO: {issue.kind} -- {issue.target}",
            summary=f"{issue.message}\n\nSuggested fix: {issue.suggestion or 'unspecified'}",
            confidence=0.0,
            sources=[issue.target],
            tags=["todo", issue.kind, issue.severity],
            added_at=utcnow(),
        )


__all__ = ["MAX_GAP_ISSUES", "STALE_AFTER_DAYS", "HealthIssue", "HealthReport", "KnowledgeHealth"]
