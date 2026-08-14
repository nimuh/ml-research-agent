"""Curator: screens candidates against inclusion criteria, extracts structured
notes (task, method, datasets, baselines, metrics, compute, limitations),
and writes them into the knowledge base with provenance."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import (
    MetricReport,
    ModelTier,
    Note,
    Paper,
    Passage,
    Provenance,
    ResearchBrief,
    ScreeningDecision,
    ScreeningDecisionKind,
)
from .base import AgentContext, BaseAgent

# ---------------------------------------------------------------------------
# Screening -- the cheap pass
# ---------------------------------------------------------------------------


class ScreeningInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: ResearchBrief
    candidates: list[Paper] = Field(min_length=1)


class ScreeningBatch(BaseModel):
    """Include/exclude with a recorded reason either way.

    The exclusion reasons matter as much as the inclusions: they are what makes
    the KB's boundary auditable instead of arbitrary.
    """

    model_config = ConfigDict(extra="forbid")

    decisions: list[ScreeningDecision] = Field(min_length=1)


class Screener(BaseAgent[ScreeningInput, ScreeningBatch]):
    """Fast-tier triage. Retrieval handed us candidates; relevance starts here."""

    name: ClassVar[str] = "screener"
    description: ClassVar[str] = "Screens candidate papers against the brief's inclusion criteria."
    tier: ClassVar[ModelTier] = ModelTier.FAST
    prompt_name: ClassVar[str] = "screener"
    output_model: ClassVar[type[BaseModel]] = ScreeningBatch

    def prompt_variables(self, payload: ScreeningInput, ctx: AgentContext) -> dict[str, object]:
        listing = "\n\n".join(
            f"[{p.key}] {p.title} ({p.year or 'n.d.'}, {p.citation_count} citations)\n"
            f"{(p.tldr or p.abstract or 'no abstract')[:700]}"
            for p in payload.candidates
        )
        return {
            "brief_title": payload.brief.title,
            "problem_statement": payload.brief.problem_statement,
            "claims": "\n".join(f"- {c.statement}" for c in payload.brief.claims),
            "scope_limits": "\n".join(f"- {s}" for s in payload.brief.scope_limits) or "- none",
            "candidates": listing,
            "count": len(payload.candidates),
        }

    def validate_output(
        self, output: ScreeningBatch, payload: ScreeningInput, ctx: AgentContext
    ) -> list[str]:
        keys = {p.key for p in payload.candidates}
        seen = {d.paper_key for d in output.decisions}
        violations: list[str] = []
        invented = seen - keys
        if invented:
            violations.append(f"decisions reference unknown papers: {sorted(invented)[:5]}")
        missing = keys - seen
        if missing:
            violations.append(
                f"{len(missing)} candidates received no decision: {sorted(missing)[:5]}"
            )
        unreasoned = [d.paper_key for d in output.decisions if not d.reason.strip()]
        if unreasoned:
            violations.append(f"decisions without a recorded reason: {unreasoned[:5]}")
        return violations


# ---------------------------------------------------------------------------
# Note extraction -- the expensive pass
# ---------------------------------------------------------------------------


class NoteInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: ResearchBrief
    paper: Paper
    passages: list[Passage] = Field(default_factory=list)


class NoteDraft(BaseModel):
    """A structured reading. Every field that asserts something needs a locator."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(description="What the paper did and found, in 3-5 sentences.")
    task: str | None = None
    method: str | None = None
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    metrics: list[MetricReport] = Field(
        default_factory=list, description="Headline numbers, each with the passage that stated it."
    )
    compute: str | None = Field(default=None, description="Hardware and scale actually used.")
    limitations: list[str] = Field(default_factory=list)
    relevance_to_brief: str = Field(
        description="Why this matters for the brief -- or why it barely does."
    )
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    citations: list[Provenance] = Field(
        min_length=1, description="Passage locators backing the fields above."
    )


class Curator(BaseAgent[NoteInput, NoteDraft]):
    """Exists to prevent a KB full of abstracts nobody extracted structure from."""

    name: ClassVar[str] = "curator"
    description: ClassVar[str] = "Extracts a structured, provenance-backed note from one paper."
    tier: ClassVar[ModelTier] = ModelTier.STANDARD
    prompt_name: ClassVar[str] = "curator"
    output_model: ClassVar[type[BaseModel]] = NoteDraft

    def prompt_variables(self, payload: NoteInput, ctx: AgentContext) -> dict[str, object]:
        passages = "\n\n".join(
            f"<<{p.id} | {p.locator}>>\n{p.text[:2500]}" for p in payload.passages[:40]
        ) or (payload.paper.abstract or "no full text available; abstract only")
        return {
            "brief_title": payload.brief.title,
            "problem_statement": payload.brief.problem_statement,
            "paper_title": payload.paper.title,
            "paper_key": payload.paper.key,
            "authors": ", ".join(a.name for a in payload.paper.authors[:8]) or "unknown",
            "year": payload.paper.year or "n.d.",
            "abstract": payload.paper.abstract[:2000],
            "passages": passages,
        }

    def validate_output(
        self, output: NoteDraft, payload: NoteInput, ctx: AgentContext
    ) -> list[str]:
        """The traceability invariant, enforced where it is cheapest to catch."""
        violations: list[str] = []
        known = {p.id for p in payload.passages} | {payload.paper.key}
        if not output.citations:
            violations.append("note has no passage-level provenance and is therefore inadmissible")
        for cite in output.citations:
            if payload.passages and cite.source not in known:
                violations.append(f"citation references an unknown passage: {cite.source}")
        for metric in output.metrics:
            if payload.passages and metric.provenance.source not in known:
                violations.append(f"metric '{metric.name}' cites an unknown passage")
        if not output.relevance_to_brief.strip():
            violations.append("relevance_to_brief is empty; the note cannot be triaged later")
        return violations

    def to_note(self, draft: NoteDraft, payload: NoteInput) -> Note:
        """Assemble the KB object with ids and provenance stamped by code."""
        return Note(
            type="paper",
            paper_key=payload.paper.key,
            title=payload.paper.title,
            summary=draft.summary,
            task=draft.task,
            method=draft.method,
            datasets=draft.datasets,
            baselines=draft.baselines,
            metrics=draft.metrics,
            compute=draft.compute,
            limitations=draft.limitations,
            relevance_to_brief=draft.relevance_to_brief,
            confidence=draft.confidence,
            sources=[payload.paper.key, *([payload.paper.url] if payload.paper.url else [])],
            provenance=draft.citations,
        )


def included(decisions: list[ScreeningDecision], *, keep_maybe: bool = False) -> list[str]:
    """Paper keys that survived screening, in decision order."""
    kinds = {ScreeningDecisionKind.INCLUDE}
    if keep_maybe:
        kinds.add(ScreeningDecisionKind.MAYBE)
    return [d.paper_key for d in decisions if d.decision in kinds]


__all__ = [
    "Curator",
    "NoteDraft",
    "NoteInput",
    "ScreeningBatch",
    "ScreeningInput",
    "Screener",
    "included",
]
