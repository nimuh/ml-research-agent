"""Writer: assembles the final report -- idea, related work, method, setup,
results, threats to validity, next experiments -- citing only KB-backed claims."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import (
    Citation,
    ModelTier,
    Report,
    ReportSection,
    ResearchBrief,
    Result,
    Verdict,
)
from .base import AgentContext, BaseAgent

REQUIRED_SECTIONS = (
    "Idea and framing",
    "What the literature says",
    "Method",
    "Experimental setup",
    "Results",
    "Threats to validity",
    "What we would do next",
)


class WriterInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: ResearchBrief
    synthesis: dict[str, Any] = Field(default_factory=dict)
    results: list[Result] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    available_citations: list[Citation] = Field(
        min_length=1, description="The ONLY citations the report may use."
    )
    followups: list[str] = Field(default_factory=list)
    budget_note: str = ""


class ReportDraft(BaseModel):
    """Prose the evidence supports. Nothing else is allowed through."""

    model_config = ConfigDict(extra="forbid")

    title: str
    abstract: str = Field(description="What was asked, what was done, what the evidence supports.")
    sections: list[ReportSection] = Field(min_length=1)
    headline_finding: str = Field(
        description="One sentence. State a negative result as plainly as a positive one."
    )


class Writer(BaseAgent[WriterInput, ReportDraft]):
    """Exists to prevent confident prose the evidence doesn't support.

    Restricted to the citation set it is handed: a claim whose citation key is
    not in ``available_citations`` is a fabrication, and it is caught here
    rather than by a reader.
    """

    name: ClassVar[str] = "writer"
    description: ClassVar[str] = "Assembles the research memo from state + KB + results."
    tier: ClassVar[ModelTier] = ModelTier.DEEP
    prompt_name: ClassVar[str] = "writer"
    tool_allowlist: ClassVar[tuple[str, ...]] = ("kb_search",)
    output_model: ClassVar[type[BaseModel]] = ReportDraft

    def prompt_variables(self, payload: WriterInput, ctx: AgentContext) -> dict[str, object]:
        verdicts = (
            "\n".join(
                f"- [{v.status.value.upper()}] {v.reasoning[:400]}\n"
                f"  rule: {v.decision_rule_statement}\n"
                f"  seeds: {v.seed_variance_note}\n"
                f"  threats: {'; '.join(v.threats_to_validity)}"
                for v in payload.verdicts
            )
            or "- no experiments reached a verdict"
        )
        results = (
            "\n".join(
                f"- {r.spec_id}: {len(r.run_ids)} runs, {r.n_seeds} seeds, {r.failed_runs} failed, "
                f"${r.total_cost_usd:.2f}\n"
                + "\n".join(
                    f"    {c.metric}: {c.treatment_arm} vs {c.baseline_arm} "
                    f"= {c.effect:+.5g} (p={c.p_value})"
                    for c in r.comparisons
                )
                for r in payload.results
            )
            or "- none"
        )
        return {
            "title": payload.brief.title,
            "problem_statement": payload.brief.problem_statement,
            "claims": "\n".join(
                f"- {c.statement} (refuted by: {c.falsifier})" for c in payload.brief.claims
            ),
            "scope_limits": "\n".join(f"- {s}" for s in payload.brief.scope_limits)
            or "- none stated",
            "synthesis": str(payload.synthesis)[:6000],
            "results": results,
            "verdicts": verdicts,
            "citations": "\n".join(
                f"- [{c.key}] {c.label} ({c.kind})" for c in payload.available_citations[:200]
            ),
            "followups": "\n".join(f"- {f}" for f in payload.followups) or "- none proposed",
            "required_sections": "\n".join(f"- {s}" for s in REQUIRED_SECTIONS),
            "budget_note": payload.budget_note or "not reported",
        }

    def validate_output(
        self, output: ReportDraft, payload: WriterInput, ctx: AgentContext
    ) -> list[str]:
        """Every claim cites a KB source or a run id -- checked, not trusted."""
        allowed = {c.key for c in payload.available_citations}
        violations: list[str] = []
        for section in output.sections:
            for cite in section.citations:
                if cite.key not in allowed:
                    violations.append(
                        f"section '{section.title}' cites '{cite.key}', which is not in the KB "
                        "or run set -- fabricated citations are not permitted"
                    )
        titles = {s.title.lower() for s in output.sections}
        missing = [
            required
            for required in ("results", "threats to validity")
            if not any(required in t for t in titles)
        ]
        if missing:
            violations.append(f"missing required section(s): {missing}")
        evidence_sections = [
            s
            for s in output.sections
            if any(k in s.title.lower() for k in ("result", "literature"))
        ]
        for section in evidence_sections:
            if not section.citations:
                violations.append(f"section '{section.title}' asserts findings with no citations")
        if not output.headline_finding.strip():
            violations.append("no headline finding stated")
        return violations

    def to_report(
        self, draft: ReportDraft, payload: WriterInput, *, path: str | None = None
    ) -> Report:
        used = {c.key for s in draft.sections for c in s.citations}
        return Report(
            brief_id=payload.brief.id,
            title=draft.title,
            abstract=draft.abstract,
            sections=sorted(draft.sections, key=lambda s: s.order),
            citations=[c for c in payload.available_citations if c.key in used],
            path=path,
        )


__all__ = ["REQUIRED_SECTIONS", "ReportDraft", "Writer", "WriterInput"]
