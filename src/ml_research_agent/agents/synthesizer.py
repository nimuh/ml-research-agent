"""Synthesizer: cross-paper synthesis -- what is settled, what is contested,
what the standard baselines/benchmarks are, and where the idea's gap sits."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import ModelTier, Note, Provenance, ResearchBrief
from .base import AgentContext, BaseAgent


class SynthesisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: ResearchBrief
    notes: list[Note] = Field(min_length=1)
    contradictions: list[tuple[str, str]] = Field(default_factory=list)
    coverage_gaps: list[dict[str, Any]] = Field(default_factory=list)


class SupportedStatement(BaseModel):
    """Nothing enters the synthesis without the notes that back it."""

    model_config = ConfigDict(extra="forbid")

    statement: str
    support: list[Provenance] = Field(min_length=1)


class SynthesisDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settled: list[SupportedStatement] = Field(default_factory=list)
    contested: list[SupportedStatement] = Field(
        default_factory=list,
        description="Where the literature disagrees -- candidate replications.",
    )
    gaps: list[SupportedStatement] = Field(
        default_factory=list, description="Holes the evidence shows, not holes you did not search."
    )
    baselines: list[str] = Field(default_factory=list, description="Standard methods to beat.")
    benchmarks: list[str] = Field(default_factory=list, description="Standard datasets/metrics.")
    novelty_assessment: str = Field(
        description="Is the brief's idea already settled, contested, or genuinely open? Be blunt."
    )
    positioning: str = Field(default="", description="Where this idea sits relative to the field.")
    novelty_survives: bool = Field(
        description="False if the literature already answers the brief's question."
    )


class Synthesizer(BaseAgent[SynthesisInput, SynthesisDraft]):
    """Exists to prevent testing something already settled, or reinventing a baseline."""

    name: ClassVar[str] = "synthesizer"
    description: ClassVar[str] = "Builds the settled / contested / gap map from the KB."
    tier: ClassVar[ModelTier] = ModelTier.DEEP
    prompt_name: ClassVar[str] = "synthesizer"
    tool_allowlist: ClassVar[tuple[str, ...]] = ("kb_search",)
    output_model: ClassVar[type[BaseModel]] = SynthesisDraft

    def prompt_variables(self, payload: SynthesisInput, ctx: AgentContext) -> dict[str, object]:
        notes = "\n\n".join(
            f"<<{n.id} | {n.paper_key}>> {n.title}\n"
            f"  method: {n.method or '-'} | datasets: {', '.join(n.datasets) or '-'}\n"
            f"  baselines: {', '.join(n.baselines) or '-'}\n"
            f"  metrics: {'; '.join(f'{m.name}={m.value}' for m in n.metrics) or '-'}\n"
            f"  limitations: {'; '.join(n.limitations) or '-'}\n"
            f"  {n.summary[:600]}"
            for n in payload.notes[:60]
        )
        return {
            "title": payload.brief.title,
            "problem_statement": payload.brief.problem_statement,
            "claims": "\n".join(
                f"- {c.statement} (refuted by: {c.falsifier})" for c in payload.brief.claims
            ),
            "notes": notes,
            "note_count": len(payload.notes),
            "contradictions": "\n".join(f"- {a} vs {b}" for a, b in payload.contradictions)
            or "- none detected",
            "coverage_gaps": "\n".join(f"- {g}" for g in payload.coverage_gaps[:20])
            or "- none computed",
        }

    def validate_output(
        self, output: SynthesisDraft, payload: SynthesisInput, ctx: AgentContext
    ) -> list[str]:
        known = {n.id for n in payload.notes} | {n.paper_key for n in payload.notes if n.paper_key}
        violations: list[str] = []
        for bucket_name, bucket in (
            ("settled", output.settled),
            ("contested", output.contested),
            ("gaps", output.gaps),
        ):
            for item in bucket:
                unknown = [p.source for p in item.support if p.source not in known]
                if unknown:
                    violations.append(
                        f"{bucket_name} statement cites sources absent from the KB: {unknown[:3]}"
                    )
        if not output.novelty_assessment.strip():
            violations.append("no novelty assessment; DESIGN cannot proceed without one")
        if not output.baselines:
            violations.append(
                "no baselines identified; an experiment with nothing to beat is not a test"
            )
        return violations


__all__ = ["SupportedStatement", "SynthesisDraft", "SynthesisInput", "Synthesizer"]
