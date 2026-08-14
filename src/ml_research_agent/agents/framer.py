"""Framer: turns a one-line idea into a ResearchBrief -- problem statement,
assumptions, success criteria, falsifiable claims, scope limits, search terms."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import (
    FalsifiableClaim,
    Idea,
    ModelTier,
    Provenance,
    ResearchBrief,
    SuccessCriterion,
)
from .base import AgentContext, BaseAgent


class FramerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idea: Idea
    context: str = ""


class BriefDraft(BaseModel):
    """What the model is allowed to author.

    Ids, timestamps and provenance are assigned by code, never generated -- an
    LLM-invented id is untraceable by definition.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Short, specific working title.")
    problem_statement: str = Field(description="What is unknown, stated precisely.")
    motivation: str = Field(default="", description="Why the answer would matter.")
    claims: list[FalsifiableClaim] = Field(
        min_length=1, description="Each with the observation that would refute it."
    )
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    scope_limits: list[str] = Field(
        default_factory=list, description="What this explicitly does NOT test."
    )
    search_terms: list[str] = Field(default_factory=list)
    related_areas: list[str] = Field(default_factory=list)


class Framer(BaseAgent[FramerInput, BriefDraft]):
    """Exists to prevent researching a vague idea nobody could ever falsify."""

    name: ClassVar[str] = "framer"
    description: ClassVar[str] = "Frames a raw idea into a falsifiable research brief."
    tier: ClassVar[ModelTier] = ModelTier.DEEP
    prompt_name: ClassVar[str] = "framer"
    output_model: ClassVar[type[BaseModel]] = BriefDraft

    def prompt_variables(self, payload: FramerInput, ctx: AgentContext) -> dict[str, object]:
        return {
            "idea_text": payload.idea.text,
            "domain": payload.idea.domain or "machine learning",
            "constraints": "\n".join(f"- {c}" for c in payload.idea.constraints) or "- none stated",
            "context": payload.context or "none",
        }

    def validate_output(
        self, output: BriefDraft, payload: FramerInput, ctx: AgentContext
    ) -> list[str]:
        """The phase exit criterion, enforced in code rather than trusted."""
        violations: list[str] = []
        for i, claim in enumerate(output.claims):
            if not claim.falsifier.strip():
                violations.append(f"claim {i + 1} has no falsifier: '{claim.statement}'")
            if claim.falsifier.strip().lower() == claim.statement.strip().lower():
                violations.append(f"claim {i + 1}'s falsifier merely restates the claim")
        if not output.success_criteria:
            violations.append("no measurable success criterion was given")
        if not output.scope_limits:
            violations.append("no scope limits stated; an unbounded brief cannot be closed out")
        if len(output.search_terms) < 3:
            violations.append("fewer than 3 search terms; SURVEY needs query material")
        return violations

    def to_brief(self, draft: BriefDraft, idea: Idea, *, model: str = "") -> ResearchBrief:
        """Assemble the domain object, stamping provenance the model cannot forge."""
        return ResearchBrief(
            idea_id=idea.id,
            title=draft.title,
            problem_statement=draft.problem_statement,
            motivation=draft.motivation,
            claims=draft.claims,
            success_criteria=draft.success_criteria,
            assumptions=draft.assumptions,
            scope_limits=draft.scope_limits,
            search_terms=draft.search_terms,
            related_areas=draft.related_areas,
            provenance=[
                Provenance(source=f"agent:{self.name}@v1", locator=model or None, quote=idea.text)
            ],
        )


__all__ = ["BriefDraft", "Framer", "FramerInput"]
