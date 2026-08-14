"""Scout (literature search): expands queries, fans out across sources, snowballs
citations, and returns ranked candidate papers with relevance rationales."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..config import Config
from ..types import ModelTier, RankedPaper, ResearchBrief
from .base import AgentContext, BaseAgent


class ScoutInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief: ResearchBrief
    known_keys: list[str] = Field(
        default_factory=list, description="Already surveyed; do not re-suggest."
    )


class QueryExpansion(BaseModel):
    """The one thing an LLM does better than the deterministic query planner:

    naming the vocabulary a field actually uses, including the terms a
    newcomer would never guess and the adjacent literatures worth a look.
    """

    model_config = ConfigDict(extra="forbid")

    search_terms: list[str] = Field(
        min_length=3, description="Query strings a domain expert would run."
    )
    seminal_works: list[str] = Field(
        default_factory=list, description="Papers that MUST appear or the survey is incomplete."
    )
    adjacent_areas: list[str] = Field(default_factory=list)
    key_authors: list[str] = Field(default_factory=list)
    venues: list[str] = Field(default_factory=list)
    rationale: str = ""


class SurveyReport(BaseModel):
    """What SURVEY hands to CURATE."""

    model_config = ConfigDict(extra="forbid")

    ranked: list[RankedPaper] = Field(default_factory=list)
    expansion: QueryExpansion | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    missing_seminal: list[str] = Field(
        default_factory=list, description="Named seminal works the fan-out failed to surface."
    )

    @property
    def paper_keys(self) -> list[str]:
        return [r.paper.key for r in self.ranked]


class Scout(BaseAgent[ScoutInput, QueryExpansion]):
    """Exists to prevent missing the obvious prior work.

    The LLM contributes vocabulary and a checklist of works that must show up;
    the retrieval itself stays deterministic and LLM-free so recall is tunable
    independently of anyone's judgment.
    """

    name: ClassVar[str] = "scout"
    description: ClassVar[str] = "Plans and executes the literature fan-out."
    tier: ClassVar[ModelTier] = ModelTier.STANDARD
    prompt_name: ClassVar[str] = "scout"
    output_model: ClassVar[type[BaseModel]] = QueryExpansion

    def prompt_variables(self, payload: ScoutInput, ctx: AgentContext) -> dict[str, object]:
        return {
            "title": payload.brief.title,
            "problem_statement": payload.brief.problem_statement,
            "claims": "\n".join(f"- {c.statement}" for c in payload.brief.claims),
            "existing_terms": ", ".join(payload.brief.search_terms) or "none",
            "related_areas": ", ".join(payload.brief.related_areas) or "none",
        }

    def survey(self, payload: ScoutInput, ctx: AgentContext, config: Config) -> SurveyReport:
        """Expand, fan out, dedupe, rank, snowball -- and report what was missed.

        Imported lazily so `agents` stays usable (and testable) without the
        literature layer's optional dependencies installed.
        """
        from ..literature import search_literature

        expansion = self.run(payload, ctx)
        enriched = payload.brief.model_copy(
            update={
                "search_terms": _dedupe_preserving_order(
                    [*payload.brief.search_terms, *expansion.search_terms]
                ),
                "related_areas": _dedupe_preserving_order(
                    [*payload.brief.related_areas, *expansion.adjacent_areas]
                ),
            }
        )
        ranked, stats = search_literature(enriched, config, logger=ctx.logger)

        known = {k.lower() for k in payload.known_keys}
        ranked = [r for r in ranked if r.paper.key.lower() not in known]

        return SurveyReport(
            ranked=ranked,
            expansion=expansion,
            stats=stats,
            missing_seminal=_missing_titles(expansion.seminal_works, ranked),
        )


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _missing_titles(expected: list[str], ranked: list[RankedPaper]) -> list[str]:
    """Cheap containment check -- the SURVEY exit criterion, made checkable.

    A seminal work the fan-out never surfaced is a recall failure worth
    surfacing at the gate, not a detail to discover three phases later.
    """
    found = " | ".join(r.paper.title.lower() for r in ranked)
    return [title for title in expected if _significant(title) not in found]


def _significant(title: str) -> str:
    return " ".join(w for w in title.lower().split() if len(w) > 3)[:60]


__all__ = ["QueryExpansion", "Scout", "ScoutInput", "SurveyReport"]
