"""The agent team. Each agent is a narrow role with an explicit typed contract:
`run(input_model, context) -> output_model`. Agents are stateless; all memory
lives in ProjectState and the knowledge base."""

from .analyst import Analyst, AnalystInput, VerdictDraft
from .base import AgentContext, BaseAgent, RetrieverLike
from .code_analyst import CodeAnalyst, CodeAnalystInput, RecipeDraft
from .critic import Critic, CritiqueDraft, CritiqueInput, gate_passed
from .curator import (
    Curator,
    NoteDraft,
    NoteInput,
    Screener,
    ScreeningBatch,
    ScreeningInput,
    included,
)
from .experiment_planner import ExperimentPlanner, PlannerInput, SpecDraft
from .framer import BriefDraft, Framer, FramerInput
from .implementer import ImplementationDraft, Implementer, ImplementerInput
from .registry import create, get_agent_class, register, registered
from .runner_agent import RunnerAgent, TriageDecision, TriageInput
from .scout import QueryExpansion, Scout, ScoutInput, SurveyReport
from .synthesizer import SynthesisDraft, SynthesisInput, Synthesizer
from .writer import ReportDraft, Writer, WriterInput

__all__ = [
    "AgentContext",
    "Analyst",
    "AnalystInput",
    "BaseAgent",
    "BriefDraft",
    "CodeAnalyst",
    "CodeAnalystInput",
    "Critic",
    "CritiqueDraft",
    "CritiqueInput",
    "Curator",
    "ExperimentPlanner",
    "Framer",
    "FramerInput",
    "ImplementationDraft",
    "Implementer",
    "ImplementerInput",
    "NoteDraft",
    "NoteInput",
    "PlannerInput",
    "QueryExpansion",
    "RecipeDraft",
    "ReportDraft",
    "RetrieverLike",
    "RunnerAgent",
    "Scout",
    "ScoutInput",
    "ScreeningBatch",
    "ScreeningInput",
    "Screener",
    "SpecDraft",
    "SurveyReport",
    "SynthesisDraft",
    "SynthesisInput",
    "Synthesizer",
    "TriageDecision",
    "TriageInput",
    "VerdictDraft",
    "Writer",
    "WriterInput",
    "create",
    "gate_passed",
    "get_agent_class",
    "included",
    "register",
    "registered",
]
