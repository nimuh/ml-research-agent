"""Agent dispatch: maps a task to an agent + model tier + tool subset + budget.
Encodes the cost policy (small model for extraction/classification, large model
for design/critique) and concurrency limits per phase."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..errors import ConfigError
from ..types import ModelTier, Phase


@dataclass(frozen=True)
class Route:
    """The complete dispatch decision for one task."""

    agent: str
    tier: ModelTier
    tools: tuple[str, ...] = ()
    max_concurrency: int = 4
    budget_usd: float = 1.0
    phase: Phase | None = None


# The cost policy, in one table. `fast` screens and extracts, `standard` writes
# notes and code, `deep` frames, designs, and critiques -- the three places
# where being wrong is expensive enough to pay for the better model.
ROUTES: dict[str, Route] = {
    "frame": Route("framer", ModelTier.DEEP, phase=Phase.FRAME, budget_usd=1.0, max_concurrency=1),
    "expand_queries": Route(
        "scout", ModelTier.STANDARD, phase=Phase.SURVEY, budget_usd=0.5, max_concurrency=1
    ),
    "screen": Route(
        "screener", ModelTier.FAST, phase=Phase.CURATE, budget_usd=2.0, max_concurrency=8
    ),
    "extract_note": Route(
        "curator", ModelTier.STANDARD, phase=Phase.CURATE, budget_usd=6.0, max_concurrency=6
    ),
    "analyze_repo": Route(
        "code_analyst",
        ModelTier.STANDARD,
        tools=("read_file", "list_dir"),
        phase=Phase.GROUND,
        budget_usd=4.0,
        max_concurrency=3,
    ),
    "synthesize": Route(
        "synthesizer",
        ModelTier.DEEP,
        tools=("kb_search",),
        phase=Phase.SYNTHESIZE,
        budget_usd=4.0,
        max_concurrency=1,
    ),
    "design": Route(
        "experiment_planner", ModelTier.DEEP, phase=Phase.DESIGN, budget_usd=4.0, max_concurrency=2
    ),
    "implement": Route(
        "implementer",
        ModelTier.STANDARD,
        tools=("read_file", "list_dir", "write_file"),
        phase=Phase.IMPLEMENT,
        budget_usd=8.0,
        max_concurrency=2,
    ),
    "triage_run": Route(
        "runner",
        ModelTier.STANDARD,
        tools=("read_file",),
        phase=Phase.RUN,
        budget_usd=2.0,
        max_concurrency=2,
    ),
    "analyze": Route(
        "analyst", ModelTier.DEEP, phase=Phase.ANALYZE, budget_usd=3.0, max_concurrency=1
    ),
    "critique": Route("critic", ModelTier.DEEP, budget_usd=6.0, max_concurrency=2),
    "report": Route(
        "writer",
        ModelTier.DEEP,
        tools=("kb_search",),
        phase=Phase.REPORT,
        budget_usd=5.0,
        max_concurrency=1,
    ),
}

# The Critic runs at four gates. Naming them here keeps the director from
# deciding when adversarial review is convenient.
CRITIQUE_POINTS: tuple[Phase, ...] = (Phase.CURATE, Phase.SYNTHESIZE, Phase.DESIGN, Phase.ANALYZE)


@dataclass
class Router:
    """Task -> (agent, tier, tools, budget, concurrency).

    The single place a task is mapped to a model tier. Anywhere else choosing a
    model would be a cost policy hiding in an agent.
    """

    config: Config
    overrides: dict[str, Route] = field(default_factory=dict)

    def route(self, task: str) -> Route:
        route = self.overrides.get(task) or ROUTES.get(task)
        if route is None:
            raise ConfigError("no route for task", task=task, known=sorted(ROUTES))
        return route

    def model_for(self, task: str) -> str:
        return self.config.model_for(self.route(task).tier)

    def concurrency_for(self, task: str) -> int:
        """Never exceed the global LLM concurrency ceiling, whatever the route asks for."""
        return min(self.route(task).max_concurrency, self.config.llm.max_concurrency)

    def budget_for(self, task: str) -> float:
        """Per-task ceiling, clamped to what remains of the project budget."""
        return min(self.route(task).budget_usd, self.config.budget.usd_per_project)

    def tools_for(self, task: str) -> tuple[str, ...]:
        return self.route(task).tools

    def critiques_after(self, phase: Phase) -> bool:
        return phase in CRITIQUE_POINTS


__all__ = ["CRITIQUE_POINTS", "ROUTES", "Route", "Router"]
