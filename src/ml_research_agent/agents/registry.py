"""Name -> agent class registry so configs and the router can reference agents
by string, and users can register custom agents."""

from __future__ import annotations

from typing import Any

from ..errors import ConfigError
from .analyst import Analyst
from .base import BaseAgent
from .code_analyst import CodeAnalyst
from .critic import Critic
from .curator import Curator, Screener
from .experiment_planner import ExperimentPlanner
from .framer import Framer
from .implementer import Implementer
from .runner_agent import RunnerAgent
from .scout import Scout
from .synthesizer import Synthesizer
from .writer import Writer

_REGISTRY: dict[str, type[BaseAgent[Any, Any]]] = {}


def register(
    agent_cls: type[BaseAgent[Any, Any]], *, name: str | None = None
) -> type[BaseAgent[Any, Any]]:
    """Register an agent class under its ``name`` (or an explicit override)."""
    key = name or agent_cls.name
    if not key:
        raise ConfigError("agent class has no name", cls=agent_cls.__name__)
    _REGISTRY[key] = agent_cls
    return agent_cls


def get_agent_class(name: str) -> type[BaseAgent[Any, Any]]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ConfigError("unknown agent", agent=name, known=sorted(_REGISTRY)) from None


def create(name: str, **kwargs: Any) -> BaseAgent[Any, Any]:
    return get_agent_class(name)(**kwargs)


def registered() -> list[str]:
    return sorted(_REGISTRY)


for _cls in (
    Framer,
    Scout,
    Screener,
    Curator,
    CodeAnalyst,
    Synthesizer,
    ExperimentPlanner,
    Implementer,
    RunnerAgent,
    Analyst,
    Critic,
    Writer,
):
    register(_cls)


__all__ = ["create", "get_agent_class", "register", "registered"]
