"""Orchestration layer: the state machine that drives the whole research loop.

Owns phase sequencing, agent dispatch, checkpointing, budget enforcement,
and human-in-the-loop gates. Agents never call each other directly; they are
invoked by the director and communicate through the shared ProjectState.
"""

from .checkpoint import Checkpoint, CheckpointStore
from .director import PhaseOutcome, ResearchDirector
from .gates import DecisionPacket, GateKeeper, auto_approve_responder, deny_responder
from .plan import LadderRung, ResearchPlan, build_plan, order_specs, replan
from .router import Route, Router
from .state import Event, EventType, ProjectSnapshot, ProjectState, Synthesis

__all__ = [
    "Checkpoint",
    "CheckpointStore",
    "DecisionPacket",
    "Event",
    "EventType",
    "GateKeeper",
    "LadderRung",
    "PhaseOutcome",
    "ProjectSnapshot",
    "ProjectState",
    "ResearchDirector",
    "ResearchPlan",
    "Route",
    "Router",
    "Synthesis",
    "auto_approve_responder",
    "build_plan",
    "deny_responder",
    "order_specs",
    "replan",
]
