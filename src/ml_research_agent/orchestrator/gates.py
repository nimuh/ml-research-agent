"""Human-in-the-loop gates: pause points (post-survey, pre-run, pre-report) that
surface a decision packet and block until approved, auto-approved, or timed out."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..config import Config
from ..errors import GateRejected
from ..types import Phase

GateName = Literal["after_survey", "before_run", "before_report"]


@dataclass
class DecisionPacket:
    """What a human needs in order to answer in under a minute.

    A gate that requires reading the whole event log is a gate people rubber
    stamp, which is worse than no gate at all.
    """

    name: str
    phase: Phase
    question: str
    summary: str
    facts: dict[str, Any] = field(default_factory=dict)
    cost_so_far_usd: float = 0.0
    estimated_next_usd: float = 0.0
    warnings: list[str] = field(default_factory=list)
    options: tuple[str, ...] = ("approve", "reject")

    def render(self) -> str:
        lines = [
            f"=== GATE: {self.name} ({self.phase.value}) ===",
            "",
            self.question,
            "",
            self.summary.strip(),
            "",
        ]
        if self.facts:
            lines.append("Facts:")
            lines += [f"  {k}: {v}" for k, v in self.facts.items()]
            lines.append("")
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  ! {w}" for w in self.warnings]
            lines.append("")
        lines.append(f"Spent so far: ${self.cost_so_far_usd:.2f}")
        if self.estimated_next_usd:
            lines.append(f"Estimated for the next phase: ${self.estimated_next_usd:.2f}")
        return "\n".join(lines)


class GateResponder(Protocol):
    """How an approval is obtained. Swappable so CI and tests never block."""

    def __call__(self, packet: DecisionPacket) -> bool: ...


def console_responder(packet: DecisionPacket) -> bool:
    """Interactive prompt. Anything other than an explicit yes is a rejection."""
    print(packet.render(), file=sys.stderr)
    try:
        answer = input("Approve? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def auto_approve_responder(packet: DecisionPacket) -> bool:
    return True


def deny_responder(packet: DecisionPacket) -> bool:
    return False


@dataclass
class GateKeeper:
    """Evaluates whether a gate is enabled, then asks.

    Gates are configured, not hardcoded: full autonomy is a config flip, but the
    first real projects should keep post-SURVEY and pre-RUN on.
    """

    config: Config
    responder: GateResponder | None = None

    def enabled(self, name: GateName) -> bool:
        return bool(getattr(self.config.gates, name))

    def ask(self, name: GateName, packet: DecisionPacket) -> bool:
        if not self.enabled(name):
            return True
        if self.config.gates.auto_approve:
            return True
        responder = self.responder or console_responder
        return responder(packet)

    def require(self, name: GateName, packet: DecisionPacket) -> None:
        """Ask, and raise if refused. Used where proceeding unapproved is unsafe."""
        if not self.ask(name, packet):
            raise GateRejected("gate rejected by operator", gate=name, phase=packet.phase.value)


def survey_packet(
    *,
    paper_count: int,
    kb_count: int,
    missing_seminal: list[str],
    cost_usd: float,
    top_titles: list[str],
) -> DecisionPacket:
    return DecisionPacket(
        name="after_survey",
        phase=Phase.SURVEY,
        question="Is this the right literature to build the knowledge base from?",
        summary="\n".join(f"  {i + 1}. {t}" for i, t in enumerate(top_titles[:15])),
        facts={"candidates": paper_count, "will_ingest": kb_count},
        cost_so_far_usd=cost_usd,
        warnings=(
            [f"named seminal work not surfaced: {t}" for t in missing_seminal]
            if missing_seminal
            else []
        ),
    )


def run_packet(
    *, specs: list[str], total_runs: int, estimated_usd: float, cost_usd: float, warnings: list[str]
) -> DecisionPacket:
    return DecisionPacket(
        name="before_run",
        phase=Phase.RUN,
        question="Approve these pre-registered experiments and the compute they will spend?",
        summary="\n".join(f"  - {s}" for s in specs),
        facts={"runs": total_runs},
        cost_so_far_usd=cost_usd,
        estimated_next_usd=estimated_usd,
        warnings=warnings,
    )


def report_packet(*, verdicts: list[str], cost_usd: float) -> DecisionPacket:
    return DecisionPacket(
        name="before_report",
        phase=Phase.REPORT,
        question="Approve writing the final memo from these verdicts?",
        summary="\n".join(f"  - {v}" for v in verdicts),
        cost_so_far_usd=cost_usd,
    )


__all__ = [
    "DecisionPacket",
    "GateKeeper",
    "GateName",
    "GateResponder",
    "auto_approve_responder",
    "console_responder",
    "deny_responder",
    "report_packet",
    "run_packet",
    "survey_packet",
]
