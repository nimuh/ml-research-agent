"""RunnerAgent: schedules and babysits executions -- launches runs, streams logs,
detects hangs/NaNs/OOM, retries with adjusted resources, records artifacts."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ..types import ExperimentSpec, FailureSignature, ModelTier, RunRecord
from .base import AgentContext, BaseAgent

# Deterministic first, model second: these are the cases where the right retry
# is knowable without asking anyone. The LLM only sees what falls through.
DETERMINISTIC_RETRY: dict[FailureSignature, str] = {
    FailureSignature.OOM: (
        "halve the batch size and raise gradient accumulation to keep the effective batch"
    ),
    FailureSignature.TIMEOUT: (
        "reduce steps to the smoke budget, or raise the ceiling if the run was progressing"
    ),
    FailureSignature.NAN: (
        "lower the learning rate and re-check the loss for an unguarded log or division"
    ),
    FailureSignature.DEPENDENCY: "pin or install the missing dependency before re-launching",
}


class TriageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spec: ExperimentSpec
    run: RunRecord
    log_tail: str = ""
    attempt: int = 1
    max_attempts: int = 3


class TriageDecision(BaseModel):
    """What to do about a failed run, and whether it is worth doing at all."""

    model_config = ConfigDict(extra="forbid")

    failure: FailureSignature
    diagnosis: str
    action: str = Field(description="retry | retry_adjusted | abandon | escalate")
    adjustments: dict[str, str] = Field(
        default_factory=dict, description="Config changes for the retry, e.g. batch_size -> 16."
    )
    rationale: str = ""


class RunnerAgent(BaseAgent[TriageInput, TriageDecision]):
    """Exists to prevent silent hangs, NaNs, OOM and wasted GPU-hours.

    Scheduling and execution are deterministic (``experiments/execute.py``);
    this agent is consulted only when a run fails in a way the signature table
    does not already answer -- retrying an OOM at the same batch size is not a
    strategy, but neither is asking a model what "OOM" means.
    """

    name: ClassVar[str] = "runner"
    description: ClassVar[str] = "Triages run failures and decides the retry policy."
    tier: ClassVar[ModelTier] = ModelTier.STANDARD
    prompt_name: ClassVar[str] = "runner"
    tool_allowlist: ClassVar[tuple[str, ...]] = ("read_file",)
    output_model: ClassVar[type[BaseModel]] = TriageDecision

    def prompt_variables(self, payload: TriageInput, ctx: AgentContext) -> dict[str, object]:
        return {
            "title": payload.spec.title,
            "arm": payload.run.arm,
            "seed": payload.run.seed,
            "status": payload.run.status.value,
            "detected_failure": payload.run.failure.value
            if payload.run.failure
            else "uncategorized",
            "failure_detail": payload.run.failure_detail or "none",
            "duration_seconds": payload.run.duration_seconds,
            "attempt": payload.attempt,
            "max_attempts": payload.max_attempts,
            "log_tail": (payload.log_tail or payload.run.stdout_tail)[-6000:]
            or "no output captured",
            "known_policies": "\n".join(
                f"- {k.value}: {v}" for k, v in DETERMINISTIC_RETRY.items()
            ),
        }

    def validate_output(
        self, output: TriageDecision, payload: TriageInput, ctx: AgentContext
    ) -> list[str]:
        violations: list[str] = []
        if output.action not in {"retry", "retry_adjusted", "abandon", "escalate"}:
            violations.append(f"unknown action '{output.action}'")
        if output.action == "retry_adjusted" and not output.adjustments:
            violations.append("retry_adjusted with no adjustments is just a retry")
        if (
            output.failure is FailureSignature.OOM
            and output.action == "retry"
            and not output.adjustments
        ):
            violations.append(
                "retrying an OOM unchanged will OOM again; adjust batch size or memory footprint"
            )
        if payload.attempt >= payload.max_attempts and output.action.startswith("retry"):
            violations.append("retry budget is exhausted; choose abandon or escalate")
        return violations


def default_policy(failure: FailureSignature | None) -> str | None:
    """The known-answer table. Returns None when the model should be consulted."""
    return DETERMINISTIC_RETRY.get(failure) if failure else None


__all__ = ["DETERMINISTIC_RETRY", "RunnerAgent", "TriageDecision", "TriageInput", "default_policy"]
