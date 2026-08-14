"""Exception hierarchy: MRAError -> {ConfigError, SourceError, BudgetExceeded,
SandboxViolation, ToolError, AgentFailure, IrreproducibleRun}. Errors carry a
`retryable` flag the orchestrator uses to decide retry vs. escalate."""

from __future__ import annotations

from typing import Any


class MRAError(Exception):
    """Base class for every error this system raises deliberately.

    ``retryable`` is the orchestrator's signal: retry the step (possibly with
    adjusted parameters) versus escalate to a gate or abort the phase.
    """

    retryable: bool = False

    def __init__(self, message: str, *, retryable: bool | None = None, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        if retryable is not None:
            self.retryable = retryable
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} [{detail}]"


class ConfigError(MRAError):
    """Malformed, missing, or contradictory configuration. Never retryable."""

    retryable = False


class SourceError(MRAError):
    """A literature/code source failed: HTTP error, rate limit, parse failure.

    Retryable by default -- source flakiness is expected and bounded fan-out
    plus backoff is the answer, not aborting a survey.
    """

    retryable = True


class BudgetExceeded(MRAError):
    """A hard ceiling (USD, tokens, wall clock) was reached.

    Never retryable: the invariant is that budgets stop work rather than
    silently degrading it.
    """

    retryable = False

    def __init__(
        self,
        message: str,
        *,
        scope: str = "project",
        limit: float | None = None,
        spent: float | None = None,
        unit: str = "usd",
        **context: Any,
    ) -> None:
        super().__init__(
            message, retryable=False, scope=scope, limit=limit, spent=spent, unit=unit, **context
        )
        self.scope = scope
        self.limit = limit
        self.spent = spent
        self.unit = unit


class SandboxViolation(MRAError):
    """Code tried to leave its sandbox: path escape, denied network, denied command.

    Never retryable -- a violation is a security event, not a transient failure.
    """

    retryable = False


class ToolError(MRAError):
    """A tool call failed or was refused (not permitted, bad arguments, timeout)."""

    retryable = True


class AgentFailure(MRAError):
    """An agent could not produce valid output after its repair attempts."""

    retryable = True


class IrreproducibleRun(MRAError):
    """The reproducibility contract broke: same (spec, code, env, seed), different metrics."""

    retryable = False


class GateRejected(MRAError):
    """A human gate was rejected; the phase must not proceed."""

    retryable = False


class PhasePreconditionFailed(MRAError):
    """A phase was entered without the state it declares it needs."""

    retryable = False


__all__ = [
    "AgentFailure",
    "BudgetExceeded",
    "ConfigError",
    "GateRejected",
    "IrreproducibleRun",
    "MRAError",
    "PhasePreconditionFailed",
    "SandboxViolation",
    "SourceError",
    "ToolError",
]
