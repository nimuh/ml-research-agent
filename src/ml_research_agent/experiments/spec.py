"""ExperimentSpec: independent/dependent variables, controls, dataset+split,
baselines, metrics, seeds, budget ceiling, stopping rule, and the pre-registered
decision rule. Hashable -> a spec hash identifies a reproducible experiment."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import ExperimentsConfig
from ..errors import MRAError
from ..types import ExperimentSpec, Scale
from ..utils.hashing import hash_obj

# Cost multipliers per rung. Rough by design -- the point is the ordering, and
# the real ceiling is enforced by the budget, not by this estimate.
SCALE_COST_FACTOR: dict[Scale, float] = {Scale.SMOKE: 0.05, Scale.SMALL: 1.0, Scale.MAIN: 12.0}
SCALE_RUNTIME_FACTOR: dict[Scale, float] = {Scale.SMOKE: 0.05, Scale.SMALL: 1.0, Scale.MAIN: 10.0}


def validate_spec(spec: ExperimentSpec, config: ExperimentsConfig) -> list[str]:
    """Return every reason this spec should not run. Empty means it may.

    Kept separate from the pydantic model because these are *policy* checks --
    they depend on configuration, and a spec that is invalid under one policy is
    valid under another.
    """
    problems: list[str] = []
    rule = spec.decision_rule

    if config.require_preregistered_decision_rule and not rule.statement.strip():
        problems.append("no pre-registered decision rule; the hypothesis would be unfalsifiable")
    if rule.metric.lower() not in {m.name.lower() for m in spec.dependent_variables}:
        problems.append(f"the decision rule keys on '{rule.metric}', which is not measured")
    if len(spec.seeds) < rule.min_seeds:
        problems.append(
            f"{len(spec.seeds)} seeds configured but the rule requires {rule.min_seeds}"
        )
    if len(spec.seeds) < 2:
        problems.append("single-seed experiments are inconclusive by construction")
    if not spec.baselines and rule.relative_to_baseline:
        problems.append("the rule is relative to a baseline but no baseline is defined")
    if spec.treatment and spec.treatment in spec.baselines:
        problems.append("the treatment is also listed as a baseline; the comparison is vacuous")
    if not spec.controls:
        problems.append("no controls declared; nothing is held fixed")
    if spec.max_runtime_minutes > config.max_runtime_minutes:
        problems.append(
            f"runtime ceiling {spec.max_runtime_minutes}m exceeds the configured "
            f"{config.max_runtime_minutes}m"
        )
    return problems


def assert_runnable(spec: ExperimentSpec, config: ExperimentsConfig) -> None:
    problems = validate_spec(spec, config)
    if problems:
        raise MRAError("spec rejected", spec=spec.id, problems=problems)


def spec_hash(spec: ExperimentSpec) -> str:
    """The experiment's identity. Delegates to the model so there is one definition."""
    return spec.spec_hash


def run_key(spec: ExperimentSpec, *, code_hash: str, env_hash: str, seed: int) -> str:
    """The reproducibility contract as a single lookup key."""
    return hash_obj([spec.spec_hash, code_hash, env_hash, seed], length=24)


def next_scale(current: Scale, config: ExperimentsConfig) -> Scale | None:
    """The next rung up, or None at the top. Skipping rungs is not offered."""
    ladder = list(config.scale_ladder)
    try:
        index = ladder.index(current)
    except ValueError:
        return None
    return ladder[index + 1] if index + 1 < len(ladder) else None


def scale_up(spec: ExperimentSpec, config: ExperimentsConfig) -> ExperimentSpec | None:
    """Promote a spec one rung, scaling its budget and runtime with it."""
    target = next_scale(spec.scale, config)
    if target is None:
        return None
    cost_ratio = SCALE_COST_FACTOR[target] / SCALE_COST_FACTOR[spec.scale]
    time_ratio = SCALE_RUNTIME_FACTOR[target] / SCALE_RUNTIME_FACTOR[spec.scale]
    return spec.model_copy(
        update={
            "id": spec.id,
            "scale": target,
            "budget_usd": spec.budget_usd * cost_ratio,
            "max_runtime_minutes": min(
                int(spec.max_runtime_minutes * time_ratio), config.max_runtime_minutes
            ),
        }
    )


def estimate_cost(spec: ExperimentSpec) -> float:
    """Total spend if every arm runs at every seed."""
    runs = len(spec.seeds) * max(len(spec.arms), 1)
    return spec.budget_usd * runs * SCALE_COST_FACTOR.get(spec.scale, 1.0)


def planned_runs(spec: ExperimentSpec) -> list[tuple[str, int]]:
    """Every (arm, seed) pair the spec commits to running."""
    return [(arm, seed) for arm in (spec.arms or ["treatment"]) for seed in spec.seeds]


def dedupe_specs(specs: Sequence[ExperimentSpec]) -> list[ExperimentSpec]:
    """Collapse specs that describe the same experiment under different ids."""
    seen: dict[str, ExperimentSpec] = {}
    for spec in specs:
        seen.setdefault(spec.spec_hash, spec)
    return list(seen.values())


__all__ = [
    "SCALE_COST_FACTOR",
    "SCALE_RUNTIME_FACTOR",
    "assert_runnable",
    "dedupe_specs",
    "estimate_cost",
    "next_scale",
    "planned_runs",
    "run_key",
    "scale_up",
    "spec_hash",
    "validate_spec",
]
