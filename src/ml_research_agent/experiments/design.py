"""Design helpers: ablation grids, seed/replicate policy, power/effect-size
sanity checks, and cost estimation before anything is launched."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

from ..config import ExperimentsConfig
from ..types import ExperimentSpec, Scale, Variable
from .metrics import t_critical
from .spec import SCALE_COST_FACTOR


def ablation_grid(variables: Sequence[Variable], *, max_cells: int = 24) -> list[dict[str, Any]]:
    """Full factorial over the variables, truncated with the truncation visible.

    Returning a silently-truncated grid would understate what the experiment
    actually covers, so the caller is expected to log the drop.
    """
    if not variables:
        return []
    combos = list(itertools.product(*[v.values or [None] for v in variables]))
    cells = [dict(zip((v.name for v in variables), combo, strict=True)) for combo in combos]
    return cells[:max_cells]


def grid_truncation(variables: Sequence[Variable], *, max_cells: int = 24) -> int:
    """How many cells a grid would drop, so it can be logged rather than hidden."""
    total = 1
    for variable in variables:
        total *= max(len(variable.values), 1)
    return max(0, total - max_cells)


def one_factor_at_a_time(
    variables: Sequence[Variable], baseline: dict[str, Any]
) -> list[dict[str, Any]]:
    """Cheaper than factorial and enough to attribute an effect to one knob."""
    cells: list[dict[str, Any]] = [dict(baseline)]
    for variable in variables:
        for value in variable.values:
            if baseline.get(variable.name) == value:
                continue
            cells.append({**baseline, variable.name: value})
    return cells


def seeds_for(config: ExperimentsConfig, *, start: int = 0, count: int | None = None) -> list[int]:
    """Contiguous seeds from a fixed start, so runs are reproducible by construction."""
    n = count or config.default_seeds
    return list(range(start, start + max(n, 1)))


def required_seeds(effect_size: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """Seeds needed to detect a standardized effect, by normal approximation.

    Deliberately approximate. Its job is to catch "you are trying to detect a
    0.1 sd effect with 3 seeds" before the compute is spent, not to be a
    statistics package.
    """
    if effect_size <= 0:
        return 99
    z_alpha = 1.959964 if alpha <= 0.05 else 1.644854
    z_power = 0.8416 if power <= 0.8 else 1.2816
    n = 2 * ((z_alpha + z_power) / effect_size) ** 2
    return max(2, math.ceil(n))


def detectable_effect(n_seeds: int, *, power: float = 0.8, alpha: float = 0.05) -> float:
    """The smallest standardized effect this many seeds could actually detect."""
    if n_seeds < 2:
        return float("inf")
    z_alpha = t_critical(n_seeds - 1) if alpha <= 0.05 else 1.644854
    z_power = 0.8416 if power <= 0.8 else 1.2816
    return (z_alpha + z_power) * math.sqrt(2.0 / n_seeds)


def power_check(spec: ExperimentSpec) -> list[str]:
    """Warn when the design cannot answer the question it pre-registered."""
    warnings: list[str] = []
    n = len(spec.seeds)
    rule = spec.decision_rule
    smallest = detectable_effect(n)

    if n < rule.min_seeds:
        warnings.append(f"{n} seeds configured but the decision rule requires {rule.min_seeds}")
    if rule.min_effect_size and rule.min_effect_size < smallest:
        warnings.append(
            f"the rule's minimum effect ({rule.min_effect_size:.3g} sd) is below what {n} seeds "
            f"can detect ({smallest:.3g} sd); an inconclusive result is the likely outcome"
        )
    if n >= 2 and not rule.min_effect_size:
        warnings.append(
            "no minimum effect size is set, so any positive difference passes regardless of size"
        )
    return warnings


def estimate_runtime_minutes(spec: ExperimentSpec, *, parallelism: int = 1) -> float:
    """Wall clock if every arm runs at every seed at the given parallelism."""
    runs = max(len(spec.arms), 1) * len(spec.seeds)
    batches = math.ceil(runs / max(parallelism, 1))
    return batches * spec.max_runtime_minutes


def estimate_total_cost(specs: Sequence[ExperimentSpec]) -> dict[str, float]:
    """Cost per scale plus the total, for the pre-RUN decision packet."""
    per_scale: dict[str, float] = {}
    for spec in specs:
        runs = max(len(spec.arms), 1) * len(spec.seeds)
        cost = spec.budget_usd * runs * SCALE_COST_FACTOR.get(spec.scale, 1.0)
        per_scale[spec.scale.value] = per_scale.get(spec.scale.value, 0.0) + cost
    per_scale["total"] = sum(v for k, v in per_scale.items() if k != "total")
    return per_scale


def ladder_for(spec: ExperimentSpec, config: ExperimentsConfig) -> list[Scale]:
    """The rungs this spec must climb, starting at its current scale."""
    ladder = list(config.scale_ladder)
    try:
        return ladder[ladder.index(spec.scale) :]
    except ValueError:
        return ladder


def design_warnings(spec: ExperimentSpec, config: ExperimentsConfig) -> list[str]:
    """Everything worth telling a human before they approve the bill."""
    warnings = power_check(spec)
    if spec.scale is not Scale.SMOKE and config.scale_ladder[0] is Scale.SMOKE:
        warnings.append(
            f"this spec starts at {spec.scale.value}, skipping the smoke rung; "
            "cheap-before-expensive is not being honored"
        )
    if not spec.controls:
        warnings.append("no controls declared: nothing is held fixed across arms")
    if (
        estimate_runtime_minutes(spec, parallelism=config.max_parallel_runs)
        > config.max_runtime_minutes * 8
    ):
        warnings.append("estimated wall clock is very large relative to the per-run ceiling")
    return warnings


__all__ = [
    "ablation_grid",
    "design_warnings",
    "detectable_effect",
    "estimate_runtime_minutes",
    "estimate_total_cost",
    "grid_truncation",
    "ladder_for",
    "one_factor_at_a_time",
    "power_check",
    "required_seeds",
    "seeds_for",
]
