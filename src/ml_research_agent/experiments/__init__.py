"""Experiment layer: hypothesis -> pre-registered spec -> code -> sandboxed run
-> tracked artifacts -> analysis -> verdict. Designed so a negative result is
as legible and as trustworthy as a positive one."""

from .ablation import best_followup, propose_followups
from .analysis import analyze_runs, evaluate_decision_rule, is_single_seed, seed_variance_note
from .design import (
    ablation_grid,
    design_warnings,
    estimate_total_cost,
    power_check,
    required_seeds,
    seeds_for,
)
from .execute import classify_failure, execute_run, execute_spec, smoke_passed
from .hypothesis import is_testable, validate_hypothesis
from .metrics import compare, overlapping_error_bars, summarize, summarize_runs
from .sandbox import Sandbox, SandboxResult
from .spec import assert_runnable, estimate_cost, next_scale, scale_up, validate_spec
from .tracking import RunStore, collect_artifacts, parse_metrics_file
from .workspace import ExperimentWorkspace

__all__ = [
    "ExperimentWorkspace",
    "RunStore",
    "Sandbox",
    "SandboxResult",
    "ablation_grid",
    "analyze_runs",
    "assert_runnable",
    "best_followup",
    "classify_failure",
    "collect_artifacts",
    "compare",
    "design_warnings",
    "estimate_cost",
    "estimate_total_cost",
    "evaluate_decision_rule",
    "execute_run",
    "execute_spec",
    "is_single_seed",
    "is_testable",
    "next_scale",
    "overlapping_error_bars",
    "parse_metrics_file",
    "power_check",
    "propose_followups",
    "required_seeds",
    "scale_up",
    "seed_variance_note",
    "seeds_for",
    "smoke_passed",
    "summarize",
    "summarize_runs",
    "validate_hypothesis",
    "validate_spec",
]
