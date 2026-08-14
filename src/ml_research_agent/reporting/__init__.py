"""Report generation: literature reviews, experiment reports, and the final
research memo. Every claim renders with its citation or run id."""

from .figures import (
    ablation_grid,
    baseline_vs_variant,
    cost_breakdown,
    figures_for_result,
    learning_curves,
)
from .report import (
    comparison_table,
    render_report,
    results_table,
    summarize_spread,
    verdict_block,
    write_literature_review,
    write_report,
)

__all__ = [
    "ablation_grid",
    "baseline_vs_variant",
    "comparison_table",
    "cost_breakdown",
    "figures_for_result",
    "learning_curves",
    "render_report",
    "results_table",
    "summarize_spread",
    "verdict_block",
    "write_literature_review",
    "write_report",
]
