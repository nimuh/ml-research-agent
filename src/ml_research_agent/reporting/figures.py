"""Standard figures: learning curves, baseline-vs-variant bars with error bars,
ablation grids, cost/compute breakdowns."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..types import Artifact, MetricSummary, Result, RunRecord
from ..utils.io import ensure_dir


def _pyplot() -> Any:
    """Import matplotlib lazily with a headless backend.

    Figures are optional: a report must still render on a machine where
    matplotlib is not installed, so callers handle ``None``.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - exercised only without the extra
        return None
    return plt


def _artifact(path: Path, name: str) -> Artifact:
    return Artifact(
        name=name,
        path=str(path),
        kind="figure",
        size_bytes=path.stat().st_size if path.exists() else 0,
    )


def learning_curves(runs: list[RunRecord], metric: str, out_dir: Path) -> Artifact | None:
    """One line per (arm, seed): the plot that makes a silent no-op visible."""
    plt = _pyplot()
    if plt is None:
        return None
    series = [
        (
            r,
            [
                (m.step or i, m.value)
                for i, m in enumerate(r.metrics)
                if m.name == metric and m.step is not None
            ],
        )
        for r in runs
    ]
    series = [(r, points) for r, points in series if len(points) > 1]
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for run, points in series:
        points.sort()
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            label=f"{run.arm} s{run.seed}",
            alpha=0.8,
        )
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} over training")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = ensure_dir(out_dir) / f"learning-curve-{metric}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return _artifact(path, f"learning curve: {metric}")


def baseline_vs_variant(
    summaries: list[MetricSummary], metric: str, out_dir: Path
) -> Artifact | None:
    """Bars with error bars. Bars without error bars would hide the whole question."""
    plt = _pyplot()
    if plt is None:
        return None
    group = [s for s in summaries if s.name == metric]
    if not group:
        return None

    fig, ax = plt.subplots(figsize=(1.6 * len(group) + 2, 4))
    arms = [s.arm for s in group]
    means = [s.mean for s in group]
    errs = [s.std for s in group]
    ax.bar(arms, means, yerr=errs, capsize=6, alpha=0.85)
    for i, summary in enumerate(group):
        # Individual seeds over the bar: the reader can see cherry-picking.
        ax.scatter(
            [i] * len(summary.values), summary.values, s=14, zorder=3, color="black", alpha=0.6
        )
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by arm (mean ± sd, n seeds shown)")
    fig.tight_layout()
    path = ensure_dir(out_dir) / f"arms-{metric}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return _artifact(path, f"arms: {metric}")


def ablation_grid(results: list[Result], metric: str, out_dir: Path) -> Artifact | None:
    """Heatmap of spec x arm for one metric."""
    plt = _pyplot()
    if plt is None:
        return None
    rows = sorted({r.spec_id for r in results})
    cols = sorted({s.arm for r in results for s in r.summaries if s.name == metric})
    if not rows or not cols:
        return None

    matrix = [
        [
            next(
                (s.mean for s in result.summaries if s.name == metric and s.arm == arm),
                float("nan"),
            )
            for arm in cols
        ]
        for spec_id in rows
        for result in [next(r for r in results if r.spec_id == spec_id)]
    ]
    fig, ax = plt.subplots(figsize=(1.4 * len(cols) + 3, 0.7 * len(rows) + 2))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), [r[:18] for r in rows])
    ax.set_title(f"{metric} across experiments")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = ensure_dir(out_dir) / f"ablation-{metric}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return _artifact(path, f"ablation grid: {metric}")


def cost_breakdown(by_phase: dict[str, float], out_dir: Path) -> Artifact | None:
    """Where the money went, by phase."""
    plt = _pyplot()
    if plt is None or not by_phase:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    items = sorted(by_phase.items(), key=lambda kv: -kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items], alpha=0.85)
    ax.set_xlabel("USD")
    ax.set_title("cost by phase")
    fig.tight_layout()
    path = ensure_dir(out_dir) / "cost-by-phase.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return _artifact(path, "cost by phase")


def figures_for_result(
    result: Result, runs: list[RunRecord], out_dir: Path, *, metrics: list[str] | None = None
) -> list[Artifact]:
    """Every standard figure a result supports, skipping what it cannot."""
    names = metrics or sorted({s.name for s in result.summaries})
    artifacts: list[Artifact] = []
    for metric in names:
        for figure in (
            baseline_vs_variant(result.summaries, metric, out_dir),
            learning_curves(runs, metric, out_dir),
        ):
            if figure is not None:
                artifacts.append(figure)
    return artifacts


__all__ = [
    "ablation_grid",
    "baseline_vs_variant",
    "cost_breakdown",
    "figures_for_result",
    "learning_curves",
]
