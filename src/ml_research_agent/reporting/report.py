"""Report assembly from ProjectState + KB + results; Markdown first, with
optional HTML/PDF export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..types import (
    Citation,
    Comparison,
    MetricSummary,
    Report,
    Result,
    Verdict,
    utcnow,
)
from ..utils.io import slugify, write_text
from .templates import (
    CITATION_LINE,
    COMPARISON_TABLE_HEADER,
    COMPARISON_TABLE_ROW,
    MEMO,
    RESULTS_TABLE_HEADER,
    RESULTS_TABLE_ROW,
    SECTION,
    VERDICT_BANNER,
)


def render_report(report: Report, *, project_id: str = "", cost_note: str = "") -> str:
    """Render the memo to Markdown.

    Citations render inline with every section that used them, so a reader can
    check a claim without scrolling to a bibliography and back.
    """
    sections = "\n".join(
        SECTION.format(
            heading="#" * max(2, min(section.level, 6)),
            title=section.title,
            body=section.body.strip(),
            citations=_render_citations(section.citations, inline=True),
        )
        for section in sorted(report.sections, key=lambda s: s.order)
    )
    return MEMO.format(
        title=report.title,
        generated_at=report.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        project_id=project_id or "-",
        cost_note=cost_note or "cost not recorded",
        abstract=report.abstract.strip(),
        sections=sections,
        references=_render_citations(report.citations) or "*none*",
    )


def _render_citations(citations: list[Citation], *, inline: bool = False) -> str:
    if not citations:
        return ""
    lines = [
        CITATION_LINE.format(
            key=c.key,
            label=c.label,
            locator=f" — {c.locator}" if c.locator else "",
            url=f" <{c.url}>" if c.url else "",
        )
        for c in citations
    ]
    body = "\n".join(lines)
    return f"\n*Sources:*\n{body}\n" if inline else body


def results_table(result: Result) -> str:
    """Per-arm summaries with spread. Means without spread are not reportable."""
    if not result.summaries:
        return "*no metrics recorded*"
    rows = [
        RESULTS_TABLE_ROW.format(
            metric=s.name, arm=s.arm, n=s.n, mean=s.mean, std=s.std, lo=s.min, hi=s.max
        )
        for s in sorted(result.summaries, key=lambda s: (s.name, s.arm))
    ]
    return "\n".join([RESULTS_TABLE_HEADER, *rows])


def comparison_table(comparisons: list[Comparison]) -> str:
    if not comparisons:
        return "*no comparisons computed*"
    rows = [
        COMPARISON_TABLE_ROW.format(
            metric=c.metric,
            treatment=c.treatment_arm,
            baseline=c.baseline_arm,
            effect=c.effect,
            relative=f"{c.relative_effect:+.2%}" if c.relative_effect is not None else "—",
            p=f"{c.p_value:.4f}" if c.p_value is not None else "—",
            n=c.n_seeds,
        )
        for c in comparisons
    ]
    return "\n".join([COMPARISON_TABLE_HEADER, *rows])


def verdict_block(verdict: Verdict) -> str:
    """A verdict rendered so a negative result reads as a result."""
    banner = VERDICT_BANNER.get(verdict.status.value, verdict.status.value)
    threats = "\n".join(f"- {t}" for t in verdict.threats_to_validity) or "- none stated"
    return (
        f"{banner}\n\n"
        f"*Pre-registered rule:* {verdict.decision_rule_statement or 'not recorded'}\n\n"
        f"{verdict.reasoning.strip()}\n\n"
        f"*Seed variance:* {verdict.seed_variance_note or 'not reported'}\n\n"
        f"**Threats to validity**\n{threats}\n"
    )


def summarize_spread(summaries: list[MetricSummary]) -> str:
    """One line per metric describing whether arms are actually distinguishable."""
    lines: list[str] = []
    by_metric: dict[str, list[MetricSummary]] = {}
    for summary in summaries:
        by_metric.setdefault(summary.name, []).append(summary)
    for metric, group in sorted(by_metric.items()):
        if len(group) < 2:
            continue
        best, worst = max(group, key=lambda s: s.mean), min(group, key=lambda s: s.mean)
        gap = best.mean - worst.mean
        noise = best.std + worst.std
        verdict = "distinguishable" if gap > noise else "within seed noise"
        lines.append(f"- **{metric}**: {best.arm} − {worst.arm} = {gap:+.5g} ({verdict})")
    return "\n".join(lines) or "- single arm; nothing to compare"


def write_report(
    report: Report, config: Config, *, project_id: str = "", cost_note: str = ""
) -> Path:
    """Write the memo into ``workspace/kb/reports/`` and return its path."""
    stamp = utcnow().strftime("%Y%m%d-%H%M")
    filename = f"{stamp}-{slugify(report.title)}.md"
    path = Path(config.paths.kb_reports) / filename
    write_text(path, render_report(report, project_id=project_id, cost_note=cost_note))
    return path


def write_literature_review(
    *, title: str, notes: list[Any], config: Config, synthesis: dict[str, Any] | None = None
) -> Path:
    """A standalone review of the KB, useful long before any experiment runs."""
    lines = [f"# {title}", "", f"*{len(notes)} notes · generated {utcnow():%Y-%m-%d}*", ""]
    if synthesis:
        for bucket in ("settled", "contested", "gaps"):
            items = synthesis.get(bucket) or []
            lines += [f"## {bucket.title()}", ""]
            lines += [f"- {item}" for item in items] or ["*none identified*"]
            lines.append("")
    lines += ["## Notes", ""]
    for note in notes:
        key = getattr(note, "paper_key", None) or getattr(note, "id", "?")
        lines.append(f"### {getattr(note, 'title', key)}")
        lines.append("")
        lines.append(f"`{key}` — {getattr(note, 'summary', '')}")
        lines.append("")
    path = Path(config.paths.kb_reports) / f"{utcnow():%Y%m%d-%H%M}-{slugify(title)}.md"
    write_text(path, "\n".join(lines))
    return path


__all__ = [
    "comparison_table",
    "render_report",
    "results_table",
    "summarize_spread",
    "verdict_block",
    "write_literature_review",
    "write_report",
]
