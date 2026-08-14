"""Markdown templates for reports and decision packets, kept as data.

Templates live here rather than inline in ``report.py`` for the same reason
prompts live under ``prompts/``: a change to how findings are presented should
be diffable on its own, without touching assembly logic.
"""

from __future__ import annotations

MEMO = """# {title}

*Generated {generated_at} · project {project_id} · {cost_note}*

## Abstract

{abstract}

{sections}

## References

{references}
"""

SECTION = """{heading} {title}

{body}
{citations}
"""

CITATION_LINE = "- `[{key}]` {label}{locator}{url}"

VERDICT_BANNER = {
    "supported": "**SUPPORTED** — the pre-registered rule fired.",
    "refuted": "**REFUTED** — the pre-registered rule fired against the hypothesis.",
    "inconclusive": "**INCONCLUSIVE** — the rule did not fire; this is not a negative result.",
}

DECISION_PACKET = """## Decision required: {name}

{question}

{summary}

| fact | value |
|---|---|
{facts}

Spent so far: **${cost:.2f}**{estimate}

{warnings}
"""

RESULTS_TABLE_HEADER = "| metric | arm | n | mean | sd | range |\n|---|---|---|---|---|---|"
RESULTS_TABLE_ROW = "| {metric} | {arm} | {n} | {mean:.5g} | {std:.5g} | {lo:.5g} – {hi:.5g} |"

COMPARISON_TABLE_HEADER = (
    "| metric | treatment vs baseline | effect | relative | p | seeds |\n|---|---|---|---|---|---|"
)
COMPARISON_TABLE_ROW = (
    "| {metric} | {treatment} vs {baseline} | {effect:+.5g} | {relative} | {p} | {n} |"
)

__all__ = [
    "CITATION_LINE",
    "COMPARISON_TABLE_HEADER",
    "COMPARISON_TABLE_ROW",
    "DECISION_PACKET",
    "MEMO",
    "RESULTS_TABLE_HEADER",
    "RESULTS_TABLE_ROW",
    "SECTION",
    "VERDICT_BANNER",
]
