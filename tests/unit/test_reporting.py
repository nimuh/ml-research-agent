"""Reporting: memo assembly, inline citations, tables, verdict prominence,
spread honesty, and figures that degrade to None instead of raising."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from kb_helpers import kb_config

from ml_research_agent.reporting import figures as figures_mod
from ml_research_agent.reporting.report import (
    comparison_table,
    render_report,
    results_table,
    summarize_spread,
    verdict_block,
    write_literature_review,
    write_report,
)
from ml_research_agent.types import (
    Citation,
    Comparison,
    Metric,
    MetricSummary,
    Note,
    Provenance,
    Report,
    ReportSection,
    Result,
    RunRecord,
    Verdict,
    VerdictStatus,
)


def _summary(arm: str, mean: float, std: float, values: list[float]) -> MetricSummary:
    return MetricSummary(
        name="accuracy",
        arm=arm,
        n=len(values),
        mean=mean,
        std=std,
        min=min(values),
        max=max(values),
        values=values,
    )


def _report() -> Report:
    return Report(
        brief_id="brief_1",
        title="Does sparse attention hold at long context?",
        abstract="Three seeds, one pre-registered rule, one negative result.",
        sections=[
            ReportSection(
                title="Method",
                body="We adapted the reference implementation and changed only the mask.",
                order=1,
                citations=[
                    Citation(key="arxiv:2101.00001", label="Sato et al., 2020", locator="§3.1 p.4")
                ],
            ),
            ReportSection(
                title="Results",
                body="The treatment did not clear the threshold.",
                order=2,
                citations=[Citation(key="run_abc", label="run abc", kind="run")],
            ),
            ReportSection(title="Background", body="Prior work is split.", order=0),
        ],
        citations=[
            Citation(
                key="arxiv:2101.00001",
                label="Sato et al., 2020",
                url="https://arxiv.org/abs/2101.00001",
            ),
            Citation(key="run_abc", label="run abc", kind="run"),
        ],
    )


def _verdict(status: VerdictStatus) -> Verdict:
    return Verdict(
        hypothesis_id="hyp_1",
        result_id="result_1",
        status=status,
        reasoning="The paired difference was 0.2 points against a 1.0 point threshold.",
        decision_rule_statement="Refuted unless accuracy improves by >= 1.0 across >= 3 seeds.",
        seed_variance_note="sd 0.4 across 3 seeds",
        threats_to_validity=["only one dataset"],
    )


def _result() -> Result:
    return Result(
        spec_id="spec_1",
        spec_hash="abc123",
        summaries=[
            _summary("baseline", 58.4, 0.3, [58.1, 58.4, 58.7]),
            _summary("treatment", 58.6, 0.4, [58.2, 58.6, 59.0]),
        ],
        comparisons=[
            Comparison(
                metric="accuracy",
                baseline_arm="baseline",
                treatment_arm="treatment",
                baseline=_summary("baseline", 58.4, 0.3, [58.1, 58.4, 58.7]),
                treatment=_summary("treatment", 58.6, 0.4, [58.2, 58.6, 59.0]),
                effect=0.2,
                relative_effect=0.0034,
                p_value=0.41,
                n_seeds=3,
            )
        ],
        n_seeds=3,
    )


# -- memo assembly ---------------------------------------------------------


def test_render_report_emits_every_section_in_order():
    markdown = render_report(_report(), project_id="proj_1", cost_note="$1.20 spent")
    assert markdown.startswith("# Does sparse attention hold at long context?")
    assert "proj_1" in markdown and "$1.20 spent" in markdown
    for heading in ("## Abstract", "## Background", "## Method", "## Results", "## References"):
        assert heading in markdown, heading
    assert (
        markdown.index("## Background") < markdown.index("## Method") < markdown.index("## Results")
    )


def test_citations_render_inline_with_the_section_that_used_them():
    markdown = render_report(_report())
    method = markdown[markdown.index("## Method") : markdown.index("## Results")]
    assert "*Sources:*" in method
    assert "§3.1 p.4" in method
    assert "run abc" not in method  # that citation belongs to Results


def test_the_reference_list_carries_every_source():
    markdown = render_report(_report())
    references = markdown[markdown.index("## References") :]
    assert "`[arxiv:2101.00001]`" in references
    assert "https://arxiv.org/abs/2101.00001" in references
    assert "`[run_abc]`" in references


def test_a_report_with_no_citations_says_none_rather_than_nothing():
    bare = Report(brief_id="b", title="Bare", abstract="No sources.")
    markdown = render_report(bare)
    assert "## References" in markdown
    assert "*none*" in markdown
    assert "cost not recorded" in markdown


def test_write_report_lands_under_kb_reports(tmp_path):
    config = kb_config(tmp_path)
    report = _report()
    path = write_report(report, config, project_id="proj_1")

    assert path.parent == config.paths.kb_reports
    assert path.parent.is_relative_to(tmp_path)
    assert path.suffix == ".md"
    assert "sparse-attention" in path.name
    assert path.read_text(encoding="utf-8") == render_report(report, project_id="proj_1")


def test_write_literature_review_lands_under_kb_reports(tmp_path):
    config = kb_config(tmp_path)
    notes = [
        Note(
            type="paper",
            title="Sparse attention",
            summary="Linear cost attention.",
            paper_key="arxiv:2101.00001",
            provenance=[Provenance(source="arxiv:2101.00001", locator="§Method p.2")],
        )
    ]
    path = write_literature_review(
        title="What the KB knows",
        notes=notes,
        config=config,
        synthesis={
            "settled": ["attention is quadratic"],
            "contested": [],
            "gaps": ["long context"],
        },
    )
    text = path.read_text(encoding="utf-8")

    assert path.parent == config.paths.kb_reports
    assert "## Settled" in text and "## Contested" in text and "## Gaps" in text
    assert "*none identified*" in text
    assert "Sparse attention" in text and "arxiv:2101.00001" in text


# -- tables ----------------------------------------------------------------


def test_results_table_reports_spread_next_to_every_mean():
    table = results_table(_result())
    lines = table.splitlines()
    assert lines[0].startswith("| metric | arm | n | mean | sd | range |")
    assert len(lines) == 4
    assert "58.4" in table and "0.3" in table
    assert "58.1 – 58.7" in table


def test_results_table_survives_a_result_with_no_metrics():
    empty = Result(spec_id="spec_1", spec_hash="abc123")
    assert results_table(empty) == "*no metrics recorded*"


def test_comparison_table_renders_effect_relative_and_p():
    table = comparison_table(_result().comparisons)
    assert "treatment vs baseline" in table
    assert "+0.2" in table
    assert "+0.34%" in table
    assert "0.4100" in table


def test_comparison_table_survives_an_empty_list():
    assert comparison_table([]) == "*no comparisons computed*"


def test_missing_relative_effect_and_p_render_as_placeholders():
    comparison = (
        _result().comparisons[0].model_copy(update={"relative_effect": None, "p_value": None})
    )
    row = comparison_table([comparison]).splitlines()[-1]
    assert row.count("—") == 2


# -- verdicts --------------------------------------------------------------


@pytest.mark.parametrize(
    "status", [VerdictStatus.SUPPORTED, VerdictStatus.REFUTED, VerdictStatus.INCONCLUSIVE]
)
def test_every_verdict_renders_with_the_same_prominence(status):
    block = verdict_block(_verdict(status))
    assert block.startswith("**")
    assert status.value.upper() in block
    assert "Pre-registered rule:" in block
    assert "Seed variance:" in block
    assert "Threats to validity" in block


def test_a_refuted_verdict_reads_as_a_result_not_a_failure():
    refuted = verdict_block(_verdict(VerdictStatus.REFUTED))
    supported = verdict_block(_verdict(VerdictStatus.SUPPORTED))

    assert "**REFUTED**" in refuted
    assert "the pre-registered rule fired" in refuted
    assert "fail" not in refuted.lower() and "unfortunate" not in refuted.lower()
    # Same shape, same weight: only the banner line differs.
    assert refuted.splitlines()[1:] == supported.splitlines()[1:]


def test_inconclusive_is_not_dressed_up_as_a_negative_result():
    block = verdict_block(_verdict(VerdictStatus.INCONCLUSIVE))
    assert "this is not a negative result" in block


def test_a_verdict_without_a_rule_or_threats_says_so():
    bare = Verdict(
        hypothesis_id="hyp_1", result_id="result_1", status=VerdictStatus.REFUTED, reasoning="No."
    )
    block = verdict_block(bare)
    assert "not recorded" in block
    assert "- none stated" in block
    assert "not reported" in block


# -- spread ----------------------------------------------------------------


def test_overlapping_arms_are_reported_as_within_seed_noise():
    line = summarize_spread(_result().summaries)
    assert "within seed noise" in line
    assert "accuracy" in line


def test_a_clear_separation_is_reported_as_distinguishable():
    summaries = [
        _summary("baseline", 50.0, 0.1, [49.9, 50.0, 50.1]),
        _summary("treatment", 60.0, 0.1, [59.9, 60.0, 60.1]),
    ]
    line = summarize_spread(summaries)
    assert "distinguishable" in line
    assert "+10" in line


def test_a_single_arm_has_nothing_to_compare():
    assert summarize_spread([_summary("baseline", 50.0, 0.1, [50.0])]) == (
        "- single arm; nothing to compare"
    )
    assert summarize_spread([]) == "- single arm; nothing to compare"


# -- figures ---------------------------------------------------------------


@pytest.fixture
def without_matplotlib(monkeypatch):
    """Simulate an install without the optional figure dependency."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    return None


def test_pyplot_returns_none_when_matplotlib_is_missing(without_matplotlib):
    assert figures_mod._pyplot() is None


def test_every_figure_helper_returns_none_without_matplotlib(without_matplotlib, tmp_path):
    result = _result()
    runs = [
        RunRecord(
            spec_id="spec_1",
            spec_hash="abc123",
            arm="treatment",
            seed=0,
            metrics=[Metric(name="accuracy", value=1.0, step=1)],
        )
    ]
    assert figures_mod.learning_curves(runs, "accuracy", tmp_path) is None
    assert figures_mod.baseline_vs_variant(result.summaries, "accuracy", tmp_path) is None
    assert figures_mod.ablation_grid([result], "accuracy", tmp_path) is None
    assert figures_mod.cost_breakdown({"run": 1.0}, tmp_path) is None
    assert figures_mod.figures_for_result(result, runs, tmp_path) == []
    assert not list(tmp_path.iterdir())


def test_figure_helpers_return_none_when_there_is_nothing_to_plot(tmp_path):
    pytest.importorskip("matplotlib")
    empty = Result(spec_id="spec_1", spec_hash="abc123")
    single_point = [
        RunRecord(
            spec_id="spec_1",
            spec_hash="abc123",
            metrics=[Metric(name="accuracy", value=1.0, step=1)],
        )
    ]
    assert figures_mod.learning_curves(single_point, "accuracy", tmp_path) is None
    assert figures_mod.baseline_vs_variant([], "accuracy", tmp_path) is None
    assert figures_mod.ablation_grid([empty], "accuracy", tmp_path) is None
    assert figures_mod.cost_breakdown({}, tmp_path) is None


def test_figures_are_produced_when_matplotlib_is_available(tmp_path):
    pytest.importorskip("matplotlib")
    result = _result()
    runs = [
        RunRecord(
            spec_id="spec_1",
            spec_hash="abc123",
            arm=arm,
            seed=seed,
            metrics=[Metric(name="accuracy", value=50.0 + step, step=step) for step in (1, 2, 3)],
        )
        for arm in ("baseline", "treatment")
        for seed in (0, 1)
    ]
    artifacts = figures_mod.figures_for_result(result, runs, tmp_path)

    assert artifacts
    assert all(a.kind == "figure" and a.size_bytes > 0 for a in artifacts)
    assert all(Path(a.path).exists() for a in artifacts)
    assert figures_mod.cost_breakdown({"run": 1.0, "survey": 0.2}, tmp_path) is not None
