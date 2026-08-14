"""Statistics and rule evaluation: the machinery that decides what a run means."""

from __future__ import annotations

import math

import pytest

from ml_research_agent.experiments.analysis import (
    analyze_runs,
    evaluate_decision_rule,
    is_single_seed,
    seed_variance_note,
)
from ml_research_agent.experiments.metrics import (
    compare,
    effect_interval,
    overlapping_error_bars,
    paired_t_test,
    summarize,
)
from ml_research_agent.types import (
    Comparison,
    DatasetRef,
    DecisionRule,
    ExperimentSpec,
    Metric,
    MetricDef,
    MetricSummary,
    Result,
    RunRecord,
    RunStatus,
    VerdictStatus,
)


def make_spec(*, higher_is_better: bool = True, **overrides) -> ExperimentSpec:
    defaults = dict(
        hypothesis_id="hyp",
        title="t",
        dependent_variables=[MetricDef(name="accuracy", higher_is_better=higher_is_better)],
        dataset=DatasetRef(name="toy"),
        baselines=["baseline"],
        treatment="treatment",
        controls=["matched compute"],
        seeds=[0, 1, 2],
        decision_rule=DecisionRule(metric="accuracy", comparator=">", threshold=0.01, min_seeds=3),
    )
    return ExperimentSpec(**{**defaults, **overrides})


def runs_for(
    spec: ExperimentSpec, values: dict[str, list[float]], *, metric: str = "accuracy"
) -> list[RunRecord]:
    records = []
    for arm, seed_values in values.items():
        for seed, value in enumerate(seed_values):
            records.append(
                RunRecord(
                    spec_id=spec.id,
                    spec_hash=spec.spec_hash,
                    seed=seed,
                    arm=arm,
                    status=RunStatus.COMPLETED,
                    metrics=[Metric(name=metric, value=value, arm=arm)],
                )
            )
    return records


def _summary(arm: str, mean: float) -> MetricSummary:
    return MetricSummary(
        name="accuracy", arm=arm, n=3, mean=mean, std=0.002, min=mean - 0.002, max=mean + 0.002
    )


def _comparison(baseline_arm: str, *, effect: float, p_value: float) -> Comparison:
    """A comparison built directly, for paths ``analyze_runs`` cannot produce."""
    return Comparison(
        metric="accuracy",
        baseline_arm=baseline_arm,
        treatment_arm="treatment",
        baseline=_summary(baseline_arm, 0.5),
        treatment=_summary("treatment", 0.5 + effect),
        effect=effect,
        p_value=p_value,
        n_seeds=3,
    )


class TestSummarize:
    def test_reports_spread_not_just_a_mean(self):
        summary = summarize([0.1, 0.2, 0.3], name="accuracy", arm="a")
        assert summary.mean == pytest.approx(0.2)
        assert summary.std > 0
        assert summary.ci_low is not None and summary.ci_high is not None

    def test_single_value_has_no_interval(self):
        summary = summarize([0.5], name="accuracy", arm="a")
        assert summary.n == 1
        assert summary.ci_low is None

    def test_empty_is_not_a_crash(self):
        summary = summarize([], name="accuracy", arm="a")
        assert summary.n == 0
        assert math.isnan(summary.mean)


class TestComparisons:
    def test_pairs_by_seed_when_seeds_match(self):
        spec = make_spec()
        runs = runs_for(spec, {"baseline": [0.50, 0.52, 0.54], "treatment": [0.55, 0.57, 0.59]})
        comparison = compare(runs, "accuracy", baseline_arm="baseline", treatment_arm="treatment")
        assert comparison is not None
        assert "paired" in comparison.note
        assert comparison.effect == pytest.approx(0.05)
        assert comparison.n_seeds == 3

    def test_returns_none_when_an_arm_is_missing(self):
        spec = make_spec()
        runs = runs_for(spec, {"baseline": [0.5, 0.5, 0.5]})
        assert compare(runs, "accuracy", baseline_arm="baseline", treatment_arm="treatment") is None

    def test_overlapping_error_bars_are_detected(self):
        spec = make_spec()
        noisy = runs_for(spec, {"baseline": [0.40, 0.55, 0.70], "treatment": [0.42, 0.57, 0.72]})
        comparison = compare(noisy, "accuracy", baseline_arm="baseline", treatment_arm="treatment")
        assert comparison is not None
        assert overlapping_error_bars(comparison)

    def test_separated_arms_do_not_overlap(self):
        spec = make_spec()
        clean = runs_for(spec, {"baseline": [0.50, 0.50, 0.51], "treatment": [0.90, 0.90, 0.91]})
        comparison = compare(clean, "accuracy", baseline_arm="baseline", treatment_arm="treatment")
        assert comparison is not None
        assert not overlapping_error_bars(comparison)

    def test_paired_test_needs_two_pairs(self):
        assert paired_t_test([(0.5, 0.4)]) is None
        assert paired_t_test([(0.5, 0.4), (0.6, 0.5)]) is not None


class TestAnalyzeRuns:
    def test_failed_runs_are_counted_never_dropped_silently(self):
        spec = make_spec()
        runs = runs_for(spec, {"baseline": [0.5, 0.5, 0.5], "treatment": [0.6, 0.6, 0.6]})
        runs.append(
            RunRecord(
                spec_id=spec.id,
                spec_hash=spec.spec_hash,
                seed=3,
                arm="treatment",
                status=RunStatus.FAILED,
            )
        )
        result = analyze_runs(spec, runs)
        assert result.failed_runs == 1
        assert any("failed" in note for note in result.notes)

    def test_overlapping_bars_are_noted_on_the_result(self):
        spec = make_spec()
        runs = runs_for(spec, {"baseline": [0.40, 0.55, 0.70], "treatment": [0.42, 0.57, 0.72]})
        result = analyze_runs(spec, runs)
        assert any("overlap" in note for note in result.notes)

    def test_missing_seeds_are_reported(self):
        spec = make_spec(seeds=[0, 1, 2, 3, 4])
        runs = runs_for(spec, {"baseline": [0.5, 0.5], "treatment": [0.6, 0.6]})
        result = analyze_runs(spec, runs)
        assert any("no successful run" in note for note in result.notes)


class TestEvaluateDecisionRule:
    def test_clear_win_is_supported(self):
        spec = make_spec()
        runs = runs_for(spec, {"baseline": [0.50, 0.51, 0.50], "treatment": [0.70, 0.71, 0.70]})
        status, detail = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.SUPPORTED
        assert "effect=" in detail

    def test_identical_arms_are_inconclusive_because_the_test_cannot_separate_them(self):
        # A dead-identical result has effect 0 and p=1.0, so the p-value gate
        # fires before the threshold is ever considered. Pinned exactly rather
        # than accepting "refuted or inconclusive": the two mean different
        # things to the replan step, and a test that accepts both cannot tell
        # us which one the rule actually produced.
        spec = make_spec()
        runs = runs_for(spec, {"baseline": [0.50, 0.51, 0.52], "treatment": [0.50, 0.51, 0.52]})
        status, detail = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.INCONCLUSIVE
        assert "p=1.0" in detail

    def test_a_significant_result_that_misses_the_bar_is_refuted(self):
        # The negative-result path: measured cleanly, separated from noise, and
        # smaller than the pre-registered threshold. This is a finding.
        spec = make_spec()
        runs = runs_for(
            spec, {"baseline": [0.500, 0.501, 0.502], "treatment": [0.503, 0.504, 0.505]}
        )
        status, _ = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.REFUTED

    def test_single_seed_cannot_be_supported(self):
        spec = make_spec(seeds=[0])
        runs = runs_for(spec, {"baseline": [0.1], "treatment": [0.9]})
        status, _ = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.INCONCLUSIVE

    def test_lower_is_better_metric_flips_the_sign(self):
        # A loss that went DOWN is an improvement; the rule is written in terms
        # of improvement, so the comparison must be sign-corrected.
        spec = make_spec(
            higher_is_better=False,
            dependent_variables=[MetricDef(name="loss", higher_is_better=False)],
            decision_rule=DecisionRule(metric="loss", comparator=">", threshold=0.01, min_seeds=3),
        )
        runs = runs_for(
            spec, {"baseline": [1.00, 1.01, 1.00], "treatment": [0.50, 0.51, 0.50]}, metric="loss"
        )
        status, _ = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.SUPPORTED

    def test_unmeasured_metric_is_inconclusive_with_a_reason(self):
        spec = make_spec(decision_rule=DecisionRule(metric="f1", comparator=">", threshold=0.1))
        runs = runs_for(spec, {"baseline": [0.5, 0.5, 0.5], "treatment": [0.9, 0.9, 0.9]})
        status, detail = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.INCONCLUSIVE
        assert "never recorded" in detail

    def test_the_strongest_baseline_decides_the_verdict_not_the_first_listed(self):
        # Adding a weak baseline must not manufacture a win, and the order the
        # planner happened to list baselines in must not change the verdict.
        # Against `strong` the treatment gains 0.01, short of the 0.01 threshold
        # it clears easily against `weak`.
        arms = {
            "weak": [0.10, 0.11, 0.10],
            "strong": [0.69, 0.70, 0.69],
            "treatment": [0.70, 0.71, 0.70],
        }
        first_order = make_spec(baselines=["weak", "strong"])
        second_order = make_spec(baselines=["strong", "weak"])

        status_a, detail_a = evaluate_decision_rule(
            first_order, analyze_runs(first_order, runs_for(first_order, arms))
        )
        status_b, detail_b = evaluate_decision_rule(
            second_order, analyze_runs(second_order, runs_for(second_order, arms))
        )

        assert "strong" in detail_a and "strong" in detail_b
        assert status_a is status_b, "baseline ordering must not change the verdict"

    def test_a_lower_is_better_metric_still_picks_the_least_favourable_baseline(self):
        # For a loss, "worst for the treatment" is the smallest reduction, so the
        # selection has to respect the metric's direction rather than raw sign.
        spec = make_spec(
            baselines=["weak", "strong"],
            dependent_variables=[MetricDef(name="loss", higher_is_better=False)],
            decision_rule=DecisionRule(metric="loss", comparator=">", threshold=0.05, min_seeds=3),
        )
        runs = runs_for(
            spec,
            {
                "weak": [2.00, 2.01, 2.00],
                "strong": [0.51, 0.52, 0.51],
                "treatment": [0.50, 0.51, 0.50],
            },
            metric="loss",
        )
        status, detail = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert "strong" in detail
        assert status is VerdictStatus.REFUTED

    def test_without_a_designated_baseline_the_least_favourable_comparison_wins(self):
        """The anti-cherry-picking fallback.

        Reached only when the result's comparisons do not include the spec's
        first baseline -- a Result carried across a spec revision, say, since
        ``analyze_runs`` builds comparisons from ``spec.baselines`` alone.
        Built by hand here for that reason: taking the *better* of two
        comparisons would let an arbitrary extra baseline manufacture a win.
        """
        spec = make_spec(baselines=["designated"])
        result = Result(
            spec_id=spec.id,
            spec_hash=spec.spec_hash,
            n_seeds=3,
            comparisons=[
                _comparison("weak", effect=0.60, p_value=0.001),
                _comparison("strong", effect=0.005, p_value=0.001),
            ],
        )

        status, detail = evaluate_decision_rule(spec, result)

        assert "strong" in detail, "the least favourable baseline must be the one that counts"
        assert status is VerdictStatus.REFUTED


class TestNotes:
    def test_single_seed_is_flagged(self):
        spec = make_spec(seeds=[0])
        result = analyze_runs(spec, runs_for(spec, {"treatment": [0.5]}))
        assert is_single_seed(result)

    def test_variance_note_says_whether_arms_are_distinguishable(self):
        spec = make_spec()
        noisy = analyze_runs(
            spec, runs_for(spec, {"baseline": [0.1, 0.5, 0.9], "treatment": [0.2, 0.6, 1.0]})
        )
        assert "within combined seed spread" in seed_variance_note(noisy, "accuracy")

        clean = analyze_runs(
            spec, runs_for(spec, {"baseline": [0.10, 0.10, 0.11], "treatment": [0.90, 0.90, 0.91]})
        )
        assert "larger than combined seed spread" in seed_variance_note(clean, "accuracy")


class TestACleanNullIsARealNegativeResult:
    """PLAN §1: a negative result must be as legible as a positive one.

    A p-value gate alone cannot deliver that — an effect indistinguishable from
    zero fails significance and lands as INCONCLUSIVE, so a well-run experiment
    that genuinely rules the effect out reads the same as one that was too noisy
    to tell. The equivalence check separates the two.
    """

    def test_a_tight_null_that_excludes_the_threshold_is_refuted(self):
        # Arms are near-identical and very tight: the effect's interval sits far
        # below the 0.05 threshold, so an effect that large is ruled out.
        spec = make_spec(
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.05, min_seeds=3
            )
        )
        # The per-seed differences flip sign, so the effect is not significant
        # (p is large) -- but every difference is tiny, so a 0.05 effect is out.
        runs = runs_for(
            spec,
            {
                "baseline": [0.5000, 0.5000, 0.5000],
                "treatment": [0.5010, 0.4990, 0.5000],
            },
        )
        status, detail = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.REFUTED
        assert "ruled out" in detail

    def test_a_noisy_null_that_cannot_exclude_the_threshold_stays_inconclusive(self):
        # Same near-zero effect, but the spread is wide enough that an effect of
        # 0.05 is still plausible. That is genuinely undecided, not a finding.
        spec = make_spec(
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.05, min_seeds=3
            )
        )
        runs = runs_for(
            spec,
            {
                "baseline": [0.30, 0.50, 0.70],
                "treatment": [0.50, 0.30, 0.70],
            },
        )
        status, _ = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.INCONCLUSIVE

    def test_too_few_seeds_is_still_inconclusive_however_tight(self):
        # The equivalence check must not smuggle a verdict past the seed floor.
        spec = make_spec(
            seeds=[0],
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.05, min_seeds=3
            ),
        )
        runs = runs_for(spec, {"baseline": [0.5000], "treatment": [0.5001]})
        status, _ = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.INCONCLUSIVE

    def test_a_real_win_is_still_supported(self):
        # The equivalence path must not steal verdicts from genuine effects.
        spec = make_spec(
            decision_rule=DecisionRule(
                metric="accuracy", comparator=">", threshold=0.05, min_seeds=3
            )
        )
        runs = runs_for(spec, {"baseline": [0.50, 0.51, 0.50], "treatment": [0.70, 0.71, 0.70]})
        status, _ = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.SUPPORTED

    def test_a_lower_is_better_metric_is_ruled_out_in_the_right_direction(self):
        spec = make_spec(
            dependent_variables=[MetricDef(name="loss", higher_is_better=False)],
            decision_rule=DecisionRule(metric="loss", comparator=">", threshold=0.05, min_seeds=3),
        )
        runs = runs_for(
            spec,
            {"baseline": [1.0000, 1.0000, 1.0000], "treatment": [1.0010, 0.9990, 1.0000]},
            metric="loss",
        )
        status, detail = evaluate_decision_rule(spec, analyze_runs(spec, runs))
        assert status is VerdictStatus.REFUTED
        assert "ruled out" in detail

    def test_the_interval_widens_with_noise(self):
        tight = compare(
            runs_for(
                make_spec(),
                {"baseline": [0.500, 0.501, 0.499], "treatment": [0.502, 0.503, 0.501]},
            ),
            "accuracy",
            baseline_arm="baseline",
            treatment_arm="treatment",
        )
        noisy = compare(
            runs_for(make_spec(), {"baseline": [0.3, 0.5, 0.7], "treatment": [0.31, 0.51, 0.71]}),
            "accuracy",
            baseline_arm="baseline",
            treatment_arm="treatment",
        )
        assert tight is not None and noisy is not None
        tight_interval, noisy_interval = effect_interval(tight), effect_interval(noisy)
        assert tight_interval is not None and noisy_interval is not None
        assert (tight_interval[1] - tight_interval[0]) < (noisy_interval[1] - noisy_interval[0])
