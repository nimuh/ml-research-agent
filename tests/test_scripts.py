#!/usr/bin/env python3
"""Tests for the deterministic core.

Only four scripts survived the conversion to skills, and they survived because
they do the things a model does unreliably: hashing, seed statistics, applying a
threshold, and counting links across a folder. Those need tests for exactly the
same reason they need to be scripts.

Stdlib `unittest`, no dependencies, no network:

    python3 tests/test_scripts.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILLS = ROOT / "skills"


def load(path: Path):
    """Import a script by path. The scripts live under skills/, not on sys.path.

    Registered in ``sys.modules`` before execution because ``@dataclass`` looks
    its own module up by name while the class body is being processed, and a
    module that is not there yet fails with an error naming neither.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


spec_hash = load(SKILLS / "design-experiment" / "scripts" / "spec_hash.py")
stats = load(SKILLS / "analyze-results" / "scripts" / "stats.py")
decide = load(SKILLS / "analyze-results" / "scripts" / "decide.py")
lint = load(SKILLS / "lint-vault" / "scripts" / "lint_vault.py")


SPEC = """---
type: spec
id: exp-001
spec_hash: ""
title: "Curriculum vs random ordering"
status: preregistered
hypothesis: "Curriculum ordering improves accuracy at matched compute"
arms: [random, curriculum]
baseline: random
dataset: {name: toy, split: test}
metrics: [accuracy]
controls: ["matched compute", "same seeds"]
seeds: [0, 1, 2]
scale: smoke
decision_rule:
  metric: accuracy
  comparator: ">"
  threshold: 0.02
  min_seeds: 3
  max_p_value: 0.05
  refutes_if: "The 95% CI sits entirely below +0.02"
added: 2026-08-14
tags: [spec]
---

# Curriculum vs random ordering

## Decision rule

```yaml
metric: accuracy
comparator: ">"
threshold: 0.02
min_seeds: 3
max_p_value: 0.05
```

## Design

Body prose that the hash must ignore.
"""


def runs(pairs, status="ok"):
    return [
        stats.Run(arm=arm, seed=seed, status=status, metrics={"accuracy": value})
        for arm, seed, value in pairs
    ]


# ---------------------------------------------------------------------------
# spec_hash.py — the pre-registration witness
# ---------------------------------------------------------------------------


class TestSpecHash(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.spec = self.dir / "spec.md"
        self.spec.write_text(SPEC, encoding="utf-8")

    def test_the_hash_is_stable_across_runs(self) -> None:
        self.assertEqual(spec_hash.compute(self.spec), spec_hash.compute(self.spec))

    def test_editing_the_decision_rule_changes_the_hash(self) -> None:
        """The whole point: a rule adjusted after the numbers is detectable."""
        before = spec_hash.compute(self.spec)
        self.spec.write_text(SPEC.replace("threshold: 0.02", "threshold: 0.005"), encoding="utf-8")

        self.assertNotEqual(before, spec_hash.compute(self.spec))

    def test_editing_the_prose_does_not(self) -> None:
        """A spec whose every edit looks like tampering trains everyone to ignore it."""
        before = spec_hash.compute(self.spec)
        self.spec.write_text(
            SPEC.replace("Body prose that the hash must ignore.", "Rewritten explanation."),
            encoding="utf-8",
        )

        self.assertEqual(before, spec_hash.compute(self.spec))

    def test_status_moving_forward_does_not_break_the_hash(self) -> None:
        before = spec_hash.compute(self.spec)
        self.spec.write_text(
            SPEC.replace("status: preregistered", "status: complete"), encoding="utf-8"
        )

        self.assertEqual(before, spec_hash.compute(self.spec))

    def test_reflowing_a_line_is_not_a_design_change(self) -> None:
        before = spec_hash.compute(self.spec)
        self.spec.write_text(
            SPEC.replace('controls: ["matched compute", "same seeds"]',
                         'controls:  ["matched compute",   "same seeds"]'),
            encoding="utf-8",
        )

        self.assertEqual(before, spec_hash.compute(self.spec))

    def test_write_then_verify_round_trips(self) -> None:
        spec_hash.write(self.spec, spec_hash.compute(self.spec))

        self.assertEqual(spec_hash.recorded(self.spec), spec_hash.compute(self.spec))

    def test_verify_fails_after_a_post_hoc_edit(self) -> None:
        script = SKILLS / "design-experiment" / "scripts" / "spec_hash.py"
        subprocess.run([sys.executable, str(script), str(self.spec), "--write"], check=True,
                       capture_output=True)
        text = self.spec.read_text(encoding="utf-8")
        self.spec.write_text(text.replace("threshold: 0.02", "threshold: 0.001"), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(script), str(self.spec), "--verify"], capture_output=True, text=True
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("PRE-REGISTRATION CHANGED", result.stderr)

    def test_the_linter_hashes_a_spec_byte_for_byte_the_same_way(self) -> None:
        """The two scripts duplicate this payload so each skill stands alone.

        The whole payload is compared, not the decision rule alone. An earlier
        version of this test checked only the rule and passed while the two
        parsers disagreed about every other field -- so `lint-vault` reported
        every honest spec as tampered, which is the worst possible failure for
        a check whose entire job is to be believed.
        """
        text = self.spec.read_text(encoding="utf-8")
        raw, body = spec_hash.split_frontmatter(text)

        self.assertEqual(
            spec_hash.payload(spec_hash.parse_frontmatter(raw), body),
            lint.spec_payload(lint.spec_frontmatter(raw), body),
        )

    def test_the_linter_accepts_a_spec_it_just_hashed(self) -> None:
        """The end-to-end version of the check above, through both CLIs."""
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / "experiments" / "exp-001").mkdir(parents=True)
            spec = vault / "experiments" / "exp-001" / "spec.md"
            spec.write_text(SPEC, encoding="utf-8")
            spec_hash.write(spec, spec_hash.compute(spec))

            checks = {f.check for f in lint.run_checks(vault)}

        self.assertNotIn("tampered-preregistration", checks)
        self.assertNotIn("no-spec-hash", checks)


# ---------------------------------------------------------------------------
# stats.py — the arithmetic
# ---------------------------------------------------------------------------


class TestStats(unittest.TestCase):
    def test_a_clear_effect_across_three_seeds(self) -> None:
        report = stats.analyse(
            runs([
                ("random", 0, 0.60), ("random", 1, 0.61), ("random", 2, 0.59),
                ("curriculum", 0, 0.70), ("curriculum", 1, 0.71), ("curriculum", 2, 0.69),
            ]),
            "accuracy",
            "random",
        )

        comparison = report["comparisons"][0]
        self.assertEqual(comparison["n_pairs"], 3)
        self.assertAlmostEqual(comparison["mean_difference"], 0.10, places=6)
        self.assertLess(comparison["p_value"], 0.001)
        self.assertGreater(comparison["ci_low"], 0.0)

    def test_noise_is_not_an_effect(self) -> None:
        """The case the whole script exists for: means that differ, and nothing there."""
        report = stats.analyse(
            runs([
                ("random", 0, 0.60), ("random", 1, 0.70), ("random", 2, 0.50),
                ("curriculum", 0, 0.71), ("curriculum", 1, 0.49), ("curriculum", 2, 0.62),
            ]),
            "accuracy",
            "random",
        )

        comparison = report["comparisons"][0]
        self.assertGreater(comparison["p_value"], 0.05)
        self.assertLess(comparison["ci_low"], 0.0)
        self.assertGreater(comparison["ci_high"], 0.0)

    def test_the_t_distribution_matches_known_values(self) -> None:
        """Checked against published tables — the implementation is home-grown."""
        self.assertAlmostEqual(stats.t_critical(2), 4.302653, places=4)
        self.assertAlmostEqual(stats.t_critical(9), 2.262157, places=4)
        self.assertAlmostEqual(stats.t_critical(29), 2.045230, places=4)
        self.assertAlmostEqual(stats.t_two_sided_p(2.262157, 9), 0.05, places=5)

    def test_sample_standard_deviation_not_population(self) -> None:
        """At n=3 the difference is not cosmetic."""
        summary = stats.summarise(
            runs([("a", 0, 1.0), ("a", 1, 2.0), ("a", 2, 3.0)]), "accuracy"
        )[0]

        self.assertAlmostEqual(summary.sd, 1.0, places=9)

    def test_failed_runs_are_excluded_from_the_mean_but_counted(self) -> None:
        mixed = runs([("a", 0, 0.7), ("a", 1, 0.7)]) + runs([("a", 2, 0.0)], status="oom")
        report = stats.analyse(mixed, "accuracy", "a")

        self.assertEqual(report["arms"][0]["n"], 2)
        self.assertEqual(report["failures"], {"oom": 1})
        self.assertEqual(report["runs_ok"], 2)

    def test_pairing_only_uses_seeds_both_arms_completed(self) -> None:
        report = stats.analyse(
            runs([
                ("random", 0, 0.60), ("random", 1, 0.61),
                ("curriculum", 0, 0.70), ("curriculum", 1, 0.71), ("curriculum", 2, 0.69),
            ]),
            "accuracy",
            "random",
        )

        self.assertEqual(report["comparisons"][0]["n_pairs"], 2)
        self.assertEqual(report["comparisons"][0]["seeds"], [0, 1])

    def test_a_single_pair_reports_no_interval_rather_than_a_narrow_one(self) -> None:
        """An interval of zero width would read as certainty about the least
        certain thing in the file."""
        report = stats.analyse(
            runs([("random", 0, 0.60), ("curriculum", 0, 0.70)]), "accuracy", "random"
        )

        comparison = report["comparisons"][0]
        self.assertEqual(comparison["n_pairs"], 1)
        self.assertNotEqual(comparison["p_value"], comparison["p_value"])  # NaN

    def test_run_records_are_read_off_markdown_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "curriculum-seed0.md").write_text(
                "---\ntype: run\narm: curriculum\nseed: 0\nstatus: ok\n"
                "metrics: {accuracy: 0.7031, loss: 1.22}\n---\n\n# run\n",
                encoding="utf-8",
            )
            (directory / "random-seed0.md").write_text(
                "---\ntype: run\narm: random\nseed: 0\nstatus: oom\nmetrics: {}\n---\n",
                encoding="utf-8",
            )

            loaded = stats.load_markdown_runs(directory)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].metrics["accuracy"], 0.7031)
        self.assertEqual(loaded[1].status, "oom")


# ---------------------------------------------------------------------------
# decide.py — the pre-registered rule, applied
# ---------------------------------------------------------------------------


class TestDecide(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.spec = self.dir / "spec.md"
        self.spec.write_text(SPEC, encoding="utf-8")
        self.rule = decide.parse_rule(self.spec)

    def analyse(self, pairs):
        return stats.analyse(runs(pairs), "accuracy", "random")

    def test_the_rule_is_read_off_the_spec(self) -> None:
        self.assertEqual(self.rule["threshold"], 0.02)
        self.assertEqual(self.rule["min_seeds"], 3)
        self.assertEqual(self.rule["comparator"], ">")

    def test_a_large_consistent_effect_is_supported(self) -> None:
        report = self.analyse([
            ("random", 0, 0.60), ("random", 1, 0.61), ("random", 2, 0.59),
            ("curriculum", 0, 0.70), ("curriculum", 1, 0.71), ("curriculum", 2, 0.69),
        ])

        self.assertEqual(decide.decide(self.rule, report)["verdict"], decide.SUPPORTED)

    def test_an_effect_ruled_out_is_refuted_not_inconclusive(self) -> None:
        """The clause that makes a negative result legible.

        The difference is tight around zero, so the whole interval sits below
        the 0.02 threshold: the experiment settled the question in the negative.
        A significance test alone would call this inconclusive, which reads
        identically to an experiment too noisy to say anything.
        """
        report = self.analyse([
            ("random", 0, 0.600), ("random", 1, 0.610), ("random", 2, 0.590),
            ("curriculum", 0, 0.601), ("curriculum", 1, 0.611), ("curriculum", 2, 0.591),
        ])

        result = decide.decide(self.rule, report)

        self.assertEqual(result["verdict"], decide.REFUTED)
        self.assertIn("ruled out", result["reason"])

    def test_a_noisy_null_is_inconclusive_not_refuted(self) -> None:
        report = self.analyse([
            ("random", 0, 0.60), ("random", 1, 0.70), ("random", 2, 0.50),
            ("curriculum", 0, 0.71), ("curriculum", 1, 0.49), ("curriculum", 2, 0.62),
        ])

        self.assertEqual(decide.decide(self.rule, report)["verdict"], decide.INCONCLUSIVE)

    def test_too_few_seeds_is_inconclusive_however_large_the_gap(self) -> None:
        """The seed floor is checked first and is not negotiable."""
        report = self.analyse([("random", 0, 0.10), ("curriculum", 0, 0.99)])

        result = decide.decide(self.rule, report)

        self.assertEqual(result["verdict"], decide.INCONCLUSIVE)
        self.assertIn("required by the pre-registered rule", result["reason"])

    def test_the_seed_floor_beats_the_equivalence_check_too(self) -> None:
        """Otherwise a sample of two could smuggle a `refuted` past the floor."""
        report = self.analyse([
            ("random", 0, 0.600), ("random", 1, 0.610),
            ("curriculum", 0, 0.601), ("curriculum", 1, 0.611),
        ])

        self.assertEqual(decide.decide(self.rule, report)["verdict"], decide.INCONCLUSIVE)

    def test_significance_below_the_threshold_is_not_a_win(self) -> None:
        """p < 0.05 on an effect smaller than the pre-registered one is a refutation."""
        report = self.analyse([
            ("random", 0, 0.6000), ("random", 1, 0.6100), ("random", 2, 0.5900),
            ("curriculum", 0, 0.6050), ("curriculum", 1, 0.6150), ("curriculum", 2, 0.5950),
        ])

        result = decide.decide(self.rule, report)

        self.assertLess(result["checks"]["significance"]["p_value"], 0.05)
        self.assertEqual(result["verdict"], decide.REFUTED)

    def test_it_judges_against_the_baseline_the_treatment_does_worst_against(self) -> None:
        """Otherwise adding a weak baseline manufactures a win."""
        report = stats.analyse(
            runs([
                ("random", 0, 0.60), ("random", 1, 0.61), ("random", 2, 0.59),
                ("tuned", 0, 0.699), ("tuned", 1, 0.709), ("tuned", 2, 0.689),
                ("curriculum", 0, 0.70), ("curriculum", 1, 0.71), ("curriculum", 2, 0.69),
            ]),
            "accuracy",
            "random",
        )

        # `curriculum` beats `random` by 0.10; `tuned` beats it by 0.099. The
        # verdict must be decided on `tuned`, the one it barely clears.
        chosen = decide.worst_comparison(report, None)

        self.assertEqual(chosen["arm"], "tuned")


# ---------------------------------------------------------------------------
# lint_vault.py — the structural checks
# ---------------------------------------------------------------------------


def write_vault(root: Path) -> None:
    (root / "sources").mkdir(parents=True)
    (root / "concepts").mkdir()
    (root / "raw").mkdir()
    (root / "_meta").mkdir()

    (root / "raw" / "arxiv-1234.5678.md").write_text(
        "---\ntype: dossier\nkey: arxiv:1234.5678\nread: full\ntags: [dossier]\n---\n\n"
        "> Mamba reaches 1.02 bits-per-byte at 8k context.\n",
        encoding="utf-8",
    )
    (root / "sources" / "arxiv-1234.5678.md").write_text(
        "---\ntype: source\nkey: arxiv:1234.5678\ntitle: A Paper\nread: full\n"
        "raw: raw/arxiv-1234.5678.md\nadded: 2026-08-14\ntags: [source]\n---\n\n"
        "# A Paper\n\n## Results\n\nReaches 1.02 bits-per-byte, per [[Mamba]].\n\n"
        "## Evidence\n\n**E1**\n> Mamba reaches 1.02 bits-per-byte at 8k context.\n"
        "— §4.2, p.7\n",
        encoding="utf-8",
    )
    (root / "concepts" / "Mamba.md").write_text(
        "---\ntype: method\ntitle: Mamba\nadded: 2026-08-14\ntags: [method]\n---\n\n"
        "# Mamba\n\n## Seen in\n\n- [[arxiv-1234.5678]]\n",
        encoding="utf-8",
    )


class TestLintVault(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp()) / "vault"
        self.dir.mkdir()
        write_vault(self.dir)

    def checks(self, findings):
        return {f.check for f in findings}

    def test_a_well_formed_vault_is_clean(self) -> None:
        """A linter that fires on a correct vault gets run once and then ignored."""
        self.assertEqual(lint.run_checks(self.dir), [])

    def test_a_broken_wikilink_is_reported(self) -> None:
        note = self.dir / "concepts" / "Mamba.md"
        note.write_text(note.read_text() + "\nSee also [[Nonexistent Note]].\n", encoding="utf-8")

        self.assertIn("broken-link", self.checks(lint.run_checks(self.dir)))

    def test_a_number_with_no_quote_behind_it_is_a_blocker(self) -> None:
        """The defect that matters most in a vault: a figure nothing supports."""
        note = self.dir / "sources" / "arxiv-1234.5678.md"
        note.write_text(
            note.read_text().replace("Reaches 1.02", "Reaches 99.9% accuracy and 1.02"),
            encoding="utf-8",
        )

        findings = lint.run_checks(self.dir)

        self.assertIn("unsupported-number", self.checks(findings))
        self.assertTrue(any(f.severity == "blocker" for f in findings))

    def test_a_source_note_with_no_evidence_section_is_a_blocker(self) -> None:
        note = self.dir / "sources" / "arxiv-1234.5678.md"
        note.write_text(note.read_text().split("## Evidence")[0], encoding="utf-8")

        self.assertIn("no-evidence", self.checks(lint.run_checks(self.dir)))

    def test_a_note_may_not_claim_to_have_read_more_than_its_dossier(self) -> None:
        dossier = self.dir / "raw" / "arxiv-1234.5678.md"
        dossier.write_text(dossier.read_text().replace("read: full", "read: abstract-only"),
                           encoding="utf-8")

        findings = lint.run_checks(self.dir)

        self.assertIn("upgraded-reading", self.checks(findings))

    def test_a_dossier_that_was_neither_noted_nor_excluded_is_reported(self) -> None:
        (self.dir / "raw" / "arxiv-9999.0001.md").write_text(
            "---\ntype: dossier\nkey: arxiv:9999.0001\nread: full\ntags: [dossier]\n---\n",
            encoding="utf-8",
        )

        self.assertIn("unscreened-dossier", self.checks(lint.run_checks(self.dir)))

    def test_a_claim_settled_while_still_contradicted_is_reported(self) -> None:
        (self.dir / "concepts" / "Claim.md").write_text(
            "---\ntype: claim\ntitle: A Claim\nstatus: settled\n"
            'supported_by: ["[[arxiv-1234.5678]]"]\n'
            'contradicted_by: ["[[arxiv-1234.5678]]"]\n'
            "added: 2026-08-14\ntags: [claim]\n---\n\n# A Claim\n\n[[Mamba]]\n",
            encoding="utf-8",
        )

        self.assertIn("resolved-contest", self.checks(lint.run_checks(self.dir)))

    def test_a_spec_edited_after_registration_is_a_blocker(self) -> None:
        """The check the whole linter exists for."""
        experiments = self.dir / "experiments" / "exp-001"
        experiments.mkdir(parents=True)
        spec = experiments / "spec.md"
        spec.write_text(SPEC, encoding="utf-8")
        spec_hash.write(spec, spec_hash.compute(spec))

        clean = lint.run_checks(self.dir)
        spec.write_text(
            spec.read_text().replace("threshold: 0.02", "threshold: 0.001"), encoding="utf-8"
        )
        tampered = lint.run_checks(self.dir)

        self.assertNotIn("tampered-preregistration", self.checks(clean))
        self.assertIn("tampered-preregistration", self.checks(tampered))

    def test_a_spec_with_no_hash_at_all_is_a_blocker(self) -> None:
        experiments = self.dir / "experiments" / "exp-001"
        experiments.mkdir(parents=True)
        (experiments / "spec.md").write_text(
            SPEC.replace('spec_hash: ""\n', ""), encoding="utf-8"
        )

        self.assertIn("no-spec-hash", self.checks(lint.run_checks(self.dir)))

    def test_the_cli_exits_non_zero_on_a_blocker(self) -> None:
        note = self.dir / "sources" / "arxiv-1234.5678.md"
        note.write_text(note.read_text().split("## Evidence")[0], encoding="utf-8")
        script = SKILLS / "lint-vault" / "scripts" / "lint_vault.py"

        result = subprocess.run(
            [sys.executable, str(script), str(self.dir), "--json"],
            capture_output=True, text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertTrue(any(f["severity"] == "blocker" for f in json.loads(result.stdout)))

    def test_write_todo_produces_a_vault_note(self) -> None:
        script = SKILLS / "lint-vault" / "scripts" / "lint_vault.py"
        subprocess.run(
            [sys.executable, str(script), str(self.dir), "--write-todo", "--fail-on", "never"],
            check=True, capture_output=True,
        )

        todo = (self.dir / "TODO.md").read_text(encoding="utf-8")

        self.assertTrue(todo.startswith("---"), "TODO.md must be a vault note")
        self.assertIn("Nothing outstanding", todo)


# ---------------------------------------------------------------------------
# The skills themselves
# ---------------------------------------------------------------------------


class TestSkills(unittest.TestCase):
    """A malformed SKILL.md is not discovered, and nothing says so at runtime."""

    def skill_dirs(self):
        return sorted(p for p in SKILLS.iterdir() if p.is_dir())

    def test_every_skill_has_a_skill_file_with_front_matter(self) -> None:
        for path in self.skill_dirs():
            with self.subTest(skill=path.name):
                skill = path / "SKILL.md"
                self.assertTrue(skill.is_file(), f"{path.name} has no SKILL.md")
                frontmatter, body = lint.parse_frontmatter(skill.read_text(encoding="utf-8"))
                self.assertEqual(frontmatter.get("name"), path.name,
                                 "name: must match the directory name")
                self.assertTrue(frontmatter.get("description"), "description: is required")
                self.assertTrue(body.strip(), "a skill with no instructions does nothing")

    def test_descriptions_say_when_to_use_the_skill(self) -> None:
        """A description that only says what the skill *is* never gets triggered.

        Claude matches a description against how the user actually phrased the
        request, so each one has to carry a trigger clause naming the situation
        and the words someone would use for it -- not just a summary of the
        instructions inside.
        """
        for path in self.skill_dirs():
            with self.subTest(skill=path.name):
                frontmatter, _ = lint.parse_frontmatter(
                    (path / "SKILL.md").read_text(encoding="utf-8")
                )
                description = frontmatter["description"].lower()
                self.assertGreater(len(description), 80,
                                   "too terse to match a user's phrasing")
                self.assertRegex(
                    description,
                    r"use (when|after|before|to|for)\b",
                    "no trigger clause: say when to reach for this, in the words "
                    "a user would use",
                )

    def test_every_referenced_script_exists(self) -> None:
        for path in self.skill_dirs():
            body = (path / "SKILL.md").read_text(encoding="utf-8")
            for line in body.splitlines():
                if "scripts/" not in line or "python3" not in line:
                    continue
                fragment = next(t for t in line.split() if "scripts/" in t)
                target = (path / fragment).resolve()
                with self.subTest(skill=path.name, script=fragment):
                    self.assertTrue(target.is_file(), f"{fragment} does not exist")

    def test_every_script_compiles_and_answers_help(self) -> None:
        for script in sorted(SKILLS.rglob("scripts/*.py")):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    [sys.executable, str(script), "--help"], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
