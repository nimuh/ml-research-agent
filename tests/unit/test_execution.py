"""Failure signatures, retry policy, workspaces and run tracking."""

from __future__ import annotations

import json

import pytest

from ml_research_agent.config import Config
from ml_research_agent.errors import IrreproducibleRun, SandboxViolation
from ml_research_agent.experiments.execute import (
    RETRY_ADJUSTMENTS,
    classify_failure,
    detect_silent_noop,
    execute_run,
    smoke_passed,
)
from ml_research_agent.experiments.sandbox import SandboxResult
from ml_research_agent.experiments.tracking import (
    RunStore,
    collect_artifacts,
    parse_metrics_file,
    parse_metrics_from_log,
)
from ml_research_agent.experiments.workspace import ExperimentWorkspace
from ml_research_agent.types import (
    DatasetRef,
    DecisionRule,
    ExperimentSpec,
    FailureSignature,
    Metric,
    MetricDef,
    RunRecord,
    RunStatus,
)


@pytest.fixture
def config(tmp_path):
    return Config(paths={"workspace": tmp_path / "ws"}).ensure_paths()


def make_spec(**overrides) -> ExperimentSpec:
    defaults = dict(
        hypothesis_id="hyp",
        title="t",
        dependent_variables=[MetricDef(name="accuracy")],
        dataset=DatasetRef(name="toy"),
        baselines=["baseline"],
        treatment="treatment",
        seeds=[0, 1, 2],
        max_runtime_minutes=1,
        decision_rule=DecisionRule(metric="accuracy", comparator=">", threshold=0.01),
    )
    return ExperimentSpec(**{**defaults, **overrides})


def result(
    stdout: str = "",
    stderr: str = "",
    code: int = 0,
    timed_out: bool = False,
    duration: float = 5.0,
):
    return SandboxResult(
        command="cmd",
        exit_code=code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        timed_out=timed_out,
    )


class TestFailureClassification:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB", FailureSignature.OOM),
            ("loss is nan at step 40", FailureSignature.NAN),
            ("ModuleNotFoundError: No module named 'torch'", FailureSignature.DEPENDENCY),
            ("FileNotFoundError: data/train.jsonl", FailureSignature.DATA_MISSING),
            ("loss diverged after 100 steps", FailureSignature.DIVERGENCE),
        ],
    )
    def test_signatures_are_named_before_anything_is_retried(self, text, expected):
        signature, detail = classify_failure(result(stderr=text, code=1))
        assert signature is expected
        assert detail

    def test_a_timeout_is_its_own_signature(self):
        signature, _ = classify_failure(result(timed_out=True, duration=3600))
        assert signature is FailureSignature.TIMEOUT

    def test_an_uncategorized_nonzero_exit_is_still_recorded(self):
        signature, detail = classify_failure(result(stderr="something odd", code=7))
        assert signature is FailureSignature.NONZERO_EXIT
        assert "something odd" in detail

    def test_a_clean_run_has_no_signature(self):
        assert classify_failure(result(stdout="accuracy 0.9"))[0] is None

    def test_an_instant_success_with_no_metrics_is_a_silent_noop(self):
        # Exits 0, produces nothing, and would otherwise contribute an empty arm
        # to the aggregate.
        assert detect_silent_noop(result(duration=0.1), metrics_found=0)
        assert not detect_silent_noop(result(duration=0.1), metrics_found=3)
        assert not detect_silent_noop(result(duration=120), metrics_found=0)

    def test_oom_retry_actually_changes_something(self):
        # Retrying an OOM at the same batch size is not a strategy.
        assert RETRY_ADJUSTMENTS[FailureSignature.OOM]["batch_size_factor"] < 1


class TestMetricParsing:
    def test_a_flat_mapping_is_parsed(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"accuracy": 0.91, "loss": 0.3, "note": "ignored"}))
        metrics = parse_metrics_file(path)
        assert {m.name for m in metrics} == {"accuracy", "loss"}

    def test_a_step_series_keeps_its_steps(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps([{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0}]))
        metrics = parse_metrics_file(path)
        assert [m.step for m in metrics] == [1, 2]

    def test_a_nested_metrics_key_is_unwrapped(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"metrics": {"accuracy": 0.5}, "config": {"lr": 0.1}}))
        assert [m.name for m in parse_metrics_file(path)] == ["accuracy"]

    def test_a_missing_or_corrupt_file_yields_nothing_rather_than_raising(self, tmp_path):
        assert parse_metrics_file(tmp_path / "absent.json") == []
        broken = tmp_path / "broken.json"
        broken.write_text("{not json")
        assert parse_metrics_file(broken) == []

    def test_log_parsing_is_restricted_to_known_metric_names(self):
        log = "step: 10 | accuracy: 0.82 | lr: 0.001 | wall_time: 12.4"
        metrics = parse_metrics_from_log(log, known=["accuracy"])
        assert [m.name for m in metrics] == ["accuracy"]
        assert metrics[0].step == 10

    def test_unrestricted_log_parsing_is_available_but_opt_in(self):
        metrics = parse_metrics_from_log("accuracy: 0.82 | lr: 0.001")
        assert {m.name for m in metrics} == {"accuracy", "lr"}


class TestWorkspace:
    def test_files_are_written_under_the_workspace(self, config):
        workspace = ExperimentWorkspace.create(config, make_spec())
        workspace.write_files({"train.py": "print('hi')", "pkg/util.py": "x = 1"})
        assert set(workspace.list_files()) == {"train.py", "pkg/util.py"}

    def test_a_path_escaping_the_workspace_is_refused(self, config):
        workspace = ExperimentWorkspace.create(config, make_spec())
        with pytest.raises(SandboxViolation):
            workspace.write_files({"../../evil.py": "x"})

    def test_the_manifest_alone_describes_the_run(self, config):
        spec = make_spec()
        workspace = ExperimentWorkspace.create(config, spec)
        workspace.write_files({"train.py": "print('hi')"})
        workspace.set_commands(entrypoint="python train.py", smoke="python train.py --smoke")
        workspace.write_manifest(divergences=["fewer steps"])

        manifest = workspace.manifest()
        assert manifest["spec_hash"] == spec.spec_hash
        assert manifest["code_hash"] and manifest["env_hash"]
        assert manifest["entrypoint"] and manifest["files"] == ["train.py"]
        assert manifest["divergences"] == ["fewer steps"]

    def test_code_hash_tracks_content(self, config):
        workspace = ExperimentWorkspace.create(config, make_spec())
        workspace.write_files({"train.py": "a = 1"})
        before = workspace.code_hash
        workspace.write_files({"train.py": "a = 2"})
        assert workspace.code_hash != before

    def test_each_arm_and_seed_gets_its_own_output_directory(self, config):
        workspace = ExperimentWorkspace.create(config, make_spec())
        first = workspace.run_dir(arm="baseline", seed=0)
        second = workspace.run_dir(arm="treatment", seed=0)
        assert first != second

    def test_command_templates_receive_the_run_identity(self, config):
        workspace = ExperimentWorkspace.create(config, make_spec())
        workspace.set_commands(entrypoint="python t.py --arm {arm} --seed {seed}", smoke="s")
        command = workspace.command_for(arm="treatment", seed=2)
        assert "--arm treatment" in command and "--seed 2" in command

    def test_a_template_without_placeholders_still_gets_the_identity(self, config):
        workspace = ExperimentWorkspace.create(config, make_spec())
        workspace.set_commands(entrypoint="python t.py", smoke="s")
        command = workspace.command_for(arm="baseline", seed=1)
        assert "--arm baseline" in command and "--seed 1" in command


class TestRunStore:
    def _run(self, spec, **overrides) -> RunRecord:
        defaults = dict(
            spec_id=spec.id,
            spec_hash=spec.spec_hash,
            code_hash="abc",
            env_hash="def",
            seed=0,
            arm="treatment",
            status=RunStatus.COMPLETED,
            metrics=[Metric(name="accuracy", value=0.9)],
        )
        return RunRecord(**{**defaults, **overrides})

    def test_runs_round_trip(self, config):
        spec = make_spec()
        store = RunStore(config)
        run = store.save(self._run(spec))
        assert store.for_spec(spec.spec_hash)[0].id == run.id

    def test_lookup_by_the_reproducibility_tuple(self, config):
        spec = make_spec()
        store = RunStore(config)
        store.save(self._run(spec))
        found = store.find(
            spec_hash=spec.spec_hash, code_hash="abc", env_hash="def", seed=0, arm="treatment"
        )
        assert found is not None

    def test_different_arms_at_the_same_seed_do_not_collide(self, config):
        # Without the arm in the key, baseline and treatment would look like the
        # same run disagreeing with itself.
        spec = make_spec()
        store = RunStore(config)
        store.save(self._run(spec, arm="baseline", metrics=[Metric(name="accuracy", value=0.5)]))
        store.save(self._run(spec, arm="treatment", metrics=[Metric(name="accuracy", value=0.9)]))

        baseline = store.find(
            spec_hash=spec.spec_hash, code_hash="abc", env_hash="def", seed=0, arm="baseline"
        )
        treatment = store.find(
            spec_hash=spec.spec_hash, code_hash="abc", env_hash="def", seed=0, arm="treatment"
        )
        assert baseline is not None and treatment is not None
        assert baseline.metric("accuracy") == 0.5
        assert treatment.metric("accuracy") == 0.9

    def test_lookup_discriminates_on_every_part_of_the_tuple(self, config):
        # A find() that ignored code_hash or env_hash would match a run made
        # before the code changed -- and then check_reproducible would compare
        # two different experiments and call the difference a contract breach.
        spec = make_spec()
        store = RunStore(config)
        store.save(self._run(spec))

        for changed in (
            {"code_hash": "different"},
            {"env_hash": "different"},
            {"seed": 1},
            {"spec_hash": "different"},
        ):
            key = {
                "spec_hash": spec.spec_hash,
                "code_hash": "abc",
                "env_hash": "def",
                "seed": 0,
                "arm": "treatment",
                **changed,
            }
            assert store.find(**key) is None, f"find() ignored {sorted(changed)[0]}"

    def test_the_reproducibility_contract_is_enforced(self, config):
        spec = make_spec()
        store = RunStore(config)
        store.save(self._run(spec))
        divergent = self._run(spec, metrics=[Metric(name="accuracy", value=0.42)])
        with pytest.raises(IrreproducibleRun):
            store.check_reproducible(divergent)

    def test_identical_metrics_satisfy_the_contract(self, config):
        spec = make_spec()
        store = RunStore(config)
        store.save(self._run(spec))
        store.check_reproducible(self._run(spec))  # must not raise

    def test_artifacts_are_indexed_and_hashed(self, config, tmp_path):
        out = tmp_path / "artifacts"
        out.mkdir()
        (out / "metrics.json").write_text("{}")
        (out / "train.log").write_text("hello")
        artifacts = collect_artifacts(out)
        assert {a.kind for a in artifacts} == {"metrics", "log"}
        assert all(a.content_hash for a in artifacts)


class TestRealExecution:
    def test_a_real_run_records_metrics_and_hashes(self, config):
        spec = make_spec()
        workspace = ExperimentWorkspace.create(config, spec)
        workspace.write_files(
            {
                "train.py": (
                    "import argparse, json, os\n"
                    "p = argparse.ArgumentParser()\n"
                    "p.add_argument('--arm')\n"
                    "p.add_argument('--seed', type=int)\n"
                    "p.add_argument('--out')\n"
                    "a, _ = p.parse_known_args()\n"
                    "os.makedirs(a.out, exist_ok=True)\n"
                    "path = os.path.join(a.out, 'metrics.json')\n"
                    "json.dump({'accuracy': 0.5 + a.seed * 0.1}, open(path, 'w'))\n"
                    "print('done')\n"
                )
            }
        )
        workspace.set_commands(
            entrypoint="python3 train.py --arm {arm} --seed {seed} --out {out}", smoke=""
        )
        workspace.write_manifest()

        record = execute_run(spec, workspace, arm="treatment", seed=1, config=config)
        assert record.succeeded, record.failure_detail
        assert record.metric("accuracy") == pytest.approx(0.6)
        assert record.code_hash and record.log_path
        assert smoke_passed([record])

    def test_a_failing_run_is_categorized_not_crashed(self, config):
        spec = make_spec()
        workspace = ExperimentWorkspace.create(config, spec)
        workspace.write_files({"train.py": "raise MemoryError('CUDA out of memory')\n"})
        workspace.set_commands(entrypoint="python3 train.py --arm {arm} --seed {seed}", smoke="")
        workspace.write_manifest()

        record = execute_run(spec, workspace, arm="treatment", seed=0, config=config)
        assert not record.succeeded
        assert record.failure is FailureSignature.OOM
        assert not smoke_passed([record])


class TestDryRunDoesNotExecute:
    """`dry_run` promises "plan and price the work without executing it"."""

    def _workspace(self, config, spec):
        workspace = ExperimentWorkspace.create(config, spec)
        workspace.write_files({"train.py": "import pathlib; pathlib.Path('RAN').write_text('x')\n"})
        workspace.set_commands(entrypoint="python3 train.py --arm {arm} --seed {seed}", smoke="")
        workspace.write_manifest()
        return workspace

    def test_a_dry_run_records_the_command_without_running_it(self, tmp_path):
        config = Config(paths={"workspace": tmp_path / "ws"}, dry_run=True).ensure_paths()
        spec = make_spec()
        workspace = self._workspace(config, spec)

        record = execute_run(spec, workspace, arm="treatment", seed=0, config=config)

        assert record.status is RunStatus.SKIPPED
        assert "would have executed" in record.stdout_tail
        assert not list(workspace.path.rglob("RAN")), "the experiment must not have run"

    def test_without_dry_run_the_same_spec_executes(self, tmp_path):
        config = Config(paths={"workspace": tmp_path / "ws"}, dry_run=False).ensure_paths()
        spec = make_spec()
        workspace = self._workspace(config, spec)

        record = execute_run(spec, workspace, arm="treatment", seed=0, config=config)

        assert record.status is not RunStatus.SKIPPED
        assert list(workspace.path.rglob("RAN")), "the control case must actually run"
