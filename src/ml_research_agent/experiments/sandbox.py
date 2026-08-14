"""Execution isolation: subprocess/venv, container, or remote backend, with
network policy, filesystem scoping, wall-clock/GPU/token ceilings, and kill switch."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config, SandboxConfig
from ..errors import SandboxViolation
from ..observability.logging import StructuredLogger, get_logger
from ..utils.io import safe_join

# Denied outright, in any backend. This is a backstop, not the security model --
# the real boundary is the backend (container/remote) plus the workspace scope.
HARD_DENY = ("rm -rf /", "mkfs", ":(){", "shutdown", "reboot", "dd if=/dev/zero of=/dev")

# Env vars that would leak credentials into untrusted code.
STRIPPED_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "S2_API_KEY",
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_API_KEY",
)

# Belt and braces: anything that *looks* like a credential goes too, so adding a
# provider to .env does not silently hand its key to third-party repo code.
_SECRET_MARKERS = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")

#: A closed port. Cooperative HTTP stacks routed here fail fast.
_BLACKHOLE_PROXY = "http://127.0.0.1:9"


@dataclass
class SandboxResult:
    """What a sandboxed command did, with output already truncated."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    killed: bool = False
    workspace: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


@dataclass
class Sandbox:
    """Runs commands under a policy. Third-party code executes nowhere else.

    The base class implements the ``subprocess`` backend; ``docker`` and
    ``remote`` are selected by config. Fetching a repo never runs anything --
    cloning is not running -- so every path into repo code comes through here.
    """

    config: SandboxConfig
    workspace: Path
    logger: StructuredLogger = field(default_factory=lambda: get_logger("sandbox"))
    max_output_chars: int = 200_000
    allow_untrusted: bool = False
    """Whether third-party repo code may execute here at all (``code.run_untrusted_code``)."""

    @classmethod
    def for_config(cls, config: Config, workspace: Path, **kwargs: object) -> Sandbox:
        kwargs.setdefault("allow_untrusted", config.code.run_untrusted_code)
        backend = config.sandbox.backend
        if backend == "docker":
            return DockerSandbox(config.sandbox, workspace, **kwargs)  # type: ignore[arg-type]
        if backend == "remote":
            return RemoteSandbox(config.sandbox, workspace, **kwargs)  # type: ignore[arg-type]
        return cls(config.sandbox, workspace, **kwargs)  # type: ignore[arg-type]

    # -- policy --------------------------------------------------------------

    def check_command(self, command: str) -> None:
        lowered = command.lower()
        for denied in (*HARD_DENY, *self.config.denied_commands):
            if denied.lower() in lowered:
                raise SandboxViolation("denied command", command=command, matched=denied)

    def resolve_path(self, relative: str) -> Path:
        """Any path a run touches is resolved under the workspace or refused."""
        if self.config.filesystem_scope == "none":
            return Path(relative)
        return safe_join(self.workspace, relative)

    def build_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """A minimal environment with credentials stripped.

        Untrusted code with the project's API key in its environment is a
        credential leak waiting for one `env` call.
        """
        env = {k: v for k, v in os.environ.items() if not _is_secret(k)}
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "MRA_SANDBOX": self.config.backend,
                "HOME": str(self.workspace),
                "TMPDIR": str(self.workspace / "tmp"),
            }
        )
        if self.config.network == "deny":
            # Route everything at a closed port, then exempt exactly the
            # allow-listed hosts via no_proxy. Setting no_proxy to "*" would
            # exempt *every* host and make the blackhole proxy unreachable --
            # denial that quietly permits everything.
            allowed = ",".join(self.config.network_allowlist)
            env["no_proxy"] = env["NO_PROXY"] = allowed
            env["HTTP_PROXY"] = env["HTTPS_PROXY"] = _BLACKHOLE_PROXY
            env["http_proxy"] = env["https_proxy"] = _BLACKHOLE_PROXY
        if self.config.gpu == "none":
            env["CUDA_VISIBLE_DEVICES"] = ""
        env.update(extra or {})
        return env

    # -- execution -----------------------------------------------------------

    def run(
        self,
        command: str,
        *,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        untrusted: bool = False,
    ) -> SandboxResult:
        """Execute a command under the sandbox policy.

        ``untrusted=True`` marks the command as running third-party repository
        code, which ``code.run_untrusted_code`` gates. Cloning is not running:
        this is the only door that code comes through.
        """
        if untrusted and not self.allow_untrusted:
            raise SandboxViolation(
                "third-party repository code may not execute; set code.run_untrusted_code "
                "to enable it deliberately",
                command=command[:200],
            )
        self.check_command(command)
        limit = min(timeout or self.config.kill_after_seconds, self.config.kill_after_seconds)
        working = self.resolve_path(cwd) if cwd else self.workspace
        working.mkdir(parents=True, exist_ok=True)
        (self.workspace / "tmp").mkdir(parents=True, exist_ok=True)

        self.warn_if_policy_is_advisory()
        self.logger.info(
            "sandbox_exec", command=command[:400], backend=self.config.backend, timeout=limit
        )
        started = time.monotonic()
        process = subprocess.Popen(  # noqa: S603 - command is policy-checked above
            self.wrap_command(command),
            cwd=str(working),
            env=self.build_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill(process)
            stdout, stderr = process.communicate()

        return SandboxResult(
            command=command,
            exit_code=process.returncode if process.returncode is not None else -1,
            stdout=self._truncate(stdout),
            stderr=self._truncate(stderr),
            duration_seconds=time.monotonic() - started,
            timed_out=timed_out,
            workspace=str(self.workspace),
        )

    def wrap_command(self, command: str) -> list[str]:
        return ["/bin/sh", "-c", command]

    @property
    def enforces_network_policy(self) -> bool:
        """Whether ``network: deny`` is actually enforced, not merely signalled.

        The subprocess backend can only set proxy variables, which cooperative
        HTTP stacks honour and a raw socket ignores. Only the container backend
        enforces denial at the network layer.
        """
        return False

    def warn_if_policy_is_advisory(self) -> None:
        if self.config.network == "deny" and not self.enforces_network_policy:
            self.logger.warning(
                "network_denial_is_advisory",
                backend=self.config.backend,
                detail=(
                    "network: deny is signalled via proxy env vars on this backend; "
                    "a raw socket can still reach the network. Use backend: docker "
                    "to enforce it."
                ),
            )

    def _kill(self, process: subprocess.Popen[str]) -> None:
        """The kill switch: terminate the whole process group, then insist."""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(2)
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):  # pragma: no cover
            process.kill()

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        half = self.max_output_chars // 2
        dropped = len(text) - self.max_output_chars
        return f"{text[:half]}\n\n[... {dropped} characters truncated ...]\n\n{text[-half:]}"


@dataclass
class DockerSandbox(Sandbox):
    """Container backend: the only one where network denial is real."""

    image: str = "python:3.11-slim"

    @property
    def enforces_network_policy(self) -> bool:
        return True

    def wrap_command(self, command: str) -> list[str]:
        args = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none" if self.config.network == "deny" else "bridge",
            "-v",
            f"{self.workspace}:/workspace",
            "-w",
            "/workspace",
            "--pids-limit",
            "512",
        ]
        if self.config.max_memory_mb:
            args += ["--memory", f"{self.config.max_memory_mb}m"]
        if self.config.gpu in ("auto", "required"):
            args += ["--gpus", "all"]
        return [*args, self.image, "/bin/sh", "-c", command]


@dataclass
class RemoteSandbox(Sandbox):
    """Remote backend placeholder.

    Deliberately raises rather than silently falling back to local execution:
    a remote policy that quietly becomes a local one is exactly the failure the
    sandbox exists to prevent.
    """

    def run(
        self,
        command: str,
        *,
        timeout: int | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        untrusted: bool = False,
    ) -> SandboxResult:
        raise SandboxViolation(
            "the remote sandbox backend is configured but not provisioned; "
            "refusing to fall back to local execution",
            command=shlex.quote(command)[:200],
        )


def _is_secret(name: str) -> bool:
    """Whether an environment variable must not reach sandboxed code."""
    upper = name.upper()
    return upper in STRIPPED_ENV or any(marker in upper for marker in _SECRET_MARKERS)


__all__ = [
    "HARD_DENY",
    "STRIPPED_ENV",
    "DockerSandbox",
    "RemoteSandbox",
    "Sandbox",
    "SandboxResult",
]
