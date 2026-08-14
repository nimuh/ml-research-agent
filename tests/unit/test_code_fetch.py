"""Repo fetching: cloning is not running.

``code/fetch.py`` carries the hardening that keeps a third-party repository from
executing anything on the researcher's machine at fetch time -- disabled hooks,
no submodule recursion, no interactive prompts, a fixed argv and no shell. That
hardening is invisible: delete any one flag and every other test still passes
while a malicious repo's post-checkout hook runs. These tests assert the flags
themselves, so the protection cannot rot silently.

git is never actually invoked: ``subprocess.run`` is replaced with a recorder.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ml_research_agent.code.fetch import cache_path, clear_repo_cache, fetch_repo
from ml_research_agent.config import Config
from ml_research_agent.errors import SourceError
from ml_research_agent.types import CodeRepo


@pytest.fixture
def config(tmp_path):
    return Config(paths={"workspace": tmp_path / "ws"}).ensure_paths()


@pytest.fixture
def repo():
    return CodeRepo(url="https://github.com/o/r", name="r")


class Recorder:
    """Stands in for ``subprocess.run``; records argv and env, runs nothing."""

    def __init__(self, *, returncode: int = 0) -> None:
        self.calls: list[dict[str, object]] = []
        self.returncode = returncode

    def __call__(self, command, **kwargs):
        self.calls.append({"command": list(command), "kwargs": kwargs})
        # A real clone creates the target; mimic that so _finalize can read it.
        if "clone" in command:
            Path(command[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, self.returncode, stdout="", stderr="boom")

    @property
    def clone(self) -> dict[str, object]:
        return next(c for c in self.calls if "clone" in c["command"])

    def flat(self, call: dict[str, object]) -> str:
        return " ".join(str(part) for part in call["command"])


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr("ml_research_agent.code.fetch.subprocess.run", rec)
    return rec


def test_the_clone_disables_every_hook_that_would_execute_repo_code(config, repo, recorder):
    fetch_repo(repo, config)
    argv = recorder.flat(recorder.clone)

    assert "core.hooksPath=/dev/null" in argv, "repo-supplied git hooks would run"
    assert "protocol.ext.allow=never" in argv, "ext:: URLs execute arbitrary commands"
    assert "submodule.recurse=false" in argv
    assert "--no-recurse-submodules" in argv


def test_the_clone_cannot_prompt_or_read_ambient_git_config(config, repo, recorder):
    fetch_repo(repo, config)
    env = recorder.clone["kwargs"]["env"]

    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_ALLOW_PROTOCOL"] == "https"
    # A scrubbed PATH, not the caller's: the environment is built, not inherited.
    assert set(env) >= {"PATH"}
    assert "LD_PRELOAD" not in env


def test_no_git_call_ever_goes_through_a_shell(config, repo, recorder):
    fetch_repo(repo.model_copy(update={"commit": "abc123"}), config)

    assert recorder.calls
    for call in recorder.calls:
        assert call["kwargs"].get("shell") in (None, False)
        assert isinstance(call["command"], list), "a string command would be shell-parsed"
        assert call["command"][0] == "git"
        assert call["kwargs"].get("timeout")


def test_the_clone_is_shallow_by_configuration(config, repo, recorder):
    fetch_repo(repo, config.with_overrides(**{"code.clone_depth": 1}))
    argv = recorder.flat(recorder.clone)
    assert "--depth 1" in argv
    assert "--no-tags" in argv


def test_a_pinned_commit_is_checked_out_and_nothing_else_runs(config, repo, recorder):
    fetch_repo(repo.model_copy(update={"commit": "abc123"}), config)

    verbs = [c["command"][c["command"].index("git") + 1 :] for c in recorder.calls]
    used = {v for call in verbs for v in call if v in {"clone", "fetch", "checkout", "rev-parse"}}
    assert used <= {"clone", "fetch", "checkout", "rev-parse"}
    assert "checkout" in used
    # nothing from the repo itself -- no build, no install, no setup.py
    for call in recorder.calls:
        argv = recorder.flat(call)
        assert not any(word in argv for word in ("make", "pip", "python", "setup.py", "npm", "sh"))


def test_a_failed_clone_raises_and_leaves_no_half_repo(config, repo, monkeypatch):
    rec = Recorder(returncode=1)
    monkeypatch.setattr("ml_research_agent.code.fetch.subprocess.run", rec)

    with pytest.raises(SourceError, match="git clone failed"):
        fetch_repo(repo, config)
    assert not cache_path(config, repo).exists()


def test_offline_refuses_to_clone_at_all(config, repo, recorder):
    with pytest.raises(SourceError, match="offline"):
        fetch_repo(repo, config.with_overrides(offline=True))
    assert recorder.calls == []


def test_a_cache_hit_invokes_git_only_to_read_the_commit(config, repo, recorder):
    cache_path(config, repo).mkdir(parents=True)

    fetched = fetch_repo(repo, config)

    assert not any("clone" in c["command"] for c in recorder.calls)
    assert fetched.local_path == str(cache_path(config, repo))


def test_the_cache_path_is_pinned_by_url_and_commit(config, repo):
    head = cache_path(config, repo)
    pinned = cache_path(config, repo.model_copy(update={"commit": "abc123"}))
    other = cache_path(config, repo.model_copy(update={"url": "https://github.com/o/other"}))

    assert head != pinned, "a pinned commit must not overwrite the HEAD clone"
    assert head != other
    assert head.parent == Path(config.paths.repos)


def test_fetch_records_where_the_bytes_came_from(config, repo, recorder):
    fetched = fetch_repo(repo, config)

    assert fetched.code_hash
    assert fetched.provenance
    assert fetched.provenance[-1].source.startswith("file:")
    assert fetched.provenance[-1].locator == fetched.code_hash


def test_clearing_the_cache_removes_the_clones(config, repo, recorder):
    fetch_repo(repo, config)
    assert cache_path(config, repo).exists()

    assert clear_repo_cache(config) == 1
    assert not cache_path(config, repo).exists()
    assert clear_repo_cache(config) == 0
