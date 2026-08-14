"""Shared fixtures. No test in this suite may touch the network or need a key."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_research_agent.config import Config, PathsConfig


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def config(workspace: Path) -> Config:
    """A Config whose every path lives under the test's tmp_path."""
    return Config(
        paths=PathsConfig(
            workspace=workspace,
            kb_raw=workspace / "kb" / "raw",
            kb_wiki=workspace / "kb" / "wiki",
            kb_reports=workspace / "kb" / "reports",
            runs=workspace / "runs",
            repos=workspace / "repos",
            cache=workspace / "cache",
        ),
        offline=False,
    )
