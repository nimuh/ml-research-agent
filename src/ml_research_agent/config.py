"""Typed configuration (pydantic-settings) loaded from YAML + env + CLI.

Sections: llm, literature, knowledge, code, experiments, sandbox, budget,
paths, observability. Precedence: CLI flag > env (MRA_*) > configs/*.yaml >
defaults. Config objects are immutable and passed explicitly (no globals).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError
from .types import ModelTier, Scale
from .utils.io import ensure_dir, read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
LOCAL_CONFIG = CONFIG_DIR / "local.yaml"


class _Section(BaseModel):
    """Immutable config section. Frozen so nothing mutates config at runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=False)


class PathsConfig(_Section):
    workspace: Path = Path("./workspace")
    kb_raw: Path = Path("./workspace/kb/raw")
    kb_wiki: Path = Path("./workspace/kb/wiki")
    kb_reports: Path = Path("./workspace/kb/reports")
    runs: Path = Path("./workspace/runs")
    repos: Path = Path("./workspace/repos")
    cache: Path = Path("./workspace/cache")

    _DERIVED: ClassVar[dict[str, tuple[str, ...]]] = {
        "kb_raw": ("kb", "raw"),
        "kb_wiki": ("kb", "wiki"),
        "kb_reports": ("kb", "reports"),
        "runs": ("runs",),
        "repos": ("repos",),
        "cache": ("cache",),
    }

    @model_validator(mode="after")
    def _derive_from_workspace(self) -> Self:
        """Re-root the sub-paths whenever ``workspace`` is overridden.

        Otherwise moving the workspace silently leaves runs, repos and the KB
        writing to the default location -- which in practice means a test or a
        second project quietly scribbling into the real one.
        """
        if "workspace" not in self.model_fields_set:
            return self
        for field, parts in self._DERIVED.items():
            if field not in self.model_fields_set:
                object.__setattr__(self, field, self.workspace.joinpath(*parts))
        return self

    @property
    def kb_sqlite(self) -> Path:
        return self.workspace / "kb" / "kb.sqlite"

    @property
    def kb_index(self) -> Path:
        return self.workspace / "kb" / "index"

    @property
    def state_dir(self) -> Path:
        return self.workspace / "state"

    @property
    def logs(self) -> Path:
        return self.workspace / "logs"

    def ensure(self) -> Self:
        """Create every directory the system writes to. Safe to call repeatedly."""
        for path in (
            self.workspace,
            self.kb_raw,
            self.kb_wiki,
            self.kb_reports,
            self.runs,
            self.repos,
            self.cache,
            self.kb_index,
            self.state_dir,
            self.logs,
        ):
            ensure_dir(path)
        return self


class TierConfig(_Section):
    fast: str = "claude-haiku-4-5-20251001"
    standard: str = "claude-sonnet-5"
    deep: str = "claude-opus-5"

    def model_for(self, tier: ModelTier | str) -> str:
        key = tier.value if isinstance(tier, ModelTier) else str(tier)
        model = getattr(self, key, None)
        if not model:
            raise ConfigError("unknown model tier", tier=key)
        return str(model)


class LLMConfig(_Section):
    provider: Literal["anthropic"] = "anthropic"
    tiers: TierConfig = Field(default_factory=TierConfig)
    max_concurrency: int = Field(default=8, ge=1)
    cache_responses: bool = True
    max_tokens: int = Field(default=8192, ge=256)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_retries: int = Field(default=3, ge=0)
    repair_attempts: int = Field(default=2, ge=0, description="Structured-output repair rounds.")
    timeout_seconds: float = Field(default=180.0, gt=0)
    api_key_env: str = "ANTHROPIC_API_KEY"


class BudgetConfig(_Section):
    usd_per_project: float = Field(default=50.0, gt=0)
    tokens_per_phase: int = Field(default=2_000_000, gt=0)
    max_wallclock_hours: float = Field(default=12.0, gt=0)
    usd_per_phase: float | None = None


class LiteratureConfig(_Section):
    sources: list[str] = Field(
        default_factory=lambda: [
            "arxiv",
            "semantic_scholar",
            "openreview",
            "paperswithcode",
            "github",
            "web",
        ]
    )
    max_candidates: int = Field(default=300, ge=1)
    max_kb_papers: int = Field(default=60, ge=1)
    snowball_depth: int = Field(default=2, ge=0)
    snowball_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    date_window_years: int = Field(default=6, ge=1)
    per_source_limit: int = Field(default=50, ge=1)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    rate_limit_per_second: float = Field(default=2.0, gt=0)
    cache_ttl_seconds: float = Field(default=7 * 24 * 3600.0, gt=0)
    user_agent: str = "ml-research-agent/0.1 (research use)"
    seminal_citation_floor: int = Field(
        default=500, description="Citation count that exempts a paper from the date window."
    )


class RetrievalConfig(_Section):
    top_k: int = Field(default=20, ge=1)
    rerank_top_n: int = Field(default=8, ge=1)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)


class KnowledgeConfig(_Section):
    embedding_model: str = "text-embedding-3-large"
    embedding_backend: Literal["hashing", "openai", "anthropic"] = "hashing"
    embedding_dim: int = Field(default=768, ge=32)
    chunk_tokens: int = Field(default=800, ge=64)
    chunk_overlap: int = Field(default=120, ge=0)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    require_citations: bool = True

    @model_validator(mode="after")
    def _overlap_fits(self) -> Self:
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk_overlap must be smaller than chunk_tokens")
        return self


class CodeConfig(_Section):
    max_repos_per_paper: int = Field(default=2, ge=0)
    clone_depth: int = Field(default=1, ge=1)
    allowed_licenses: list[str] = Field(
        default_factory=lambda: ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "MPL-2.0"]
    )
    run_untrusted_code: bool = False
    max_repo_mb: int = Field(default=500, ge=1)
    max_files_to_analyze: int = Field(default=400, ge=1)


class ExperimentsConfig(_Section):
    default_seeds: int = Field(default=3, ge=1)
    max_parallel_runs: int = Field(default=2, ge=1)
    max_runtime_minutes: int = Field(default=120, ge=1)
    require_preregistered_decision_rule: bool = True
    scale_ladder: list[Scale] = Field(
        default_factory=lambda: [Scale.SMOKE, Scale.SMALL, Scale.MAIN]
    )
    max_retries_per_run: int = Field(default=2, ge=0)
    max_inconclusive_rounds: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _ladder_starts_cheap(self) -> Self:
        if not self.scale_ladder or self.scale_ladder[0] is not Scale.SMOKE:
            raise ValueError("the scale ladder must start at 'smoke' -- cheap before expensive")
        return self


class SandboxConfig(_Section):
    backend: Literal["subprocess", "docker", "remote"] = "subprocess"
    network: Literal["deny", "allow"] = "deny"
    network_allowlist: list[str] = Field(
        default_factory=lambda: [
            "pypi.org",
            "files.pythonhosted.org",
            "huggingface.co",
            "raw.githubusercontent.com",
        ]
    )
    filesystem_scope: Literal["workspace", "none"] = "workspace"
    gpu: Literal["auto", "none", "required"] = "auto"
    max_memory_mb: int | None = None
    kill_after_seconds: int = Field(default=7200, ge=1)
    denied_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "shutdown", "reboot", "mkfs", "dd if=", ":(){:|:&};:"]
    )


class GatesConfig(_Section):
    after_survey: bool = True
    before_run: bool = True
    before_report: bool = False
    auto_approve: bool = Field(default=False, description="Non-interactive runs approve gates.")
    timeout_seconds: float | None = None


class ObservabilityConfig(_Section):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    trace: bool = True
    cost_ledger: bool = True
    console: bool = True
    redact_keys: list[str] = Field(
        default_factory=lambda: ["api_key", "authorization", "token", "secret", "password"]
    )


class Config(BaseSettings):
    """The whole configuration tree, immutable and passed explicitly.

    Precedence, lowest to highest: field defaults, ``configs/default.yaml``,
    ``configs/local.yaml``, ``MRA_*`` env vars, CLI overrides. Nested keys use a
    double underscore in env: ``MRA_BUDGET__USD_PER_PROJECT=50``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MRA_",
        env_nested_delimiter="__",
        frozen=True,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    experiments: ExperimentsConfig = Field(default_factory=ExperimentsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    gates: GatesConfig = Field(default_factory=GatesConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    offline: bool = Field(default=False, description="Never touch the network; caches only.")
    dry_run: bool = Field(
        default=False, description="Plan and price the work without executing it."
    )

    @classmethod
    def load(
        cls,
        *,
        config_file: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
        use_local: bool = True,
    ) -> Config:
        """Build a Config by layering YAML, env and explicit overrides."""
        data: dict[str, Any] = {}
        if DEFAULT_CONFIG.exists():
            data = _deep_merge(data, read_yaml(DEFAULT_CONFIG, default={}) or {})
        if use_local and LOCAL_CONFIG.exists():
            data = _deep_merge(data, read_yaml(LOCAL_CONFIG, default={}) or {})
        if config_file:
            path = Path(config_file)
            if not path.exists():
                raise ConfigError("config file not found", path=str(path))
            data = _deep_merge(data, read_yaml(path, default={}) or {})
        if overrides:
            data = _deep_merge(data, _expand_dotted(overrides))
        try:
            # BaseSettings applies MRA_* env on top of these initialization values.
            return cls(**data)
        except Exception as exc:  # pragma: no cover - re-raised with context
            raise ConfigError(f"invalid configuration: {exc}") from exc

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.llm.api_key_env)

    def require_api_key(self) -> str:
        key = self.api_key
        if not key:
            raise ConfigError(
                "missing API key; copy .env.example to .env and set it",
                env_var=self.llm.api_key_env,
            )
        return key

    def model_for(self, tier: ModelTier | str) -> str:
        return self.llm.tiers.model_for(tier)

    def with_overrides(self, **overrides: Any) -> Config:
        """Return a new Config; the original is never mutated."""
        merged = _deep_merge(self.model_dump(mode="python"), _expand_dotted(overrides))
        return type(self)(**merged)

    def ensure_paths(self) -> Config:
        self.paths.ensure()
        return self


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge where ``overlay`` wins on conflicts."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_dotted(values: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{"budget.usd_per_project": 5}`` into nested dicts (CLI flag form)."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None:
            continue
        parts = key.split(".")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


__all__ = [
    "CONFIG_DIR",
    "DEFAULT_CONFIG",
    "REPO_ROOT",
    "BudgetConfig",
    "CodeConfig",
    "Config",
    "ExperimentsConfig",
    "GatesConfig",
    "KnowledgeConfig",
    "LLMConfig",
    "LiteratureConfig",
    "ObservabilityConfig",
    "PathsConfig",
    "RetrievalConfig",
    "SandboxConfig",
    "TierConfig",
]
