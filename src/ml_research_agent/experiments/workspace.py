"""Per-experiment workspace: isolated dir, pinned env, code snapshot, config,
data links, and a manifest that makes the run reproducible from the manifest alone."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config
from ..errors import MRAError
from ..types import EnvSpec, ExperimentSpec, utcnow
from ..utils.hashing import hash_dir
from ..utils.io import ensure_dir, read_json, safe_join, write_json

MANIFEST = "manifest.json"


@dataclass
class ExperimentWorkspace:
    """One spec's isolated directory, plus the manifest that makes it rebuildable.

    The manifest is the contract: it must be sufficient to reconstruct the run
    without the session that created it, which is why it carries the spec, the
    env, the commands and the hashes rather than pointers into project state.
    """

    path: Path
    spec: ExperimentSpec
    entrypoint: str = ""
    smoke: str = ""
    metrics_path: str = "metrics.json"
    env: EnvSpec = field(default_factory=EnvSpec)

    @classmethod
    def create(cls, config: Config, spec: ExperimentSpec) -> ExperimentWorkspace:
        root = ensure_dir(Path(config.paths.runs) / spec.spec_hash[:12])
        for sub in ("code", "artifacts", "logs", "data", "tmp"):
            ensure_dir(root / sub)
        workspace = cls(path=root, spec=spec, env=spec.env)
        existing = read_json(root / MANIFEST)
        if existing:
            workspace.entrypoint = existing.get("entrypoint", "")
            workspace.smoke = existing.get("smoke", "")
            workspace.metrics_path = existing.get("metrics_path", "metrics.json")
        return workspace

    @classmethod
    def load(cls, path: Path, spec: ExperimentSpec) -> ExperimentWorkspace:
        data = read_json(Path(path) / MANIFEST) or {}
        return cls(
            path=Path(path),
            spec=spec,
            entrypoint=data.get("entrypoint", ""),
            smoke=data.get("smoke", ""),
            metrics_path=data.get("metrics_path", "metrics.json"),
            env=EnvSpec.model_validate(data.get("env", {})) if data.get("env") else spec.env,
        )

    # -- contents ------------------------------------------------------------

    @property
    def code_dir(self) -> Path:
        return self.path / "code"

    @property
    def artifacts_dir(self) -> Path:
        return self.path / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    def list_files(self) -> list[str]:
        if not self.code_dir.exists():
            return []
        return sorted(
            p.relative_to(self.code_dir).as_posix() for p in self.code_dir.rglob("*") if p.is_file()
        )

    def write_files(self, files: dict[str, str]) -> list[Path]:
        """Write generated code, refusing any path that escapes the workspace."""
        written: list[Path] = []
        for relative, content in files.items():
            target = safe_join(self.code_dir, relative)
            ensure_dir(target.parent)
            target.write_text(content, encoding="utf-8")
            written.append(target)
        return written

    def read_file(self, relative: str) -> str:
        return safe_join(self.code_dir, relative).read_text(encoding="utf-8")

    def set_commands(self, *, entrypoint: str, smoke: str, metrics_path: str | None = None) -> None:
        self.entrypoint = entrypoint
        self.smoke = smoke
        if metrics_path:
            self.metrics_path = metrics_path

    # -- reproducibility -----------------------------------------------------

    @property
    def code_hash(self) -> str:
        return hash_dir(self.code_dir) if self.code_dir.exists() else ""

    @property
    def env_hash(self) -> str:
        return self.env.env_hash

    def run_dir(self, *, arm: str, seed: int) -> Path:
        """Per-(arm, seed) output directory, so runs never overwrite each other."""
        return ensure_dir(self.artifacts_dir / f"{arm}-seed{seed}")

    def write_manifest(self, **extra: Any) -> Path:
        manifest = {
            "created_at": utcnow().isoformat(),
            "spec_hash": self.spec.spec_hash,
            "spec": self.spec.model_dump(mode="json"),
            "code_hash": self.code_hash,
            "env_hash": self.env_hash,
            "env": self.env.model_dump(mode="json"),
            "entrypoint": self.entrypoint,
            "smoke": self.smoke,
            "metrics_path": self.metrics_path,
            "files": self.list_files(),
            **extra,
        }
        return write_json(self.path / MANIFEST, manifest)

    def manifest(self) -> dict[str, Any]:
        return read_json(self.path / MANIFEST) or {}

    def command_for(self, *, arm: str, seed: int, smoke: bool = False) -> str:
        """Substitute the run's identity into the recorded command template.

        Templates use ``{arm}``/``{seed}``/``{out}``; a command that ignores them
        would silently run the same configuration every time, which is the
        "silent no-op" failure signature.
        """
        template = (self.smoke or self.entrypoint) if smoke else (self.entrypoint or self.smoke)
        if not template.strip():
            raise MRAError(
                "workspace has no runnable command; IMPLEMENT did not set one",
                workspace=str(self.path),
                spec=self.spec.id,
            )
        out = self.run_dir(arm=arm, seed=seed)
        if "{arm}" in template or "{seed}" in template or "{out}" in template:
            return template.format(arm=arm, seed=seed, out=out, metrics=out / self.metrics_path)
        return f"{template} --arm {arm} --seed {seed} --out {out}"


__all__ = ["MANIFEST", "ExperimentWorkspace"]
