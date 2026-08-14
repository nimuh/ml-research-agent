"""Run tracking: RunRecord (spec hash, code hash, env hash, seed, metrics,
artifacts, cost, status). Local-first store with optional W&B/MLflow export."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from ..config import Config
from ..errors import IrreproducibleRun
from ..types import Artifact, Metric, RunRecord
from ..utils.hashing import hash_file
from ..utils.io import append_jsonl, ensure_dir, read_jsonl, write_json

# Loose but explicit: "step 120 | loss 0.53 | acc 0.81". Runs that write JSON are
# parsed properly; this is the fallback for logs, and it is deliberately not
# clever -- a regex that guesses too much invents metrics that never existed.
_LOG_METRIC = re.compile(r"\b([a-zA-Z][\w/\.]{1,40})\s*[=:]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_STEP = re.compile(r"\b(?:step|iter|iteration|epoch)\s*[=:]?\s*(\d+)", re.IGNORECASE)


class RunStore:
    """Append-only local store of run records under ``workspace/runs``."""

    def __init__(self, config: Config) -> None:
        self.root = ensure_dir(Path(config.paths.runs))
        self.index_path = self.root / "runs.jsonl"

    def save(self, run: RunRecord) -> RunRecord:
        append_jsonl(self.index_path, run.model_dump(mode="json"))
        write_json(self.root / run.spec_hash[:12] / f"{run.id}.json", run.model_dump(mode="json"))
        return run

    def all(self) -> list[RunRecord]:
        records: list[RunRecord] = []
        for row in read_jsonl(self.index_path):
            try:
                records.append(RunRecord.model_validate(row))
            except Exception:
                continue
        # Later writes win: a retried run supersedes its earlier record.
        latest: dict[str, RunRecord] = {r.id: r for r in records}
        return list(latest.values())

    def for_spec(self, spec_hash: str) -> list[RunRecord]:
        return [r for r in self.all() if r.spec_hash == spec_hash]

    def find(
        self, *, spec_hash: str, code_hash: str, env_hash: str, seed: int, arm: str = "treatment"
    ) -> RunRecord | None:
        """Look up by the reproducibility tuple."""
        key = (spec_hash, code_hash, env_hash, arm, seed)
        return next((r for r in self.all() if r.repro_key == key), None)

    def check_reproducible(self, run: RunRecord, *, tolerance: float = 1e-9) -> None:
        """Same tuple must give the same metrics, or the contract is broken.

        Raising here rather than warning is deliberate: a system that quietly
        tolerates unreproducible runs cannot support any claim it makes.
        """
        previous = self.find(
            spec_hash=run.spec_hash,
            code_hash=run.code_hash,
            env_hash=run.env_hash,
            seed=run.seed,
            arm=run.arm,
        )
        if previous is None or previous.id == run.id or not previous.succeeded or not run.succeeded:
            return
        for metric in run.metrics:
            before = previous.metric(metric.name)
            if before is None:
                continue
            if abs(before - metric.value) > tolerance:
                raise IrreproducibleRun(
                    "identical (spec, code, env, seed) produced different metrics",
                    metric=metric.name,
                    before=before,
                    after=metric.value,
                    run=run.id,
                    previous=previous.id,
                )


def parse_metrics_file(path: Path, *, arm: str | None = None) -> list[Metric]:
    """Parse a run's ``metrics.json``.

    Accepts a flat mapping, a list of step records, or ``{"metrics": {...}}`` --
    the three shapes generated code actually tends to write.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    metrics: list[Metric] = []
    if isinstance(data, dict) and isinstance(data.get("metrics"), dict | list):
        data = data["metrics"]
    if isinstance(data, dict):
        for name, value in data.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                metrics.append(Metric(name=str(name), value=float(value), arm=arm))
    elif isinstance(data, list):
        for record in data:
            if not isinstance(record, dict):
                continue
            step = record.get("step") or record.get("epoch") or record.get("iteration")
            for name, value in record.items():
                if name in ("step", "epoch", "iteration"):
                    continue
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics.append(
                        Metric(
                            name=str(name),
                            value=float(value),
                            step=int(step) if isinstance(step, int | float) else None,
                            arm=arm,
                        )
                    )
    return metrics


def parse_metrics_from_log(
    text: str, *, arm: str | None = None, known: Iterable[str] = ()
) -> list[Metric]:
    """Best-effort metric extraction from stdout, restricted to known names.

    Restricted on purpose: an unfiltered regex over training logs will happily
    report `lr` and `time` as experimental results.
    """
    wanted = {k.lower() for k in known}
    metrics: list[Metric] = []
    step: int | None = None
    for line in text.splitlines():
        found_step = _STEP.search(line)
        if found_step:
            step = int(found_step.group(1))
        for name, value in _LOG_METRIC.findall(line):
            if wanted and name.lower() not in wanted:
                continue
            try:
                metrics.append(Metric(name=name, value=float(value), step=step, arm=arm))
            except ValueError:
                continue
    return metrics


def collect_artifacts(directory: Path, *, limit: int = 200) -> list[Artifact]:
    """Index a run's outputs, content-hashing everything small enough to matter."""
    if not directory.exists():
        return []
    artifacts: list[Artifact] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or len(artifacts) >= limit:
            continue
        size = path.stat().st_size
        artifacts.append(
            Artifact(
                name=path.name,
                path=str(path),
                kind=_kind_for(path),
                size_bytes=size,
                content_hash=hash_file(path) if size < 50_000_000 else None,
            )
        )
    return artifacts


def _kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".log", ".txt", ".out"):
        return "log"
    if suffix == ".json" and "metric" in path.name.lower():
        return "metrics"
    if suffix in (".pt", ".ckpt", ".safetensors", ".bin"):
        return "checkpoint"
    if suffix in (".png", ".jpg", ".pdf", ".svg"):
        return "figure"
    if suffix in (".yaml", ".yml", ".toml", ".ini"):
        return "config"
    if suffix in (".csv", ".tsv"):
        return "table"
    return "other"


__all__ = [
    "RunStore",
    "collect_artifacts",
    "parse_metrics_file",
    "parse_metrics_from_log",
]
