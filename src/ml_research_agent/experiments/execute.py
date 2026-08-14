"""Run execution: launch, stream and parse logs, checkpoint, detect failure
signatures (OOM, NaN, hang, divergence), and apply the retry policy."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Config
from ..observability.logging import StructuredLogger, get_logger
from ..types import (
    ExperimentSpec,
    FailureSignature,
    Provenance,
    RunRecord,
    RunStatus,
    Scale,
    utcnow,
)
from ..utils.concurrency import gather_bounded
from .sandbox import Sandbox, SandboxResult
from .tracking import RunStore, collect_artifacts, parse_metrics_file, parse_metrics_from_log
from .workspace import ExperimentWorkspace

# Ordered: the first match wins, so specific signatures precede generic ones.
FAILURE_PATTERNS: tuple[tuple[FailureSignature, re.Pattern[str]], ...] = (
    (
        FailureSignature.OOM,
        re.compile(r"out of memory|CUDA out of memory|OOMKilled|MemoryError", re.I),
    ),
    (FailureSignature.NAN, re.compile(r"\bnan\b|\binf\b|loss is nan|not finite", re.I)),
    (
        FailureSignature.DEPENDENCY,
        re.compile(r"ModuleNotFoundError|ImportError|No module named", re.I),
    ),
    (
        FailureSignature.DATA_MISSING,
        re.compile(r"FileNotFoundError|No such file or directory|dataset not found", re.I),
    ),
    (
        FailureSignature.DIVERGENCE,
        re.compile(r"loss (?:diverged|exploded)|gradient overflow", re.I),
    ),
)

# Retries that change nothing are not retries. Each entry is (parameter, factor).
RETRY_ADJUSTMENTS: dict[FailureSignature, dict[str, Any]] = {
    FailureSignature.OOM: {"batch_size_factor": 0.5, "grad_accum_factor": 2},
    FailureSignature.NAN: {"lr_factor": 0.5},
    FailureSignature.TIMEOUT: {"steps_factor": 0.5},
}


def classify_failure(result: SandboxResult) -> tuple[FailureSignature | None, str]:
    """Name the failure before deciding what to do about it."""
    if result.timed_out:
        return (
            FailureSignature.TIMEOUT,
            f"exceeded the wall-clock ceiling after {result.duration_seconds:.0f}s",
        )
    text = f"{result.stdout}\n{result.stderr}"
    for signature, pattern in FAILURE_PATTERNS:
        match = pattern.search(text)
        if match:
            return signature, _context(text, match.start())
    if result.exit_code != 0:
        return FailureSignature.NONZERO_EXIT, (result.stderr or result.stdout)[-800:]
    return None, ""


def detect_silent_noop(result: SandboxResult, metrics_found: int) -> bool:
    """A run that exits 0 in no time having produced nothing did not run.

    This is the failure mode that quietly poisons an analysis, because it
    reports success and contributes an empty arm.
    """
    return result.ok and metrics_found == 0 and result.duration_seconds < 2.0


def execute_run(
    spec: ExperimentSpec,
    workspace: ExperimentWorkspace,
    *,
    arm: str,
    seed: int,
    config: Config,
    logger: StructuredLogger | None = None,
    attempt: int = 1,
    adjustments: dict[str, Any] | None = None,
) -> RunRecord:
    """Execute one (arm, seed) inside the sandbox and record what happened."""
    log = (logger or get_logger("execute")).bind(spec=spec.id, arm=arm, seed=seed)
    sandbox = Sandbox.for_config(config, workspace.path, logger=log)
    out_dir = workspace.run_dir(arm=arm, seed=seed)
    command = workspace.command_for(arm=arm, seed=seed, smoke=spec.scale is Scale.SMOKE)
    if adjustments:
        command = _apply_adjustments(command, adjustments)

    record = RunRecord(
        spec_id=spec.id,
        spec_hash=spec.spec_hash,
        code_hash=workspace.code_hash,
        env_hash=workspace.env_hash,
        seed=seed,
        arm=arm,
        scale=spec.scale,
        status=RunStatus.RUNNING,
        started_at=utcnow(),
        workspace_path=str(workspace.path),
        provenance=[Provenance(source=f"spec:{spec.id}", locator=f"{arm}/seed={seed}")],
    )

    if config.dry_run:
        # "Plan and price the work without executing it" has to mean it: a
        # dry run that quietly launches the experiment is worse than no flag.
        log.info("dry_run_skipped", command=command[:400])
        return record.model_copy(
            update={
                "status": RunStatus.SKIPPED,
                "finished_at": utcnow(),
                "stdout_tail": f"[dry run] would have executed: {command}",
            }
        )

    result = sandbox.run(command, timeout=spec.max_runtime_minutes * 60, cwd="code")
    log_path = workspace.logs_dir / f"{arm}-seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {command}\n\n{result.stdout}\n\n--- stderr ---\n{result.stderr}", encoding="utf-8"
    )

    metrics = parse_metrics_file(out_dir / workspace.metrics_path, arm=arm)
    if not metrics:
        metrics = parse_metrics_from_log(
            result.stdout, arm=arm, known=[m.name for m in spec.dependent_variables]
        )

    failure, detail = classify_failure(result)
    if failure is None and detect_silent_noop(result, len(metrics)):
        failure, detail = FailureSignature.SILENT_NOOP, "exited 0 instantly and produced no metrics"

    finished = utcnow()
    record = record.model_copy(
        update={
            "status": RunStatus.COMPLETED if failure is None else RunStatus.FAILED,
            "finished_at": finished,
            "duration_seconds": result.duration_seconds,
            "metrics": metrics,
            "artifacts": collect_artifacts(out_dir),
            "log_path": str(log_path),
            "stdout_tail": result.stdout[-4000:],
            "failure": failure,
            "failure_detail": detail or None,
        }
    )
    log.info(
        "run_finished",
        status=record.status.value,
        failure=failure.value if failure else None,
        duration=round(result.duration_seconds, 2),
        metrics=len(metrics),
        attempt=attempt,
    )
    return record


def execute_spec(
    spec: ExperimentSpec,
    config: Config,
    *,
    state: Any = None,
    logger: StructuredLogger | None = None,
    workspace: ExperimentWorkspace | None = None,
) -> list[RunRecord]:
    """Run every (arm, seed), retry categorized failures, and record everything.

    Concurrency is capped by ``max_parallel_runs`` rather than by however many
    seeds happen to exist -- an unbounded fan-out of GPU jobs is a way to spend
    a budget without learning anything.
    """
    log = (logger or get_logger("execute")).bind(spec=spec.id)
    workspace = workspace or ExperimentWorkspace.create(config, spec)
    store = RunStore(config)
    records: list[RunRecord] = []

    for arm in spec.arms or ["treatment"]:
        for seed in spec.seeds:
            record = execute_run(spec, workspace, arm=arm, seed=seed, config=config, logger=log)
            attempt = 1
            while (
                record.failure is not None
                and attempt <= config.experiments.max_retries_per_run
                and record.failure in RETRY_ADJUSTMENTS
            ):
                attempt += 1
                adjustments = RETRY_ADJUSTMENTS[record.failure]
                log.warning(
                    "run_retry",
                    failure=record.failure.value,
                    attempt=attempt,
                    adjustments=adjustments,
                )
                retried = execute_run(
                    spec,
                    workspace,
                    arm=arm,
                    seed=seed,
                    config=config,
                    logger=log,
                    attempt=attempt,
                    adjustments=adjustments,
                )
                record = retried.model_copy(update={"retry_of": record.id})
            store.save(record)
            try:
                store.check_reproducible(record)
            except Exception as exc:  # surfaced, never swallowed
                log.error("reproducibility_violation", error=str(exc))
            records.append(record)

    return records


async def execute_spec_async(
    spec: ExperimentSpec,
    config: Config,
    *,
    logger: StructuredLogger | None = None,
) -> list[RunRecord]:
    """Bounded-parallel variant honoring ``max_parallel_runs``."""
    import asyncio

    workspace = ExperimentWorkspace.create(config, spec)
    pairs = [(arm, seed) for arm in (spec.arms or ["treatment"]) for seed in spec.seeds]

    async def _one(arm: str, seed: int) -> RunRecord:
        return await asyncio.to_thread(
            execute_run, spec, workspace, arm=arm, seed=seed, config=config, logger=logger
        )

    results = await gather_bounded(
        [lambda a=arm, s=seed: _one(a, s) for arm, seed in pairs],  # type: ignore[misc]
        limit=config.experiments.max_parallel_runs,
    )
    store = RunStore(config)
    records = [r for r in results if isinstance(r, RunRecord)]
    for record in records:
        store.save(record)
    return records


def smoke_passed(records: Sequence[RunRecord]) -> bool:
    """The ladder gate: at least one arm produced metrics without failing."""
    return any(r.succeeded and r.metrics for r in records)


def _apply_adjustments(command: str, adjustments: dict[str, Any]) -> str:
    """Append adjustment flags so the retry is visibly different from the original."""
    flags = []
    if "batch_size_factor" in adjustments:
        flags.append(f"--batch-size-factor {adjustments['batch_size_factor']}")
    if "grad_accum_factor" in adjustments:
        flags.append(f"--grad-accum-factor {adjustments['grad_accum_factor']}")
    if "lr_factor" in adjustments:
        flags.append(f"--lr-factor {adjustments['lr_factor']}")
    if "steps_factor" in adjustments:
        flags.append(f"--steps-factor {adjustments['steps_factor']}")
    return f"{command} {' '.join(flags)}".strip()


def _context(text: str, position: int, *, width: int = 400) -> str:
    start = max(0, position - width // 2)
    return text[start : position + width // 2].strip()


def workspace_for(config: Config, spec: ExperimentSpec) -> ExperimentWorkspace:
    return ExperimentWorkspace.create(config, spec)


def logs_for(workspace: ExperimentWorkspace, *, arm: str, seed: int) -> str:
    path = Path(workspace.logs_dir) / f"{arm}-seed{seed}.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


__all__ = [
    "FAILURE_PATTERNS",
    "RETRY_ADJUSTMENTS",
    "classify_failure",
    "detect_silent_noop",
    "execute_run",
    "execute_spec",
    "execute_spec_async",
    "logs_for",
    "smoke_passed",
    "workspace_for",
]
