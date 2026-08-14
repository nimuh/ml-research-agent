"""Checkpoint/resume: durable snapshots of ProjectState keyed by run id, plus
diffing so a resumed run re-executes only invalidated steps."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..types import Phase, utcnow
from ..utils.hashing import hash_obj
from ..utils.io import ensure_dir, read_json, write_json
from .state import ProjectSnapshot, ProjectState


@dataclass(frozen=True)
class Checkpoint:
    """A phase boundary, with the hash of the inputs that produced it."""

    project_id: str
    phase: Phase
    seq: int
    input_hash: str
    at: datetime
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "phase": self.phase.value,
            "seq": self.seq,
            "input_hash": self.input_hash,
            "at": self.at.isoformat(),
        }


class CheckpointStore:
    """Snapshots at phase boundaries, plus the input hash that justified each.

    Resume re-executes a phase only when its *inputs* changed. Comparing
    outputs instead would re-run everything after any nondeterministic step.
    """

    def __init__(self, root: Path) -> None:
        self.root = ensure_dir(root)

    def _dir(self, project_id: str) -> Path:
        return ensure_dir(self.root / project_id)

    def save(self, state: ProjectState, phase: Phase, *, inputs: Any = None) -> Checkpoint:
        directory = self._dir(state.project_id)
        input_hash = hash_obj(inputs) if inputs is not None else ""
        path = directory / f"{phase.value}.json"
        checkpoint = Checkpoint(
            project_id=state.project_id,
            phase=phase,
            seq=state.snapshot.event_count,
            input_hash=input_hash,
            at=utcnow(),
            path=path,
        )
        write_json(
            path,
            {"meta": checkpoint.to_dict(), "snapshot": state.snapshot.model_dump(mode="json")},
        )
        return checkpoint

    def load(self, project_id: str, phase: Phase) -> tuple[Checkpoint, ProjectSnapshot] | None:
        path = self._dir(project_id) / f"{phase.value}.json"
        data = read_json(path)
        if not data:
            return None
        meta = data["meta"]
        checkpoint = Checkpoint(
            project_id=meta["project_id"],
            phase=Phase(meta["phase"]),
            seq=int(meta["seq"]),
            input_hash=meta.get("input_hash", ""),
            at=datetime.fromisoformat(meta["at"]),
            path=path,
        )
        return checkpoint, ProjectSnapshot.model_validate(data["snapshot"])

    def latest(self, project_id: str) -> Checkpoint | None:
        found: list[Checkpoint] = []
        for phase in Phase:
            loaded = self.load(project_id, phase)
            if loaded:
                found.append(loaded[0])
        return max(found, key=lambda c: c.seq) if found else None

    def is_valid(self, project_id: str, phase: Phase, *, inputs: Any) -> bool:
        """True when a completed phase's inputs are unchanged -- safe to skip."""
        loaded = self.load(project_id, phase)
        if not loaded:
            return False
        checkpoint, _ = loaded
        return bool(checkpoint.input_hash) and checkpoint.input_hash == hash_obj(inputs)

    def invalidate(self, project_id: str, phase: Phase) -> None:
        """Drop this phase and every phase downstream of it."""
        from ..types import PHASE_ORDER

        try:
            start = PHASE_ORDER.index(phase)
        except ValueError:
            return
        for downstream in PHASE_ORDER[start:]:
            (self._dir(project_id) / f"{downstream.value}.json").unlink(missing_ok=True)

    def phases_completed(self, project_id: str) -> list[Phase]:
        return [
            phase for phase in Phase if (self._dir(project_id) / f"{phase.value}.json").exists()
        ]


__all__ = ["Checkpoint", "CheckpointStore"]
