"""Ingest pipeline: raw doc -> parse -> chunk -> embed -> index -> note stub.
Idempotent by content hash; safe to re-run over a growing raw/ folder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..types import Note, Paper, PaperSection, Passage
from ..utils.hashing import hash_text
from .chunk import chunk_sections, chunk_text
from .index import HybridIndex
from .notes import note_stub_from_paper
from .store import KnowledgeStore, paper_content_hash

# Text formats this layer can ingest on its own. PDFs and LaTeX are parsed by
# literature/, which knowledge/ must not import -- see ingest_paper's `sections`.
TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".text")


@dataclass
class IngestResult:
    """What one ingest attempt did, including deciding to do nothing."""

    doc_id: str
    passages: int
    skipped: bool
    reason: str = ""
    note_stub: Note | None = None


class Ingestor:
    """raw doc -> store -> chunks -> index -> note stub.

    Idempotent by content hash: re-running over a growing ``raw/`` re-ingests
    only what actually changed, which is what lets the survey phase be resumed
    and re-run without curating the same paper twice.
    """

    def __init__(self, config: Config, store: KnowledgeStore, index: HybridIndex) -> None:
        self.config = config
        self.store = store
        self.index = index

    def ingest_paper(
        self, paper: Paper, *, sections: Sequence[PaperSection] | None = None
    ) -> IngestResult:
        """Ingest a paper, chunking its parsed sections when they exist.

        This layer never parses a PDF: ``sections`` come from literature/, and
        without them the paper is ingested abstract-only and says so, because a
        silently abstract-only KB is the failure mode that looks like success.
        """
        if sections is not None and not paper.sections:
            paper = paper.model_copy(update={"sections": list(sections)})
        content_hash = paper_content_hash(paper)

        existing = self.store.document_by_hash(content_hash)
        if existing is not None:
            self.store.log_ingest(
                content_hash=content_hash,
                doc_id=existing.id,
                kind="paper",
                key=paper.key,
                passages=0,
                status="skipped",
                reason="content hash already ingested",
            )
            return IngestResult(
                doc_id=existing.id,
                passages=0,
                skipped=True,
                reason=f"already ingested (content_hash={content_hash[:12]})",
            )

        doc_id = self.store.add_paper(paper)
        reason = ""
        if paper.sections:
            passages = chunk_sections(paper.sections, doc_id, self.config.knowledge)
        else:
            reason = "abstract-only: no parsed sections were supplied"
            passages = chunk_text(
                paper.abstract or paper.title,
                doc_id,
                self.config.knowledge,
                section="Abstract",
            )

        self._replace_passages(doc_id, passages)
        self.store.log_ingest(
            content_hash=content_hash,
            doc_id=doc_id,
            kind="paper",
            key=paper.key,
            passages=len(passages),
            status="ingested",
            reason=reason,
        )
        return IngestResult(
            doc_id=doc_id,
            passages=len(passages),
            skipped=False,
            reason=reason,
            note_stub=note_stub_from_paper(paper, passages=passages),
        )

    def ingest_text(
        self,
        *,
        key: str,
        title: str,
        text: str,
        kind: str = "web",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a plain text document (README, web page, report)."""
        content_hash = hash_text(text, length=32)
        existing = self.store.document_by_hash(content_hash)
        if existing is not None:
            self.store.log_ingest(
                content_hash=content_hash,
                doc_id=existing.id,
                kind=kind,
                key=key,
                passages=0,
                status="skipped",
                reason="content hash already ingested",
            )
            return IngestResult(
                doc_id=existing.id,
                passages=0,
                skipped=True,
                reason=f"already ingested (content_hash={content_hash[:12]})",
            )

        record = self.store.add_document(
            kind=kind,
            key=key,
            title=title,
            path=(metadata or {}).get("path"),
            content_hash=content_hash,
            metadata=metadata or {},
        )
        passages = chunk_text(text, record.id, self.config.knowledge)
        self._replace_passages(record.id, passages)
        self.store.log_ingest(
            content_hash=content_hash,
            doc_id=record.id,
            kind=kind,
            key=key,
            passages=len(passages),
            status="ingested",
        )
        return IngestResult(doc_id=record.id, passages=len(passages), skipped=False)

    def ingest_raw_dir(self, path: Path | None = None) -> list[IngestResult]:
        """Scan ``kb_raw`` (or ``path``) and ingest every text document found.

        Binary formats are reported as skipped rather than ignored: a PDF the KB
        never read is a gap somebody has to see.
        """
        root = Path(path or self.config.paths.kb_raw)
        results: list[IngestResult] = []
        if not root.exists():
            return results
        for file in sorted(p for p in root.rglob("*") if p.is_file()):
            key = file.relative_to(root).as_posix()
            if file.suffix.lower() not in TEXT_SUFFIXES:
                results.append(
                    IngestResult(
                        doc_id="",
                        passages=0,
                        skipped=True,
                        reason=(
                            f"{file.suffix or 'binary'} is not parsed in knowledge/; "
                            "literature/parse.py produces the sections"
                        ),
                    )
                )
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
            results.append(
                self.ingest_text(
                    key=key,
                    title=_title_for(text, file),
                    text=text,
                    kind="web",
                    metadata={"path": str(file)},
                )
            )
        return results

    def _replace_passages(self, doc_id: str, passages: Sequence[Passage]) -> None:
        """Swap a document's passages in store and index together.

        Chunk ids are deterministic, so a re-ingested document reuses the ids of
        the chunks that did not change; the ones that vanished are removed from
        the index rather than left behind as unreachable hits.
        """
        stale = {p.id for p in self.store.list_passages(doc_id=doc_id)} - {p.id for p in passages}
        self.store.delete_passages(doc_id=doc_id)
        self.store.add_passages(passages)
        if stale:
            self.index.remove(sorted(stale))
        self.index.upsert(passages)
        self.index.save()

    def rebuild_indexes(self) -> int:
        """Drop and rebuild the derived indexes from the store. Returns passages."""
        return self.index.rebuild(self.store)


def _title_for(text: str, file: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return file.stem


__all__ = ["TEXT_SUFFIXES", "IngestResult", "Ingestor"]
