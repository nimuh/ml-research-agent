"""KnowledgeStore: CRUD over the KB (SQLite for metadata/relations, files for
content, blobs for PDFs). Transactions, provenance and content-hash dedup."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from ..config import Config
from ..errors import MRAError
from ..types import (
    Claim,
    ClaimRelation,
    GraphEdge,
    Note,
    Paper,
    PaperSection,
    Passage,
    Provenance,
    new_id,
    utcnow,
)
from ..utils.hashing import hash_obj
from ..utils.io import ensure_dir, read_text, write_text
from .schema import (
    SCHEMA_VERSION,
    SQL_SCHEMA,
    DocumentRecord,
    note_filename,
    note_from_markdown,
    note_to_markdown,
)


def paper_content_hash(paper: Paper, sections: Sequence[PaperSection] | None = None) -> str:
    """Content identity of a paper: what would change the KB if it changed.

    Ids, timestamps and provenance are excluded so re-fetching the same paper
    from a second source hashes identically -- that is what makes ingest
    idempotent rather than merely deduplicated.
    """
    parts = list(sections if sections is not None else paper.sections)
    return hash_obj(
        {
            "key": paper.key,
            "title": paper.title,
            "abstract": paper.abstract,
            "sections": [(s.title, s.text, s.order) for s in parts],
        },
        length=32,
    )


class KnowledgeStore:
    """The KB's transactional metadata layer.

    SQLite holds documents, passages, note *metadata*, claims and graph edges.
    Note bodies are Markdown files under ``paths.kb_wiki`` -- the database is a
    derived index over them and can be deleted and rebuilt.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.paths = config.paths
        ensure_dir(self.paths.kb_wiki)
        ensure_dir(Path(self.paths.kb_sqlite).parent)
        self.conn = sqlite3.connect(str(self.paths.kb_sqlite))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SQL_SCHEMA)
        self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        self.conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        row = self.conn.execute("PRAGMA user_version").fetchone()
        return int(row[0])

    # -- documents ---------------------------------------------------------

    def add_document(
        self,
        *,
        kind: str,
        key: str,
        title: str,
        path: str | None,
        content_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        """Insert or update a document, keyed by ``(kind, key)``."""
        existing = self._document_row(kind=kind, key=key)
        doc_id = str(existing["id"]) if existing else new_id("doc")
        added_at = _parse_dt(existing["added_at"]) if existing else utcnow()
        record = DocumentRecord(
            id=doc_id,
            kind=kind,
            key=key,
            title=title,
            path=path,
            content_hash=content_hash,
            added_at=added_at,
            metadata=metadata or {},
        )
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO documents (id, kind, key, title, path, content_hash, added_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (kind, key) DO UPDATE SET
                    title = excluded.title,
                    path = excluded.path,
                    content_hash = excluded.content_hash,
                    metadata = excluded.metadata
                """,
                (
                    record.id,
                    record.kind,
                    record.key,
                    record.title,
                    record.path,
                    record.content_hash,
                    record.added_at.isoformat(),
                    json.dumps(record.metadata, default=str),
                ),
            )
        return record

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self.conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _document_from_row(row) if row else None

    def document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        """The idempotency check: has this exact content already been ingested?"""
        row = self.conn.execute(
            "SELECT * FROM documents WHERE content_hash = ? ORDER BY added_at LIMIT 1",
            (content_hash,),
        ).fetchone()
        return _document_from_row(row) if row else None

    def document_by_key(self, kind: str, key: str) -> DocumentRecord | None:
        row = self._document_row(kind=kind, key=key)
        return _document_from_row(row) if row else None

    def list_documents(self, *, kind: str | None = None) -> list[DocumentRecord]:
        if kind is None:
            rows = self.conn.execute("SELECT * FROM documents ORDER BY added_at, id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM documents WHERE kind = ? ORDER BY added_at, id", (kind,)
            ).fetchall()
        return [_document_from_row(row) for row in rows]

    def _document_row(self, *, kind: str, key: str) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(
            "SELECT * FROM documents WHERE kind = ? AND key = ?", (kind, key)
        ).fetchone()
        return row

    # -- papers ------------------------------------------------------------

    def add_paper(self, paper: Paper) -> str:
        """Upsert a paper by :attr:`Paper.key`; returns its document id."""
        record = self.add_document(
            kind="paper",
            key=paper.key,
            title=paper.title,
            path=paper.raw_path,
            content_hash=paper_content_hash(paper),
            metadata={"paper": paper.model_dump(mode="json")},
        )
        return record.id

    def get_paper(self, key: str) -> Paper | None:
        record = self.document_by_key("paper", key)
        if record is None:
            return None
        payload = record.metadata.get("paper")
        return Paper.model_validate(payload) if payload else None

    def list_papers(self) -> list[Paper]:
        papers = []
        for record in self.list_documents(kind="paper"):
            payload = record.metadata.get("paper")
            if payload:
                papers.append(Paper.model_validate(payload))
        return papers

    def paper_for_doc(self, doc_id: str) -> Paper | None:
        record = self.get_document(doc_id)
        if record is None or record.kind != "paper":
            return None
        payload = record.metadata.get("paper")
        return Paper.model_validate(payload) if payload else None

    # -- passages ----------------------------------------------------------

    def add_passages(self, passages: Sequence[Passage]) -> int:
        """Insert or replace passages by id. Returns how many were written."""
        rows = [
            (
                p.id,
                p.doc_id,
                p.text,
                p.section,
                p.page,
                p.order,
                p.token_count,
                p.content_hash,
                json.dumps(p.metadata, default=str),
            )
            for p in passages
        ]
        if not rows:
            return 0
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO passages
                    (id, doc_id, text, section, page, ord, token_count, content_hash, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    text = excluded.text,
                    section = excluded.section,
                    page = excluded.page,
                    ord = excluded.ord,
                    token_count = excluded.token_count,
                    content_hash = excluded.content_hash,
                    metadata = excluded.metadata
                """,
                rows,
            )
        return len(rows)

    def get_passage(self, passage_id: str) -> Passage | None:
        row = self.conn.execute("SELECT * FROM passages WHERE id = ?", (passage_id,)).fetchone()
        return _passage_from_row(row) if row else None

    def list_passages(self, *, doc_id: str | None = None) -> list[Passage]:
        if doc_id is None:
            rows = self.conn.execute("SELECT * FROM passages ORDER BY doc_id, ord, id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM passages WHERE doc_id = ? ORDER BY ord, id", (doc_id,)
            ).fetchall()
        return [_passage_from_row(row) for row in rows]

    def delete_passages(self, *, doc_id: str) -> int:
        with self.conn:
            cursor = self.conn.execute("DELETE FROM passages WHERE doc_id = ?", (doc_id,))
        return int(cursor.rowcount)

    # -- notes -------------------------------------------------------------

    def save_note(self, note: Note) -> Path:
        """Write the note's Markdown (truth) and index its metadata (derived)."""
        path = ensure_dir(self.paths.kb_wiki) / note_filename(note)
        write_text(path, note_to_markdown(note))
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO notes
                    (id, type, paper_key, title, path, confidence, added_at, supersedes,
                     superseded_by, sources, claims, tags, provenance_n)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    type = excluded.type,
                    paper_key = excluded.paper_key,
                    title = excluded.title,
                    path = excluded.path,
                    confidence = excluded.confidence,
                    added_at = excluded.added_at,
                    supersedes = excluded.supersedes,
                    sources = excluded.sources,
                    claims = excluded.claims,
                    tags = excluded.tags,
                    provenance_n = excluded.provenance_n
                """,
                (
                    note.id,
                    note.type,
                    note.paper_key,
                    note.title,
                    str(path),
                    float(note.confidence),
                    note.added_at.isoformat(),
                    note.supersedes,
                    json.dumps(list(note.sources)),
                    json.dumps(list(note.claims)),
                    json.dumps(list(note.tags)),
                    len(note.provenance),
                ),
            )
        return path

    def get_note(self, note_id: str) -> Note | None:
        row = self.conn.execute("SELECT path FROM notes WHERE id = ?", (note_id,)).fetchone()
        if row is None:
            return None
        path = Path(str(row["path"]))
        if not path.exists():
            raise MRAError("note file missing; the wiki is truth", note_id=note_id, path=str(path))
        return note_from_markdown(read_text(path))

    def list_notes(self, *, type: str | None = None, include_superseded: bool = True) -> list[Note]:
        sql = "SELECT id FROM notes"
        params: list[Any] = []
        clauses = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type)
        if not include_superseded:
            clauses.append("superseded_by IS NULL")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY added_at, id"
        notes = []
        for row in self.conn.execute(sql, params).fetchall():
            note = self.get_note(str(row["id"]))
            if note is not None:
                notes.append(note)
        return notes

    def supersede_note(self, old_id: str, new: Note) -> Note:
        """Record ``new`` as the successor of ``old_id``; the old note is kept.

        History is the point: the old Markdown file is never rewritten, so the
        KB can always show what it used to believe and why that changed.
        """
        old = self.get_note(old_id)
        if old is None:
            raise MRAError("cannot supersede a note that is not in the KB", note_id=old_id)
        successor = new.model_copy(update={"supersedes": old_id})
        self.save_note(successor)
        with self.conn:
            self.conn.execute(
                "UPDATE notes SET superseded_by = ? WHERE id = ?", (successor.id, old_id)
            )
        return successor

    def superseded_by(self, note_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT superseded_by FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        return str(row["superseded_by"]) if row and row["superseded_by"] else None

    # -- claims ------------------------------------------------------------

    def add_claim(self, claim: Claim) -> str:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO claims
                    (id, statement, subject, relation, object, confidence, evidence_count,
                     data, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    statement = excluded.statement,
                    subject = excluded.subject,
                    relation = excluded.relation,
                    object = excluded.object,
                    confidence = excluded.confidence,
                    evidence_count = excluded.evidence_count,
                    data = excluded.data
                """,
                (
                    claim.id,
                    claim.statement,
                    claim.subject,
                    claim.relation,
                    claim.object,
                    float(claim.confidence),
                    len(claim.evidence),
                    json.dumps(claim.model_dump(mode="json")),
                    claim.created_at.isoformat(),
                ),
            )
        return claim.id

    def get_claim(self, claim_id: str) -> Claim | None:
        row = self.conn.execute("SELECT data FROM claims WHERE id = ?", (claim_id,)).fetchone()
        return Claim.model_validate(json.loads(str(row["data"]))) if row else None

    def list_claims(self) -> list[Claim]:
        rows = self.conn.execute("SELECT data FROM claims ORDER BY added_at, id").fetchall()
        return [Claim.model_validate(json.loads(str(row["data"]))) for row in rows]

    # -- edges -------------------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO edges (source, target, relation, weight, provenance, added_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (source, target, relation) DO UPDATE SET
                    weight = excluded.weight,
                    provenance = excluded.provenance
                """,
                (
                    edge.source,
                    edge.target,
                    edge.relation.value,
                    edge.weight,
                    json.dumps(edge.provenance.model_dump(mode="json"))
                    if edge.provenance
                    else None,
                    utcnow().isoformat(),
                ),
            )

    def list_edges(self, *, relation: ClaimRelation | None = None) -> list[GraphEdge]:
        if relation is None:
            rows = self.conn.execute("SELECT * FROM edges ORDER BY added_at, source").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM edges WHERE relation = ? ORDER BY added_at, source",
                (relation.value,),
            ).fetchall()
        return [
            GraphEdge(
                source=str(row["source"]),
                target=str(row["target"]),
                relation=ClaimRelation(str(row["relation"])),
                weight=float(row["weight"]),
                provenance=(
                    Provenance.model_validate(json.loads(str(row["provenance"])))
                    if row["provenance"]
                    else None
                ),
            )
            for row in rows
        ]

    # -- ingest log --------------------------------------------------------

    def log_ingest(
        self,
        *,
        content_hash: str,
        doc_id: str | None,
        kind: str,
        key: str,
        passages: int,
        status: str,
        reason: str = "",
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO ingest_log
                    (content_hash, doc_id, kind, key, passages, status, reason, at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (content_hash, doc_id, kind, key, passages, status, reason, utcnow().isoformat()),
            )

    def ingest_log(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM ingest_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Counts and sizes behind ``mra kb stats``."""

        def count(table: str) -> int:
            row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            return int(row["n"])

        by_kind = {
            str(row["kind"]): int(row["n"])
            for row in self.conn.execute(
                "SELECT kind, COUNT(*) AS n FROM documents GROUP BY kind ORDER BY kind"
            ).fetchall()
        }
        by_note_type = {
            str(row["type"]): int(row["n"])
            for row in self.conn.execute(
                "SELECT type, COUNT(*) AS n FROM notes GROUP BY type ORDER BY type"
            ).fetchall()
        }
        superseded = self.conn.execute(
            "SELECT COUNT(*) AS n FROM notes WHERE superseded_by IS NOT NULL"
        ).fetchone()
        last = self.conn.execute("SELECT at FROM ingest_log ORDER BY id DESC LIMIT 1").fetchone()
        tokens = self.conn.execute("SELECT SUM(token_count) AS n FROM passages").fetchone()
        sqlite_path = Path(self.paths.kb_sqlite)
        return {
            "schema_version": self.schema_version,
            "documents": count("documents"),
            "documents_by_kind": by_kind,
            "passages": count("passages"),
            "passage_tokens": int(tokens["n"] or 0),
            "notes": count("notes"),
            "notes_by_type": by_note_type,
            "superseded_notes": int(superseded["n"]),
            "claims": count("claims"),
            "edges": count("edges"),
            "repos": count("repos"),
            "ingest_events": count("ingest_log"),
            "last_ingest_at": str(last["at"]) if last else None,
            "sqlite_bytes": sqlite_path.stat().st_size if sqlite_path.exists() else 0,
            "wiki_bytes": sum(
                p.stat().st_size for p in Path(self.paths.kb_wiki).glob("*.md") if p.is_file()
            ),
        }


def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=str(row["id"]),
        kind=str(row["kind"]),
        key=str(row["key"]),
        title=str(row["title"]),
        path=str(row["path"]) if row["path"] else None,
        content_hash=str(row["content_hash"]),
        added_at=_parse_dt(row["added_at"]),
        metadata=json.loads(str(row["metadata"] or "{}")),
    )


def _passage_from_row(row: sqlite3.Row) -> Passage:
    return Passage(
        id=str(row["id"]),
        doc_id=str(row["doc_id"]),
        text=str(row["text"]),
        section=str(row["section"]) if row["section"] is not None else None,
        page=int(row["page"]) if row["page"] is not None else None,
        order=int(row["ord"]),
        token_count=int(row["token_count"]),
        content_hash=str(row["content_hash"]),
        metadata=json.loads(str(row["metadata"] or "{}")),
    )


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


__all__ = ["KnowledgeStore", "paper_content_hash"]
