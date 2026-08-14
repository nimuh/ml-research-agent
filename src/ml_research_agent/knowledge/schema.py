"""KB record schemas + the note/wiki front-matter contract (id, type, sources,
claims, tags, confidence, added_at, supersedes)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from ..types import MetricReport, Note, Provenance, utcnow
from ..utils.io import front_matter, with_front_matter

SCHEMA_VERSION = 1

# One statement per table plus the indexes that matter. Executed with
# ``executescript`` on every open, so every statement is IF NOT EXISTS.
#
# The schema deliberately stores *metadata and relations* only: note bodies live
# in Markdown under kb_wiki/ and blobs under kb_raw/, because files are truth and
# this database has to be droppable and rebuildable from them.
SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    key           TEXT NOT NULL,
    title         TEXT NOT NULL DEFAULT '',
    path          TEXT,
    content_hash  TEXT NOT NULL,
    added_at      TEXT NOT NULL,
    metadata      TEXT NOT NULL DEFAULT '{}',
    UNIQUE (kind, key)
);
CREATE INDEX IF NOT EXISTS documents_hash_idx ON documents (content_hash);
CREATE INDEX IF NOT EXISTS documents_kind_idx ON documents (kind);

CREATE TABLE IF NOT EXISTS passages (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    text          TEXT NOT NULL,
    section       TEXT,
    page          INTEGER,
    ord           INTEGER NOT NULL DEFAULT 0,
    token_count   INTEGER NOT NULL DEFAULT 0,
    content_hash  TEXT NOT NULL DEFAULT '',
    metadata      TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS passages_doc_idx ON passages (doc_id, ord);

CREATE TABLE IF NOT EXISTS notes (
    id             TEXT PRIMARY KEY,
    type           TEXT NOT NULL,
    paper_key      TEXT,
    title          TEXT NOT NULL DEFAULT '',
    path           TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.0,
    added_at       TEXT NOT NULL,
    supersedes     TEXT,
    superseded_by  TEXT,
    sources        TEXT NOT NULL DEFAULT '[]',
    claims         TEXT NOT NULL DEFAULT '[]',
    tags           TEXT NOT NULL DEFAULT '[]',
    provenance_n   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS notes_type_idx ON notes (type);
CREATE INDEX IF NOT EXISTS notes_paper_idx ON notes (paper_key);

CREATE TABLE IF NOT EXISTS claims (
    id              TEXT PRIMARY KEY,
    statement       TEXT NOT NULL,
    subject         TEXT,
    relation        TEXT,
    object          TEXT,
    confidence      REAL NOT NULL DEFAULT 0.0,
    evidence_count  INTEGER NOT NULL DEFAULT 0,
    data            TEXT NOT NULL,
    added_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    provenance  TEXT,
    added_at    TEXT NOT NULL,
    PRIMARY KEY (source, target, relation)
);
CREATE INDEX IF NOT EXISTS edges_relation_idx ON edges (relation);

CREATE TABLE IF NOT EXISTS repos (
    id          TEXT PRIMARY KEY,
    url         TEXT NOT NULL,
    name        TEXT NOT NULL DEFAULT '',
    commit_sha  TEXT,
    license     TEXT,
    paper_key   TEXT,
    data        TEXT NOT NULL DEFAULT '{}',
    added_at    TEXT NOT NULL,
    UNIQUE (url, commit_sha)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash  TEXT NOT NULL,
    doc_id        TEXT,
    kind          TEXT NOT NULL,
    key           TEXT NOT NULL,
    passages      INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT '',
    at            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ingest_log_hash_idx ON ingest_log (content_hash);
"""


class DocumentRecord(BaseModel):
    """One ingested source document: a paper, a repo snapshot, a page, a report."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: str = "paper"
    key: str
    title: str = ""
    path: str | None = None
    content_hash: str
    added_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NoteFrontMatter(BaseModel):
    """The YAML header every wiki note carries.

    ``extra="ignore"`` because :func:`note_to_markdown` writes further machine
    fields (metrics, provenance, ...) alongside this contract; parsing the
    contract must not choke on them.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str = "paper"
    title: str = ""
    paper_key: str | None = None
    sources: list[str] = Field(default_factory=list)
    claims: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    added_at: datetime = Field(default_factory=utcnow)
    supersedes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "paper_key": self.paper_key,
            "sources": list(self.sources),
            "claims": list(self.claims),
            "tags": list(self.tags),
            "confidence": self.confidence,
            "added_at": self.added_at.isoformat(),
            "supersedes": self.supersedes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoteFrontMatter:
        return cls.model_validate(data)

    @classmethod
    def from_note(cls, note: Note) -> Self:
        return cls(
            id=note.id,
            type=note.type,
            title=note.title,
            paper_key=note.paper_key,
            sources=list(note.sources),
            claims=list(note.claims),
            tags=list(note.tags),
            confidence=float(note.confidence),
            added_at=note.added_at,
            supersedes=note.supersedes,
        )


# Headings note_to_markdown emits. Parsing the summary stops at one of these, so
# a summary containing its own '## ...' subheadings still round-trips.
_BODY_HEADINGS = (
    "Summary",
    "Task",
    "Method",
    "Datasets",
    "Baselines",
    "Metrics",
    "Compute",
    "Limitations",
    "Relevance to brief",
    "Provenance",
)


def note_to_markdown(note: Note) -> str:
    """Render a note as wiki Markdown: front matter + human-readable body.

    Everything not reconstructible from the body is kept in the front matter, so
    :func:`note_from_markdown` is an exact inverse -- the note file, not the
    database row, is the note.
    """
    meta = NoteFrontMatter.from_note(note).to_dict()
    meta.update(
        {
            "created_at": note.created_at.isoformat(),
            "task": note.task,
            "method": note.method,
            "datasets": list(note.datasets),
            "baselines": list(note.baselines),
            "compute": note.compute,
            "limitations": list(note.limitations),
            "relevance_to_brief": note.relevance_to_brief,
            "metrics": [m.model_dump(mode="json") for m in note.metrics],
            "provenance": [p.model_dump(mode="json") for p in note.provenance],
        }
    )

    lines = [f"# {note.title}", "", "## Summary", "", note.summary or "_(no summary)_"]
    if note.task:
        lines += ["", "## Task", "", note.task]
    if note.method:
        lines += ["", "## Method", "", note.method]
    if note.datasets:
        lines += ["", "## Datasets", "", *(f"- {d}" for d in note.datasets)]
    if note.baselines:
        lines += ["", "## Baselines", "", *(f"- {b}" for b in note.baselines)]
    if note.metrics:
        lines += ["", "## Metrics", "", *(f"- {_metric_line(m)}" for m in note.metrics)]
    if note.compute:
        lines += ["", "## Compute", "", note.compute]
    if note.limitations:
        lines += ["", "## Limitations", "", *(f"- {limit}" for limit in note.limitations)]
    if note.relevance_to_brief:
        lines += ["", "## Relevance to brief", "", note.relevance_to_brief]
    if note.provenance:
        lines += ["", "## Provenance", "", *(f"- {p}" for p in note.provenance)]
    return with_front_matter(meta, "\n".join(lines))


def note_from_markdown(text: str) -> Note:
    """Parse a wiki note back into a :class:`Note`. Inverse of the renderer."""
    meta, body = front_matter(text)
    fm = NoteFrontMatter.from_dict(meta)
    summary = _read_section(body, "Summary")
    return Note(
        id=fm.id,
        type=fm.type,
        paper_key=fm.paper_key,
        title=fm.title,
        summary="" if summary == "_(no summary)_" else summary,
        task=meta.get("task"),
        method=meta.get("method"),
        datasets=list(meta.get("datasets") or []),
        baselines=list(meta.get("baselines") or []),
        metrics=[MetricReport.model_validate(m) for m in meta.get("metrics") or []],
        compute=meta.get("compute"),
        limitations=list(meta.get("limitations") or []),
        relevance_to_brief=meta.get("relevance_to_brief"),
        confidence=fm.confidence,
        claims=list(fm.claims),
        sources=list(fm.sources),
        supersedes=fm.supersedes,
        added_at=fm.added_at,
        created_at=_parse_dt(meta.get("created_at")) or fm.added_at,
        provenance=[Provenance.model_validate(p) for p in meta.get("provenance") or []],
        tags=list(fm.tags),
    )


def note_filename(note: Note) -> str:
    """Stable, human-greppable file name for a note in the wiki."""
    from ..utils.io import slugify

    return f"{slugify(note.title)}--{note.id}.md"


def _metric_line(metric: MetricReport) -> str:
    bits = [metric.name]
    if metric.value is not None:
        bits.append(f"= {metric.value}{metric.unit or ''}")
    if metric.dataset:
        bits.append(f"on {metric.dataset}")
    if metric.split:
        bits.append(f"({metric.split})")
    if metric.method:
        bits.append(f"[{metric.method}]")
    bits.append(f"-- {metric.provenance}")
    return " ".join(bits)


def _read_section(body: str, heading: str) -> str:
    """Return the text under ``## heading`` up to the next *known* heading."""
    marker = f"## {heading}"
    start = body.find(marker)
    if start < 0:
        return ""
    cursor = start + len(marker)
    rest = body[cursor:]
    end = len(rest)
    for other in _BODY_HEADINGS:
        if other == heading:
            continue
        found = rest.find(f"\n## {other}")
        if found >= 0:
            end = min(end, found)
    return rest[:end].strip()


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        return datetime.fromisoformat(value)
    return None


__all__ = [
    "SCHEMA_VERSION",
    "SQL_SCHEMA",
    "DocumentRecord",
    "NoteFrontMatter",
    "note_filename",
    "note_from_markdown",
    "note_to_markdown",
]
