"""Note/wiki generation and maintenance: per-paper notes, per-concept pages,
per-method comparisons; merge/supersede logic when new evidence arrives."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import KnowledgeConfig
from ..types import Claim, MetricReport, Note, Paper, Passage, Provenance, new_id, utcnow

# A stub is a placeholder for a reading, not a reading: the Curator raises this
# once it has actually extracted structure from the passages.
STUB_CONFIDENCE = 0.2


def note_stub_from_paper(paper: Paper, *, passages: Sequence[Passage] = ()) -> Note:
    """The empty note ingest leaves behind for the Curator to fill in.

    It carries provenance from the moment it exists -- creating a note that
    cannot yet be traced would make the "no note without provenance" rule
    something ingest routinely violates and everyone learns to ignore.
    """
    provenance = [p.to_provenance() for p in list(passages)[:3]]
    if not provenance:
        provenance = [
            Provenance(
                source=paper.key,
                locator="§Abstract",
                quote=(paper.abstract or paper.title)[:280],
                confidence=0.5,
            )
        ]
    sources = [s for s in (paper.url, paper.pdf_url, paper.key) if s]
    return Note(
        id=new_id("note"),
        type="paper",
        paper_key=paper.key,
        title=paper.title,
        summary=paper.tldr or paper.abstract or "",
        confidence=STUB_CONFIDENCE,
        sources=sources,
        provenance=provenance,
        tags=["stub"],
        added_at=utcnow(),
    )


def validate_note(note: Note, config: KnowledgeConfig) -> list[str]:
    """Return the reasons this note is inadmissible. Empty means admissible.

    This is where "notes without passage-level provenance are rejected" actually
    bites: a locator-free provenance entry names a document but not a place in
    it, which is exactly the shape a fabricated citation takes.
    """
    violations: list[str] = []
    exempt = note.type == "todo"

    if not note.title:
        violations.append("note has no title")
    if not note.summary and not exempt:
        violations.append("note has no summary")
    if note.type == "paper" and not note.paper_key:
        violations.append("paper note has no paper_key")

    if not note.provenance and not exempt:
        violations.append("note has no provenance; it asserts things nothing backs")
    elif config.require_citations:
        for i, prov in enumerate(note.provenance):
            if not prov.locator:
                violations.append(
                    f"provenance[{i}] ({prov.source}) has no locator; "
                    "passage-level provenance is required"
                )

    for metric in note.metrics:
        if not metric.provenance.locator:
            violations.append(f"metric '{metric.name}' has no passage locator")

    if note.claims and not (note.sources or note.provenance):
        violations.append("note asserts claims but records no sources")
    if config.require_citations and note.claims and not note.provenance:
        violations.append("claims are not backed by any passage")
    if note.confidence <= 0.0 and not exempt:
        violations.append("note has no confidence set")
    return violations


def merge_notes(old: Note, new: Note) -> Note:
    """Fold ``old`` into ``new`` and mark the succession.

    Supersede rather than overwrite: the KB keeps what it used to believe, so a
    later reader can see that a number changed and go find out why.
    """
    merged = new.model_copy(
        update={
            "supersedes": old.id,
            "task": new.task or old.task,
            "method": new.method or old.method,
            "compute": new.compute or old.compute,
            "relevance_to_brief": new.relevance_to_brief or old.relevance_to_brief,
            "paper_key": new.paper_key or old.paper_key,
            "datasets": _union(old.datasets, new.datasets),
            "baselines": _union(old.baselines, new.baselines),
            "limitations": _union(old.limitations, new.limitations),
            "sources": _union(old.sources, new.sources),
            "claims": _union(old.claims, new.claims),
            "tags": _union(old.tags, new.tags),
            "metrics": _merge_metrics(old.metrics, new.metrics),
            "provenance": _merge_provenance(old.provenance, new.provenance),
            "summary": new.summary or old.summary,
            "added_at": utcnow(),
        }
    )
    return merged


def concept_page(name: str, notes: Sequence[Note], *, claims: Sequence[Claim] = ()) -> Note:
    """A wiki page that gathers what the KB knows about one concept."""
    lines = [f"What the KB records about **{name}**, compiled from {len(notes)} note(s)."]
    if notes:
        lines += ["", "Notes:"]
        lines += [f"- {n.title} ({n.paper_key or n.id})" for n in notes]
    if claims:
        lines += ["", "Claims:"]
        lines += [f"- {c.statement}" + (" [contested]" if c.is_contested else "") for c in claims]
    datasets = _union(*[n.datasets for n in notes]) if notes else []
    if datasets:
        lines += ["", "Datasets seen: " + ", ".join(datasets)]

    provenance = _merge_provenance(*[list(n.provenance) for n in notes]) if notes else []
    return Note(
        id=new_id("note"),
        type="concept",
        title=name,
        summary="\n".join(lines),
        datasets=datasets,
        claims=[c.id for c in claims],
        sources=[n.id for n in notes],
        provenance=provenance,
        confidence=max((n.confidence for n in notes), default=0.3),
        tags=["concept"],
        added_at=utcnow(),
    )


def comparison_table(notes: Sequence[Note], *, metric: str | None = None) -> str:
    """Markdown table comparing notes side by side, optionally on one metric."""
    header = ["Paper", "Method", "Datasets"]
    header.append(metric if metric else "Headline metric")
    header += ["Confidence", "Source"]
    rows = [f"| {' | '.join(header)} |", f"|{'---|' * len(header)}"]
    for note in notes:
        value = _metric_cell(note, metric)
        rows.append(
            "| "
            + " | ".join(
                [
                    _escape(note.title),
                    _escape(note.method or "-"),
                    _escape(", ".join(note.datasets) or "-"),
                    value,
                    f"{note.confidence:.2f}",
                    _escape(note.paper_key or (note.sources[0] if note.sources else note.id)),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _metric_cell(note: Note, metric: str | None) -> str:
    candidates = [m for m in note.metrics if metric is None or m.name == metric]
    if not candidates:
        return "-"
    best = candidates[0]
    value = "?" if best.value is None else f"{best.value:g}{best.unit or ''}"
    name = "" if metric else f"{best.name} = "
    return _escape(f"{name}{value}")


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _union(*lists: Sequence[str]) -> list[str]:
    """Order-preserving union; the KB's lists are read by humans."""
    out: list[str] = []
    for items in lists:
        for item in items:
            if item not in out:
                out.append(item)
    return out


def _merge_metrics(old: Sequence[MetricReport], new: Sequence[MetricReport]) -> list[MetricReport]:
    merged = list(new)
    seen = {(m.name, m.dataset, m.split, m.method) for m in merged}
    for metric in old:
        key = (metric.name, metric.dataset, metric.split, metric.method)
        if key not in seen:
            merged.append(metric)
            seen.add(key)
    return merged


def _merge_provenance(*groups: Sequence[Provenance]) -> list[Provenance]:
    merged: list[Provenance] = []
    seen: set[tuple[str, str | None]] = set()
    for group in groups:
        for prov in group:
            key = (prov.source, prov.locator)
            if key not in seen:
                merged.append(prov)
                seen.add(key)
    return merged


__all__ = [
    "STUB_CONFIDENCE",
    "comparison_table",
    "concept_page",
    "merge_notes",
    "note_stub_from_paper",
    "validate_note",
]
