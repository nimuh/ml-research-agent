"""Section-aware chunking with overlap, preserving figure/table context and
enough locators (paper id, section, page) to cite precisely."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..config import KnowledgeConfig
from ..types import PaperSection, Passage
from ..utils.hashing import hash_obj, hash_text

# ~4 characters per token is the usual English rule of thumb for BPE vocabularies.
# It is deliberately cheap: chunking runs over every ingested paper, and a
# tokenizer dependency here would tie the KB to one model family.
_CHARS_PER_TOKEN = 4

_CAPTION = re.compile(r"^\s*(figure|fig\.?|table|algorithm|alg\.?|listing)\s*\d+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_PARAGRAPH = re.compile(r"\n\s*\n")
_WHITESPACE = re.compile(r"[ \t]+")


def estimate_tokens(text: str) -> int:
    """Approximate token count at ~4 characters per token (see ``_CHARS_PER_TOKEN``).

    Always rounds up, so a non-empty string never costs zero and a budget is
    never overrun by rounding.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, -(-len(stripped) // _CHARS_PER_TOKEN))


def chunk_sections(
    sections: Sequence[PaperSection], doc_id: str, config: KnowledgeConfig
) -> list[Passage]:
    """Chunk a parsed paper section by section.

    A chunk never spans two sections: crossing that boundary would make the
    passage's locator a lie, and the citation built from it unverifiable.
    """
    passages: list[Passage] = []
    order = 0
    for section in sorted(sections, key=lambda s: (s.order, s.title)):
        for body in _pack(_units(section.text), config):
            passages.append(
                _passage(
                    body,
                    doc_id=doc_id,
                    section=section.title,
                    page=section.page_start,
                    order=order,
                    metadata={"section_order": section.order, "section_level": section.level},
                )
            )
            order += 1
    return passages


def chunk_text(
    text: str,
    doc_id: str,
    config: KnowledgeConfig,
    *,
    section: str | None = None,
    page: int | None = None,
) -> list[Passage]:
    """Chunk a flat block of text (abstract, README, web page) into passages."""
    return [
        _passage(body, doc_id=doc_id, section=section, page=page, order=order)
        for order, body in enumerate(_pack(_units(text), config))
    ]


def _passage(
    body: str,
    *,
    doc_id: str,
    section: str | None,
    page: int | None,
    order: int,
    metadata: dict[str, object] | None = None,
) -> Passage:
    content_hash = hash_text(body, length=32)
    meta = dict(metadata or {})
    meta["has_caption"] = any(_CAPTION.match(line) for line in body.splitlines())
    # Deterministic id: re-ingesting unchanged content must upsert the same rows
    # rather than accumulate duplicates under fresh uuids.
    identity = {"doc": doc_id, "section": section, "order": order, "hash": content_hash}
    return Passage(
        id=f"psg_{hash_obj(identity, length=12)}",
        doc_id=doc_id,
        text=body,
        section=section,
        page=page,
        order=order,
        token_count=estimate_tokens(body),
        content_hash=content_hash,
        metadata=meta,
    )


def _units(text: str) -> list[str]:
    """Split text into the atoms a chunk is built from: paragraphs, then sentences.

    A figure/table caption is glued to its neighbouring paragraph so that the
    number in the caption never ends up in a different chunk than the text that
    explains it.
    """
    units: list[str] = []
    for para in _PARAGRAPH.split(text.strip()):
        cleaned = _WHITESPACE.sub(" ", para.strip())
        if cleaned:
            units.append(cleaned)
    merged: list[str] = []
    for unit in units:
        if merged and _CAPTION.match(merged[-1]):
            merged[-1] = f"{merged[-1]}\n{unit}"
        else:
            merged.append(unit)
    if len(merged) > 1 and _CAPTION.match(merged[-1]):
        tail = merged.pop()
        merged[-1] = f"{merged[-1]}\n{tail}"
    return merged


def _sentences(unit: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_END.split(unit) if s.strip()]
    return parts or ([unit] if unit.strip() else [])


def _hard_split(unit: str, limit: int) -> list[str]:
    """Split an over-long atom on word boundaries, as a last resort."""
    words = unit.split()
    out: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        cost = estimate_tokens(word) + 1
        if current and size + cost > limit:
            out.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += cost
    if current:
        out.append(" ".join(current))
    return out


def _pack(units: Sequence[str], config: KnowledgeConfig) -> list[str]:
    """Greedily pack atoms into ``chunk_tokens`` windows with ``chunk_overlap``."""
    limit = config.chunk_tokens
    atoms: list[str] = []
    for unit in units:
        if estimate_tokens(unit) <= limit:
            atoms.append(unit)
            continue
        for sentence in _sentences(unit):
            if estimate_tokens(sentence) <= limit:
                atoms.append(sentence)
            else:
                atoms.extend(_hard_split(sentence, limit))

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for atom in atoms:
        cost = estimate_tokens(atom)
        if current and size + cost > limit:
            chunks.append("\n\n".join(current))
            current = _overlap(current, config.chunk_overlap)
            size = sum(estimate_tokens(a) for a in current)
        current.append(atom)
        size += cost
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _overlap(atoms: Sequence[str], budget: int) -> list[str]:
    """The tail of a chunk that is repeated at the head of the next one.

    Never returns the whole chunk: the next chunk must contain at least one new
    atom or packing would not terminate.
    """
    if budget <= 0 or len(atoms) < 2:
        return []
    tail: list[str] = []
    size = 0
    for atom in reversed(atoms[1:]):
        cost = estimate_tokens(atom)
        if size + cost > budget:
            break
        tail.insert(0, atom)
        size += cost
    return tail


__all__ = ["chunk_sections", "chunk_text", "estimate_tokens"]
