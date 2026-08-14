"""Hybrid index: vector store + BM25 + metadata filters. Rebuildable from
disk; supports incremental upserts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rank_bm25 import BM25Okapi

from ..config import KnowledgeConfig
from ..types import Passage
from ..utils.io import ensure_dir, read_json, write_json
from .embed import Embedder

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from .store import KnowledgeStore

# The plan left the vector store open (LanceDB / Chroma / pgvector). Settled:
# a numpy matrix plus a json sidecar, persisted under paths.kb_index. At the
# scale this KB targets (tens of thousands of passages) a brute-force dot
# product is fast, it adds no service and no wheel to install, and -- the reason
# that actually decides it -- the whole index is trivially rebuildable from the
# files that are truth, which is the invariant a vector database would obscure.
VECTORS_FILE = "vectors.npy"
IDS_FILE = "ids.json"
RECORDS_FILE = "records.json"
MANIFEST_FILE = "manifest.json"

_TOKEN = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """The one tokenizer BM25 uses, shared so index and query agree."""
    return _TOKEN.findall(text.lower())


def load_index_ids(root: str | Path) -> set[str]:
    """Passage ids currently in the on-disk index, without building an embedder.

    Health checks need to compare index membership against the store; making
    them construct an embedder to answer that would be silly.
    """
    ids = read_json(Path(root) / IDS_FILE, default=[])
    return {str(i) for i in ids or []}


class HybridIndex:
    """Vector + BM25 index over passages, with metadata filters.

    Derived state only: :meth:`rebuild` reconstructs it from the store alone,
    which is the operational proof that files are truth.
    """

    def __init__(self, root: Path, embedder: Embedder, config: KnowledgeConfig) -> None:
        self.root = Path(root)
        self.embedder = embedder
        self.config = config
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self._records: dict[str, dict[str, Any]] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []

    # -- mutation ----------------------------------------------------------

    def upsert(self, passages: Sequence[Passage]) -> int:
        """Add or replace passages by id. Re-upserting identical input is a no-op."""
        if not passages:
            return 0
        vectors = self.embedder.embed([p.text for p in passages])
        appended: list[np.ndarray] = []
        for passage, vector in zip(passages, vectors, strict=True):
            self._records[passage.id] = _record(passage)
            slot = self._pos.get(passage.id)
            if slot is None:
                self._pos[passage.id] = len(self._ids)
                self._ids.append(passage.id)
                appended.append(vector)
            else:
                self._vectors[slot] = vector
        if appended:
            self._vectors = np.vstack([self._vectors, np.asarray(appended, dtype=np.float32)])
        self._bm25 = None
        return len(passages)

    def remove(self, passage_ids: Sequence[str]) -> int:
        drop = {pid for pid in passage_ids if pid in self._pos}
        if not drop:
            return 0
        keep = [i for i, pid in enumerate(self._ids) if pid not in drop]
        self._vectors = (
            self._vectors[keep] if keep else np.zeros((0, self.embedder.dim), np.float32)
        )
        self._ids = [self._ids[i] for i in keep]
        self._pos = {pid: i for i, pid in enumerate(self._ids)}
        for pid in drop:
            self._records.pop(pid, None)
        self._bm25 = None
        return len(drop)

    def clear(self) -> None:
        self._ids = []
        self._pos = {}
        self._vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._records = {}
        self._bm25 = None

    # -- query -------------------------------------------------------------

    def vector_search(
        self, query: str, k: int, *, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        if not self._ids or k <= 0:
            return []
        candidates = self._candidates(filters)
        if not candidates:
            return []
        vector = self.embedder.embed([query])[0]
        scores = self._vectors[candidates] @ vector
        ranked = sorted(
            ((self._ids[idx], float(score)) for idx, score in zip(candidates, scores, strict=True)),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:k]

    def keyword_search(
        self, query: str, k: int, *, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        tokens = tokenize(query)
        if not self._ids or k <= 0 or not tokens:
            return []
        if filters:
            ids = [self._ids[i] for i in self._candidates(filters)]
            if not ids:
                return []
            bm25 = BM25Okapi([tokenize(self._records[pid]["text"]) for pid in ids])
        else:
            bm25, ids = self._full_bm25()
        scores = bm25.get_scores(tokens)
        ranked = sorted(
            ((pid, float(score)) for pid, score in zip(ids, scores, strict=True) if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:k]

    def get_record(self, passage_id: str) -> dict[str, Any] | None:
        return self._records.get(passage_id)

    def _full_bm25(self) -> tuple[BM25Okapi, list[str]]:
        if self._bm25 is None:
            self._bm25_ids = list(self._ids)
            corpus = [tokenize(self._records[pid]["text"]) for pid in self._bm25_ids]
            self._bm25 = BM25Okapi(corpus or [[""]])
        return self._bm25, self._bm25_ids

    def _candidates(self, filters: dict[str, Any] | None) -> list[int]:
        if not filters:
            return list(range(len(self._ids)))
        out = []
        for idx, pid in enumerate(self._ids):
            if _matches(self._records[pid], filters):
                out.append(idx)
        return out

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        ensure_dir(self.root)
        np.save(self.root / VECTORS_FILE, self._vectors)
        write_json(self.root / IDS_FILE, self._ids)
        write_json(self.root / RECORDS_FILE, self._records)
        write_json(
            self.root / MANIFEST_FILE,
            {
                "dim": int(self.embedder.dim),
                "model": getattr(self.embedder, "model_name", "unknown"),
                "size": len(self._ids),
            },
        )

    def load(self) -> None:
        """Load from disk; an absent or mismatched index simply starts empty."""
        vectors_path = self.root / VECTORS_FILE
        ids = read_json(self.root / IDS_FILE, default=None)
        if not vectors_path.exists() or ids is None:
            self.clear()
            return
        vectors = np.load(vectors_path).astype(np.float32)
        records = read_json(self.root / RECORDS_FILE, default={}) or {}
        if vectors.shape[0] != len(ids) or (vectors.size and vectors.shape[1] != self.embedder.dim):
            # A dimension or length mismatch means the index no longer matches
            # the embedder; drop it rather than serve corrupt neighbours.
            self.clear()
            return
        self._ids = [str(i) for i in ids]
        self._pos = {pid: i for i, pid in enumerate(self._ids)}
        self._vectors = vectors if vectors.size else np.zeros((0, self.embedder.dim), np.float32)
        self._records = {str(pid): dict(rec) for pid, rec in records.items()}
        self._bm25 = None

    def rebuild(self, store: KnowledgeStore) -> int:
        """Throw the index away and rebuild it from the store. Returns passage count."""
        self.clear()
        passages = store.list_passages()
        self.upsert(passages)
        self.save()
        return len(passages)

    @property
    def size(self) -> int:
        return len(self._ids)


def _record(passage: Passage) -> dict[str, Any]:
    return {
        "doc_id": passage.doc_id,
        "text": passage.text,
        "section": passage.section,
        "page": passage.page,
        "order": passage.order,
        "meta": dict(passage.metadata),
    }


def _matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    for field, wanted in filters.items():
        value = record.get(field, record.get("meta", {}).get(field))
        if isinstance(wanted, list | tuple | set):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


__all__ = ["HybridIndex", "load_index_ids", "tokenize"]
