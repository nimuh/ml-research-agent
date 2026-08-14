"""The knowledge base: a durable, queryable, citable store of everything the
system has learned. Files on disk are the source of truth; indexes are derived
and rebuildable. Layout mirrors the kb-* skills: raw/ -> wiki/ -> reports/."""

from .chunk import chunk_sections, chunk_text, estimate_tokens
from .embed import CachedEmbedder, Embedder, HashingEmbedder, get_embedder
from .graph import KnowledgeGraph, node_id
from .health import HealthIssue, HealthReport, KnowledgeHealth
from .index import HybridIndex, load_index_ids
from .ingest import Ingestor, IngestResult
from .notes import (
    comparison_table,
    concept_page,
    merge_notes,
    note_stub_from_paper,
    validate_note,
)
from .schema import (
    SCHEMA_VERSION,
    SQL_SCHEMA,
    DocumentRecord,
    NoteFrontMatter,
    note_from_markdown,
    note_to_markdown,
)
from .search import Retriever, SearchResult, build_retriever, reciprocal_rank_fusion
from .store import KnowledgeStore, paper_content_hash

__all__ = [
    "SCHEMA_VERSION",
    "SQL_SCHEMA",
    "CachedEmbedder",
    "DocumentRecord",
    "Embedder",
    "HashingEmbedder",
    "HealthIssue",
    "HealthReport",
    "HybridIndex",
    "IngestResult",
    "Ingestor",
    "KnowledgeGraph",
    "KnowledgeHealth",
    "KnowledgeStore",
    "NoteFrontMatter",
    "Retriever",
    "SearchResult",
    "build_retriever",
    "chunk_sections",
    "chunk_text",
    "comparison_table",
    "concept_page",
    "estimate_tokens",
    "get_embedder",
    "load_index_ids",
    "merge_notes",
    "node_id",
    "note_from_markdown",
    "note_stub_from_paper",
    "note_to_markdown",
    "paper_content_hash",
    "reciprocal_rank_fusion",
    "validate_note",
]
