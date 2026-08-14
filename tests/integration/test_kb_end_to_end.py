"""End-to-end KB: ingest a small corpus, retrieve cited passages, rebuild the
derived indexes from disk, and confirm nothing about retrieval changed."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ml_research_agent.config import Config, KnowledgeConfig, PathsConfig
from ml_research_agent.knowledge.embed import get_embedder
from ml_research_agent.knowledge.graph import KnowledgeGraph
from ml_research_agent.knowledge.health import KnowledgeHealth
from ml_research_agent.knowledge.index import HybridIndex, load_index_ids
from ml_research_agent.knowledge.ingest import Ingestor
from ml_research_agent.knowledge.notes import validate_note
from ml_research_agent.knowledge.search import Retriever, build_retriever
from ml_research_agent.knowledge.store import KnowledgeStore
from ml_research_agent.types import Author, Paper, PaperSection

CORPUS: list[tuple[str, str, str, list[tuple[str, str]]]] = [
    (
        "2101.00001",
        "Sparse attention for long context transformers",
        "Ada Lovelace",
        [
            ("Method", "A sparse attention mask keeps a local window plus a few global tokens."),
            ("Results", "Sparse attention matches dense attention on the long range arena."),
        ],
    ),
    (
        "2102.00002",
        "Curriculum learning for machine translation",
        "Marie Curie",
        [
            ("Method", "A difficulty score orders sentence pairs from easy to hard."),
            ("Results", "BLEU on WMT14 English German improves by 0.6 points."),
        ],
    ),
    (
        "2103.00003",
        "Quantization aware training for edge inference",
        "Grace Hopper",
        [
            ("Method", "Weights and activations are quantized to eight bits during training."),
            ("Results", "ImageNet top one accuracy drops 0.3 points while latency halves."),
        ],
    ),
    (
        "2104.00004",
        "Retrieval augmented generation for question answering",
        "Alan Turing",
        [
            ("Method", "A dense retriever selects passages from a wikipedia index."),
            ("Results", "Exact match on natural questions rises from 32.1 to 41.5."),
        ],
    ),
    (
        "2105.00005",
        "Scaling laws for sparse mixture of experts",
        "Katherine Johnson",
        [
            ("Method", "Mixture of experts layers route each token to two of sixty four experts."),
            (
                "Results",
                "At fixed compute the model reaches the loss of a dense model four times its size.",
            ),
        ],
    ),
]

QUERIES = [
    "sparse attention long range arena",
    "curriculum difficulty sentence pairs",
    "eight bit quantization edge latency",
    "dense retriever wikipedia passages",
    "mixture of experts routing tokens",
]


def _config(tmp_path: Path) -> Config:
    workspace = tmp_path / "workspace"
    config = Config(
        paths=PathsConfig(
            workspace=workspace,
            kb_raw=workspace / "kb" / "raw",
            kb_wiki=workspace / "kb" / "wiki",
            kb_reports=workspace / "kb" / "reports",
            runs=workspace / "runs",
            repos=workspace / "repos",
            cache=workspace / "cache",
        ),
        knowledge=KnowledgeConfig(chunk_tokens=64, chunk_overlap=16),
    )
    config.ensure_paths()
    return config


def _papers() -> list[Paper]:
    return [
        Paper(
            title=title,
            abstract=f"{title}. " + " ".join(text for _, text in sections),
            arxiv_id=arxiv_id,
            year=2020 + i,
            authors=[Author(name=author)],
            sections=[
                PaperSection(title=name, text=text, order=j, page_start=j + 1)
                for j, (name, text) in enumerate(sections)
            ],
        )
        for i, (arxiv_id, title, author, sections) in enumerate(CORPUS)
    ]


def _open(config: Config) -> tuple[KnowledgeStore, HybridIndex]:
    store = KnowledgeStore(config)
    index = HybridIndex(config.paths.kb_index, get_embedder(config), config.knowledge)
    index.load()
    return store, index


def _snapshot(retriever: Retriever) -> dict[str, list[tuple[str, str, float]]]:
    """Retrieval behaviour, in a form two runs can be compared on exactly."""
    return {
        query: [
            (hit.passage.id, hit.citation, round(hit.score, 9))
            for hit in retriever.search(query, k=5).hits
        ]
        for query in QUERIES
    }


@pytest.fixture
def ingested(tmp_path) -> Any:
    config = _config(tmp_path)
    store, index = _open(config)
    ingestor = Ingestor(config, store, index)
    results = [ingestor.ingest_paper(paper) for paper in _papers()]
    yield config, store, index, ingestor, results
    store.close()


def test_ingest_builds_a_searchable_kb(ingested):
    config, store, index, _ingestor, results = ingested
    assert len(results) == 5
    assert not any(r.skipped for r in results)
    assert all(r.passages > 0 for r in results)
    assert len(store.list_papers()) == 5
    assert index.size == len(store.list_passages()) > 5
    assert store.stats()["documents_by_kind"] == {"paper": 5}


def test_every_note_stub_is_traceable(ingested):
    config, _store, _index, _ingestor, results = ingested
    for result in results:
        stub = result.note_stub
        assert stub is not None
        assert validate_note(stub, config.knowledge) == []


def test_re_ingesting_the_same_corpus_changes_nothing(ingested):
    config, store, index, ingestor, _results = ingested
    before = store.stats()

    again = [ingestor.ingest_paper(paper) for paper in _papers()]
    assert all(r.skipped for r in again)
    assert all("already ingested" in r.reason for r in again)
    assert store.stats()["documents"] == before["documents"]
    assert store.stats()["passages"] == before["passages"]
    assert index.size == before["passages"]


def test_a_revised_paper_replaces_its_passages(ingested):
    config, store, index, ingestor, _results = ingested
    paper = _papers()[0]
    revised = paper.model_copy(
        update={
            "sections": [
                *paper.sections,
                PaperSection(
                    title="Appendix", text="Extra ablations over the window size.", order=2
                ),
            ]
        }
    )
    result = ingestor.ingest_paper(revised)
    assert not result.skipped
    assert len(store.list_papers()) == 5
    doc_id = store.document_by_key("paper", paper.key).id
    assert {p.section for p in store.list_passages(doc_id=doc_id)} >= {"Appendix"}
    assert index.size == len(store.list_passages())


def test_a_revision_that_drops_text_takes_its_passages_with_it(ingested):
    # The other half of "a revision replaces its passages": chunks that vanished
    # must leave the store and the index together, or retrieval keeps citing
    # text the KB no longer holds and no health check ever notices.
    config, store, index, ingestor, _results = ingested
    paper = _papers()[1]
    doc_id = store.document_by_key("paper", paper.key).id
    before = {p.id for p in store.list_passages(doc_id=doc_id)}

    result = ingestor.ingest_paper(paper.model_copy(update={"sections": [paper.sections[0]]}))
    assert not result.skipped
    after = {p.id for p in store.list_passages(doc_id=doc_id)}
    dropped = before - after
    assert dropped and after < before

    assert not (dropped & load_index_ids(index.root))
    assert index.size == len(store.list_passages())
    drift = [i for i in KnowledgeHealth(config, store).check().issues if i.kind == "index_drift"]
    assert drift == []

    hits = Retriever(config, store, index).search("BLEU WMT14 English German", k=5).hits
    assert all(hit.passage.id not in dropped for hit in hits)


def test_search_returns_passages_that_can_be_cited(ingested):
    config, store, index, _ingestor, _results = ingested
    retriever = Retriever(config, store, index)
    for query in QUERIES:
        result = retriever.search(query, k=3)
        assert result.hits, query
        top = result.hits[0]
        assert top.citation
        assert store.get_passage(top.passage.id) == top.passage
        assert top.passage.section and top.passage.page


def test_a_paper_without_sections_is_ingested_abstract_only_and_says_so(tmp_path):
    config = _config(tmp_path)
    store, index = _open(config)
    ingestor = Ingestor(config, store, index)
    paper = _papers()[0].model_copy(update={"sections": []})

    result = ingestor.ingest_paper(paper)
    assert not result.skipped
    assert "abstract-only" in result.reason
    assert {p.section for p in store.list_passages(doc_id=result.doc_id)} == {"Abstract"}
    store.close()


def test_raw_text_files_are_ingested_and_binaries_are_reported(tmp_path):
    config = _config(tmp_path)
    store, index = _open(config)
    ingestor = Ingestor(config, store, index)
    (config.paths.kb_raw / "notes.md").write_text(
        "# Lab notebook\n\nThe smoke run diverged after twelve steps.", encoding="utf-8"
    )
    (config.paths.kb_raw / "paper.pdf").write_bytes(b"%PDF-1.7 binary")

    results = ingestor.ingest_raw_dir()
    assert len(results) == 2
    ingested = [r for r in results if not r.skipped]
    assert len(ingested) == 1
    assert store.get_document(ingested[0].doc_id).title == "Lab notebook"
    assert any("not parsed in knowledge/" in r.reason for r in results if r.skipped)

    assert all(r.skipped for r in ingestor.ingest_raw_dir())
    store.close()


def test_rebuilding_the_indexes_from_disk_changes_no_result(ingested):
    config, store, index, ingestor, _results = ingested
    retriever = Retriever(config, store, index)
    before = _snapshot(retriever)

    count = ingestor.rebuild_indexes()
    assert count == len(store.list_passages())
    assert _snapshot(Retriever(config, store, index)) == before


def test_a_fresh_process_reads_the_same_kb_off_disk(ingested):
    config, store, index, _ingestor, _results = ingested
    before = _snapshot(Retriever(config, store, index))
    store.close()

    reopened = build_retriever(config)
    assert reopened.index.size == index.size
    assert _snapshot(reopened) == before
    reopened.store.close()


def test_deleting_the_index_loses_nothing_that_matters(ingested):
    config, store, index, _ingestor, _results = ingested
    before = _snapshot(Retriever(config, store, index))
    for path in Path(config.paths.kb_index).glob("*"):
        path.unlink()

    fresh = HybridIndex(config.paths.kb_index, get_embedder(config), config.knowledge)
    fresh.load()
    assert fresh.size == 0
    assert fresh.rebuild(store) == len(store.list_passages())
    assert _snapshot(Retriever(config, store, index=fresh)) == before


def test_the_kb_is_healthy_and_the_graph_finds_gaps(ingested):
    config, store, _index, _ingestor, results = ingested
    for result in results:
        stub = result.note_stub
        assert stub is not None
        store.save_note(
            stub.model_copy(
                update={
                    "method": stub.title.split()[0].lower(),
                    "datasets": ["imagenet"] if "Quantization" in stub.title else ["wikitext"],
                }
            )
        )

    graph = KnowledgeGraph(store)
    report = KnowledgeHealth(config, store, graph).check()
    assert report.errors == []
    assert graph.gaps(), "five methods over two datasets must leave holes"
    assert report.stats["notes"] == 5
