"""Shared fixtures for the knowledge-base tests: a Config pointed at tmp_path
and a small synthetic corpus. Nothing here touches the real workspace/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ml_research_agent.config import Config, KnowledgeConfig, PathsConfig
from ml_research_agent.types import Author, Paper, PaperSection


def kb_config(tmp_path: Path, **knowledge: Any) -> Config:
    """A Config whose every path lives under ``tmp_path``."""
    workspace = tmp_path / "workspace"
    paths = PathsConfig(
        workspace=workspace,
        kb_raw=workspace / "kb" / "raw",
        kb_wiki=workspace / "kb" / "wiki",
        kb_reports=workspace / "kb" / "reports",
        runs=workspace / "runs",
        repos=workspace / "repos",
        cache=workspace / "cache",
    )
    config = Config(paths=paths, knowledge=KnowledgeConfig(**knowledge))
    config.ensure_paths()
    return config


def make_paper(
    *,
    arxiv_id: str,
    title: str,
    abstract: str,
    sections: list[tuple[str, str]] | None = None,
    year: int = 2020,
    author: str = "Ada Lovelace",
) -> Paper:
    return Paper(
        title=title,
        abstract=abstract,
        arxiv_id=arxiv_id,
        year=year,
        authors=[Author(name=author)],
        sections=[
            PaperSection(title=name, text=text, order=i, page_start=i + 1)
            for i, (name, text) in enumerate(sections or [])
        ],
    )


def synthetic_papers() -> list[Paper]:
    """Five papers with enough distinct vocabulary to tell retrieval apart."""
    return [
        make_paper(
            arxiv_id="2101.00001",
            title="Sparse attention for long context transformers",
            abstract="We introduce a sparse attention pattern that reduces the quadratic cost.",
            sections=[
                (
                    "Method",
                    "The sparse attention mask keeps a local window and a few global tokens. "
                    "Complexity drops from quadratic to linear in sequence length.",
                ),
                (
                    "Results",
                    "On the long range arena benchmark the sparse model matches dense attention "
                    "while using 40 percent less memory.\n\n"
                    "Table 1: accuracy on long range arena.\n"
                    "Sparse 58.1, dense 58.4.",
                ),
            ],
            author="Rin Sato",
        ),
        make_paper(
            arxiv_id="2102.00002",
            title="Curriculum learning for machine translation",
            abstract="Ordering training examples by difficulty speeds up convergence.",
            sections=[
                (
                    "Method",
                    "A difficulty score orders sentence pairs, and the curriculum anneals from "
                    "easy to hard examples during training.",
                ),
                (
                    "Results",
                    "BLEU on WMT14 English German improves by 0.6 points over a shuffled "
                    "baseline at equal compute.",
                ),
            ],
            author="Marie Curie",
        ),
        make_paper(
            arxiv_id="2103.00003",
            title="Quantization aware training for edge inference",
            abstract="Eight bit quantization with straight through estimators preserves accuracy.",
            sections=[
                (
                    "Method",
                    "Weights and activations are quantized to eight bits during the forward pass "
                    "while gradients flow through a straight through estimator.",
                ),
                (
                    "Results",
                    "ImageNet top one accuracy drops by 0.3 points while inference latency on the "
                    "edge device halves.",
                ),
            ],
            author="Grace Hopper",
        ),
        make_paper(
            arxiv_id="2104.00004",
            title="Retrieval augmented generation for question answering",
            abstract="A retriever supplies passages that condition the generator.",
            sections=[
                (
                    "Method",
                    "A dense retriever selects passages from a wikipedia index and the generator "
                    "conditions on the retrieved passages.",
                ),
                (
                    "Results",
                    "Exact match on natural questions rises from 32.1 to 41.5 compared with a "
                    "closed book baseline.",
                ),
            ],
            author="Alan Turing",
        ),
        make_paper(
            arxiv_id="2105.00005",
            title="Scaling laws for sparse mixture of experts",
            abstract="Expert count trades off against dense parameter count under fixed compute.",
            sections=[
                (
                    "Method",
                    "Mixture of experts layers route each token to two experts out of sixty four.",
                ),
                (
                    "Results",
                    "At fixed training compute the mixture of experts model reaches the loss of a "
                    "dense model four times its size.",
                ),
            ],
            author="Katherine Johnson",
        ),
    ]
