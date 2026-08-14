"""Shared factories for the literature tests.

Kept as a plain module (mirroring ``kb_helpers``) rather than a conftest so the
factories can also be imported by the integration cassette tests. Nothing here
touches the network, the clock, or the real ``workspace/``.
"""

from __future__ import annotations

from typing import Any

from ml_research_agent.types import (
    Author,
    FalsifiableClaim,
    Paper,
    ResearchBrief,
    Venue,
)


def make_paper(**kw: Any) -> Paper:
    """A Paper with harmless defaults; every field is overridable by keyword."""
    fields: dict[str, Any] = {"title": "An untitled candidate paper"}
    fields.update(kw)
    if isinstance(fields.get("authors"), list) and fields["authors"]:
        fields["authors"] = [Author(name=a) if isinstance(a, str) else a for a in fields["authors"]]
    if isinstance(fields.get("venue"), str):
        fields["venue"] = Venue(name=fields["venue"])
    return Paper(**fields)


def make_brief(**kw: Any) -> ResearchBrief:
    """The sparse-attention brief every ranking/query test scores against."""
    fields: dict[str, Any] = {
        "idea_id": "idea_test",
        "title": "Sparse attention for long context transformers",
        "problem_statement": (
            "Dense attention is quadratic in sequence length, which makes long context "
            "training expensive."
        ),
        "claims": [
            FalsifiableClaim(
                statement="Block sparse attention matches dense perplexity at 32k context",
                falsifier="The perplexity gap exceeds 0.2 nats",
            )
        ],
        "search_terms": ["block sparse attention", "long context"],
    }
    fields.update(kw)
    return ResearchBrief(**fields)


def on_topic_paper(index: int = 0, **kw: Any) -> Paper:
    """A paper that scores well above ``snowball_min_score`` against ``make_brief``.

    ``year=None`` on purpose: the recency and velocity components then take
    their unknown-year branches, so the score does not depend on today's date.
    """
    fields: dict[str, Any] = {
        "title": f"Block sparse attention for long context transformers, part {index}",
        "abstract": (
            "We show block sparse attention matches dense perplexity at 32k context length "
            "while cutting the quadratic cost of long context training."
        ),
        "arxiv_id": f"2401.1{index:04d}",
        "year": None,
        "citation_count": 300,
        "venue": Venue(name="ICLR", kind="conference"),
        "code_urls": ["https://github.com/example/sparse"],
    }
    fields.update(kw)
    return Paper(**fields)


def off_topic_paper(index: int = 0, **kw: Any) -> Paper:
    """A paper that scores well below ``snowball_min_score`` against ``make_brief``."""
    fields: dict[str, Any] = {
        "title": f"Protein folding from crystallography priors, volume {index}",
        "abstract": "We predict tertiary structure from amino acid sequence alone.",
        "arxiv_id": f"2402.2{index:04d}",
        "year": None,
        "citation_count": 0,
    }
    fields.update(kw)
    return Paper(**fields)


class RecordingLogger:
    """A logger that keeps structured events instead of writing them anywhere.

    Accepts ``**fields`` so ``literature/_log.py`` takes its structured branch
    rather than folding the fields into a message string.
    """

    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str, event: str, **fields: Any) -> None:
        self.records.append((level, event, fields))

    def debug(self, event: str, **fields: Any) -> None:
        self._record("debug", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._record("info", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._record("warning", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._record("error", event, **fields)

    def events(self, name: str) -> list[dict[str, Any]]:
        return [fields for _level, event, fields in self.records if event == name]

    @property
    def names(self) -> list[str]:
        return [event for _level, event, _fields in self.records]


__all__ = [
    "RecordingLogger",
    "make_brief",
    "make_paper",
    "off_topic_paper",
    "on_topic_paper",
]
