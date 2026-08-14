"""Knowledge graph: papers, methods, datasets, metrics, repos, claims and their
edges (cites, uses, contradicts, reproduces). Powers gap-finding and lineage."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import Any

import networkx as nx

from ..types import ClaimRelation, GraphEdge, Note
from .store import KnowledgeStore

# Node ids are namespaced so a dataset called "transformer" and a method called
# "transformer" are two nodes, not one silently merged one.
NodeKind = str
AXES = ("method", "dataset", "metric", "paper", "note")


def node_id(kind: NodeKind, name: str) -> str:
    return f"{kind}:{name}"


class KnowledgeGraph:
    """The KB's relational view, built on demand from the store.

    Derived like every other index: nothing is stored here that could not be
    rebuilt from notes, claims and edges.
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self._graph: nx.MultiDiGraph | None = None

    # -- construction ------------------------------------------------------

    def build(self) -> nx.MultiDiGraph:
        """(Re)build the graph from notes, claims, papers and stored edges."""
        graph: nx.MultiDiGraph = nx.MultiDiGraph()

        for paper in self.store.list_papers():
            pid = node_id("paper", paper.key)
            graph.add_node(pid, kind="paper", label=paper.title, year=paper.year)
            for ref in paper.references:
                target = node_id("paper", ref)
                if target not in graph:
                    graph.add_node(target, kind="paper", label=ref)
                graph.add_edge(
                    pid, target, key=ClaimRelation.CITES.value, relation=ClaimRelation.CITES
                )

        notes = self.store.list_notes()
        for note in notes:
            nid = node_id("note", note.id)
            graph.add_node(nid, kind="note", label=note.title, confidence=note.confidence)
            if note.paper_key:
                pid = node_id("paper", note.paper_key)
                if pid not in graph:
                    graph.add_node(pid, kind="paper", label=note.title)
                _link(graph, nid, pid, ClaimRelation.CITES)
            for kind, name in _note_entities(note):
                target = node_id(kind, name)
                if target not in graph:
                    graph.add_node(target, kind=kind, label=name)
                _link(graph, nid, target, ClaimRelation.USES)
            if note.supersedes:
                _link(graph, nid, node_id("note", note.supersedes), ClaimRelation.EXTENDS)
            for claim_id in note.claims:
                _link(graph, nid, node_id("claim", claim_id), ClaimRelation.SUPPORTS)

        for claim in self.store.list_claims():
            cid = node_id("claim", claim.id)
            graph.add_node(cid, kind="claim", label=claim.statement, confidence=claim.confidence)
            for other in claim.contradicts:
                target = node_id("claim", other)
                if target not in graph:
                    graph.add_node(target, kind="claim", label=other)
                _link(graph, cid, target, ClaimRelation.CONTRADICTS)
            for other in claim.supports_claims:
                target = node_id("claim", other)
                if target not in graph:
                    graph.add_node(target, kind="claim", label=other)
                _link(graph, cid, target, ClaimRelation.SUPPORTS)

        for edge in self.store.list_edges():
            for endpoint in (edge.source, edge.target):
                if endpoint not in graph:
                    graph.add_node(endpoint, kind=_kind_of(endpoint), label=endpoint)
            _link(graph, edge.source, edge.target, edge.relation, weight=edge.weight)

        self._graph = graph
        return graph

    @property
    def graph(self) -> nx.MultiDiGraph:
        if self._graph is None:
            return self.build()
        return self._graph

    def add_edge(self, edge: GraphEdge) -> None:
        """Persist an edge and reflect it in the in-memory graph."""
        self.store.add_edge(edge)
        graph = self.graph
        for endpoint in (edge.source, edge.target):
            if endpoint not in graph:
                graph.add_node(endpoint, kind=_kind_of(endpoint), label=endpoint)
        _link(graph, edge.source, edge.target, edge.relation, weight=edge.weight)

    # -- queries -----------------------------------------------------------

    def neighbors(self, node_id: str, *, relation: ClaimRelation | None = None) -> list[str]:
        """Nodes related to this one, in either direction.

        Direction matters for interpretation but rarely for the question being
        asked ("what is this connected to?"), so both are returned, deduped and
        ordered.
        """
        graph = self.graph
        if node_id not in graph:
            return []
        out: list[str] = []
        for source, target, data in graph.out_edges(node_id, data=True):
            if relation is None or data.get("relation") is relation:
                _append(out, target if source == node_id else source)
        for source, target, data in graph.in_edges(node_id, data=True):
            if relation is None or data.get("relation") is relation:
                _append(out, source if target == node_id else target)
        return out

    def contradictions(self) -> list[tuple[str, str]]:
        """Every contested pair -- each one a candidate replication."""
        pairs: set[tuple[str, str]] = set()
        for source, target, data in self.graph.edges(data=True):
            if data.get("relation") is ClaimRelation.CONTRADICTS:
                pairs.add((source, target) if source <= target else (target, source))
        return sorted(pairs)

    def coverage_matrix(
        self, *, rows: str = "method", cols: str = "dataset"
    ) -> dict[tuple[str, str], int]:
        """How many notes cover each ``(row, col)`` cell. Holes are the point."""
        matrix: dict[tuple[str, str], int] = {}
        for note in self.store.list_notes():
            for row, col in product(_axis(note, rows), _axis(note, cols)):
                matrix[(row, col)] = matrix.get((row, col), 0) + 1
        return matrix

    def gaps(self, *, min_support: int = 1) -> list[dict[str, Any]]:
        """Cells the literature does not cover: candidate experiments."""
        return self.gaps_for(rows="method", cols="dataset", min_support=min_support)

    def gaps_for(
        self, *, rows: str = "method", cols: str = "dataset", min_support: int = 1
    ) -> list[dict[str, Any]]:
        matrix = self.coverage_matrix(rows=rows, cols=cols)
        row_values = sorted({r for r, _ in matrix})
        col_values = sorted({c for _, c in matrix})
        out: list[dict[str, Any]] = []
        for row, col in product(row_values, col_values):
            support = matrix.get((row, col), 0)
            if support < min_support:
                out.append(
                    {
                        "rows": rows,
                        "row": row,
                        "cols": cols,
                        "col": col,
                        "support": support,
                        "reason": f"no note reports {row} on {col}",
                    }
                )
        return out

    def lineage(self, node_id: str) -> list[str]:
        """What this node builds on: supersession, extension, reproduction, citation."""
        graph = self.graph
        if node_id not in graph:
            return []
        follow = {ClaimRelation.EXTENDS, ClaimRelation.REPRODUCES, ClaimRelation.CITES}
        order: list[str] = [node_id]
        frontier = [node_id]
        seen = {node_id}
        while frontier:
            current = frontier.pop(0)
            for _source, target, data in sorted(graph.out_edges(current, data=True)):
                if data.get("relation") in follow and target not in seen:
                    seen.add(target)
                    order.append(target)
                    frontier.append(target)
        return order

    def to_dict(self) -> dict[str, Any]:
        graph = self.graph
        return {
            "nodes": [
                {"id": n, "kind": d.get("kind", _kind_of(n)), "label": d.get("label", n)}
                for n, d in sorted(graph.nodes(data=True))
            ],
            "edges": [
                {
                    "source": s,
                    "target": t,
                    "relation": (
                        d["relation"].value if isinstance(d.get("relation"), ClaimRelation) else k
                    ),
                    "weight": d.get("weight", 1.0),
                }
                for s, t, k, d in sorted(graph.edges(keys=True, data=True), key=lambda e: e[:3])
            ],
            "stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
                "contradictions": len(self.contradictions()),
            },
        }


def _link(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
    relation: ClaimRelation,
    *,
    weight: float = 1.0,
) -> None:
    # The relation is the edge key, so re-adding the same typed edge updates it
    # instead of piling up parallel duplicates.
    graph.add_edge(source, target, key=relation.value, relation=relation, weight=weight)


def _note_entities(note: Note) -> Iterable[tuple[str, str]]:
    if note.method:
        yield ("method", note.method)
    for dataset in note.datasets:
        yield ("dataset", dataset)
    for metric in note.metrics:
        yield ("metric", metric.name)


def _axis(note: Note, axis: str) -> list[str]:
    match axis:
        case "method":
            return [note.method] if note.method else []
        case "dataset":
            return list(note.datasets)
        case "metric":
            return [m.name for m in note.metrics]
        case "paper":
            return [note.paper_key] if note.paper_key else []
        case "note":
            return [note.id]
        case _:
            raise ValueError(f"unknown coverage axis {axis!r}; expected one of {AXES}")


def _kind_of(node: str) -> str:
    return node.split(":", 1)[0] if ":" in node else "unknown"


def _append(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


__all__ = ["AXES", "KnowledgeGraph", "node_id"]
