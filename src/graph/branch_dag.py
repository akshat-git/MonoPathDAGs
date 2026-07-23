# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0

"""Causal-model-ready DAG view over a Monopath graph.

`BranchDAG` wraps the pipeline's ordered nodes (patient states) and edges
(objective value-changes) plus the collapsible branch spans, and exposes the
operations a downstream causal model needs:

- ``collapse_all`` / ``collapse_branch`` — replace resolved (matched) excursions
  with a single net-change summary edge, isolating persistent state change.
- ``main_path`` — the persistent spine (all transient branches collapsed).
- ``branch_stack`` — the nested (stacked) branch structure for causal reasoning.
- ``to_causal_export`` — a flat, JSON-serialisable structure keyed for modeling.

Pure Python / no LLM: structure is reconstructed from the edge annotations, so a
BranchDAG can be rebuilt deterministically from serialized pipeline output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .branch import (
    BranchSpan,
    collapse_matched_spans,
    edge_endpoints,
    spans_from_edges,
)


class BranchDAG:
    def __init__(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
                 spans: Optional[List[BranchSpan]] = None):
        self.nodes = nodes
        self.edges = edges
        self.spans = spans if spans is not None else spans_from_edges(edges)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_pipeline(cls, nodes_obj: List[Dict[str, Any]],
                      edges_obj: List[Dict[str, Any]]) -> "BranchDAG":
        """Build from the pipeline's node/edge dicts (edges already branch-annotated)."""
        return cls(list(nodes_obj), list(edges_obj))

    # -- basic topology -----------------------------------------------------
    def node_order(self) -> List[str]:
        return [n.get("node_id") for n in self.nodes]

    def roots(self) -> List[str]:
        targets = {edge_endpoints(e)[1] for e in self.edges}
        return [n for n in self.node_order() if n not in targets]

    def leaves(self) -> List[str]:
        sources = {edge_endpoints(e)[0] for e in self.edges}
        return [n for n in self.node_order() if n not in sources]

    # -- collapse -----------------------------------------------------------
    def collapse_all(self) -> "BranchDAG":
        """Collapse every matched (resolved) excursion → persistent trajectory."""
        nodes, edges = collapse_matched_spans(self.nodes, self.edges, self.spans)
        return BranchDAG(nodes, edges)

    def collapse_branch(self, branch_id: str) -> "BranchDAG":
        """Collapse only the named matched branch."""
        nodes, edges = collapse_matched_spans(
            self.nodes, self.edges, self.spans, only_branch_id=branch_id)
        return BranchDAG(nodes, edges)

    def main_path(self) -> "BranchDAG":
        """The persistent spine: alias for collapse_all()."""
        return self.collapse_all()

    # -- branch nesting (stack) --------------------------------------------
    def branch_stack(self) -> List[Dict[str, Any]]:
        """Nested branch tree by containment (a span is a child of the innermost
        span whose [open..close] edge range strictly contains it)."""
        edge_pos = {e.get("edge_id"): i for i, e in enumerate(self.edges)}

        def rng(span: BranchSpan):
            op = edge_pos.get(span.open_edge_id)
            cp = edge_pos.get(span.close_edge_id) if span.close_edge_id else op
            return op, (cp if cp is not None else op)

        # order spans by open position; build a containment forest
        ordered = sorted(
            (s for s in self.spans if edge_pos.get(s.open_edge_id) is not None),
            key=lambda s: rng(s)[0],
        )
        nodes_by_id: Dict[str, Dict[str, Any]] = {}
        for span in ordered:
            entry = span.to_dict()
            entry["children"] = []
            nodes_by_id[span.branch_id] = entry

        forest: List[Dict[str, Any]] = []
        stack: List[BranchSpan] = []
        for span in ordered:
            open_pos, close_pos = rng(span)
            while stack and rng(stack[-1])[1] < open_pos:
                stack.pop()
            if stack:
                nodes_by_id[stack[-1].branch_id]["children"].append(nodes_by_id[span.branch_id])
            else:
                forest.append(nodes_by_id[span.branch_id])
            stack.append(span)
        return forest

    # -- causal export ------------------------------------------------------
    def to_causal_export(self) -> Dict[str, Any]:
        """Flat structure for a downstream causal model: variables (nodes),
        directed value-change edges with branch labels, the branch index/stack,
        and the collapsed persistent spine."""
        spine = self.collapse_all()
        return {
            "nodes": [
                {
                    "id": n.get("node_id"),
                    "content": n.get("content"),
                    "clinical_data": n.get("clinical_data", {}),
                }
                for n in self.nodes
            ],
            "edges": [self._edge_export(e) for e in self.edges],
            "branches": [s.to_dict() for s in self.spans],
            "branch_stack": self.branch_stack(),
            "spine": {
                "nodes": spine.node_order(),
                "edges": [self._edge_export(e) for e in spine.edges],
            },
        }

    @staticmethod
    def _edge_export(edge: Dict[str, Any]) -> Dict[str, Any]:
        src, dst = edge_endpoints(edge)
        return {
            "id": edge.get("edge_id"),
            "from": src,
            "to": dst,
            "change": edge.get("change", []),
            "branch_flag": edge.get("branch_flag", False),
            "branch_role": edge.get("branch_role", "none"),
            "branch_id": edge.get("branch_id"),
            "branch_depth": edge.get("branch_depth", 0),
        }
