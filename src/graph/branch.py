# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0

"""Collapsible branch structure for Monopath DAGs.

A patient trajectory is a linear spine of edges. Layered on top are *branches*:
ephemeral excursions that open when an objective value moves (e.g. heart rate
70 -> 110) and close when a downstream edge inverts that move (110 -> 75). Matched
open/close pairs behave like balanced brackets and can be *collapsed* to isolate
transient perturbations from persistent state change.

This module owns the **structure only** — the stack-based bracket matching and the
`BranchSpan` bookkeeping. It has no LLM / DSPy dependency and is fully
deterministic: the two *semantic* decisions (does this edge open an excursion? does
this edge invert the open one?) are injected as plain callables, so the caller wires
in the LLM while this logic stays reproducible and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# An edge is a plain JSON-serialisable dict (as produced by the DSPy pipeline).
EdgeDict = Dict[str, Any]

# Semantic predicates, injected by the caller (LLM-backed in the pipeline).
OpeningPredicate = Callable[[EdgeDict], bool]
InversePredicate = Callable[[EdgeDict, EdgeDict], bool]


@dataclass
class BranchSpan:
    """A single collapsible excursion: an opening edge and its inverse closer.

    `close_edge_id` is None while the excursion is unresolved (opened but never
    inverted). `depth` is the nesting level at open time (outermost pair = 0).
    """

    branch_id: str
    open_edge_id: Optional[str]
    close_edge_id: Optional[str] = None
    depth: int = 0
    variable: Optional[str] = None

    @property
    def matched(self) -> bool:
        """True when the excursion was inverted/closed → collapsible."""
        return self.close_edge_id is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "open_edge_id": self.open_edge_id,
            "close_edge_id": self.close_edge_id,
            "depth": self.depth,
            "variable": self.variable,
            "matched": self.matched,
        }


def primary_variable(edge: EdgeDict) -> Optional[str]:
    """Return the first objective variable an edge changes, if any."""
    for change in edge.get("change", []) or []:
        if isinstance(change, dict) and change.get("variable"):
            return change["variable"]
    return None


# ---------------------------------------------------------------------------
# Domain gating: a branch is a *transient patient-state* excursion. Only
# reversible physiologic domains are branch-eligible; treatment, diagnosis,
# genetic and structural-imaging changes are persistent spine edges and must
# never open or close a branch. This keeps branch matching from being
# over-eager and from pairing a state change with a treatment.
# ---------------------------------------------------------------------------

# Reversible patient-state domains (a value here can move up and later back).
REVERSIBLE_STATE_DOMAINS = {
    "vital_sign", "vital_signs", "vital", "vitals",
    "lab", "labs", "laboratory",
    "symptom", "symptoms",
    "functional_status",
}
# Persistent / interventional domains — never branch.
NON_BRANCH_DOMAINS = {
    "medication", "procedure", "diagnosis", "imaging",
    "genetic", "genomic", "pathology", "administrative",
}


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def change_domain(change: Dict[str, Any]) -> str:
    return _norm(change.get("domain"))


def is_reversible_state_edge(edge: EdgeDict) -> bool:
    """True only if the edge's objective change is on a reversible patient-state
    variable (vital / lab / symptom / functional status) that genuinely moved
    (increase/decrease). Appearance/resolution and non-state domains are excluded."""
    for vc in edge.get("change", []) or []:
        if not isinstance(vc, dict):
            continue
        domain = change_domain(vc)
        direction = _norm(vc.get("direction"))
        if domain in REVERSIBLE_STATE_DOMAINS and direction in ("increase", "decrease"):
            return True
    return False


def reversible_variable(edge: EdgeDict) -> Optional[str]:
    """Name of the reversible-state variable this edge moves, if any."""
    for vc in edge.get("change", []) or []:
        if not isinstance(vc, dict):
            continue
        if (change_domain(vc) in REVERSIBLE_STATE_DOMAINS
                and _norm(vc.get("direction")) in ("increase", "decrease")):
            return vc.get("variable")
    return None


def _reversible_dir(edge: EdgeDict, variable: str) -> Optional[str]:
    for vc in edge.get("change", []) or []:
        if isinstance(vc, dict) and _norm(vc.get("variable")) == _norm(variable):
            d = _norm(vc.get("direction"))
            if d in ("increase", "decrease"):
                return d
    return None


def is_direct_state_inverse(open_edge: EdgeDict, candidate: EdgeDict) -> bool:
    """Deterministic precondition for a closing edge: it must move the SAME
    reversible-state variable in the OPPOSITE direction as the opener. This is
    the gate that keeps 'treatment' or unrelated edges from closing a branch."""
    var = reversible_variable(open_edge)
    if not var:
        return False
    if not is_reversible_state_edge(candidate):
        return False
    open_dir = _reversible_dir(open_edge, var)
    close_dir = _reversible_dir(candidate, var)
    return bool(open_dir and close_dir and {open_dir, close_dir} == {"increase", "decrease"})


def _reset_annotation(edge: EdgeDict) -> None:
    edge["branch_role"] = "none"     # "open" | "close" | "none"
    edge["branch_id"] = None
    edge["branch_depth"] = 0
    edge["branch_flag"] = False


def match_branch_structure(edges: List[EdgeDict],
                           is_opening: OpeningPredicate,
                           is_inverse: InversePredicate,
                           demote_unmatched: bool = True) -> List[BranchSpan]:
    """Annotate edges in place with a stack-nested, collapsible branch structure.

    Walks edges in narrative order maintaining a stack of open branch edges:

    - If the top-of-stack opener is inverted by the current edge (`is_inverse`),
      that edge CLOSES it (proper bracket nesting — only the top is matched) and
      the stack is popped.
    - Otherwise, if `is_opening` flags the edge as an ephemeral excursion, it OPENS
      a new branch and is pushed.
    - Everything else is a persistent, non-branch spine edge.

    Each edge gains: ``branch_flag`` (bool), ``branch_role`` ("open"/"close"/"none"),
    ``branch_id`` (shared per matched pair, else None) and ``branch_depth`` (int).
    Openers left on the stack at the end are unresolved excursions
    (``close_edge_id is None``). Returns the list of :class:`BranchSpan`.
    """
    spans: List[BranchSpan] = []
    span_by_id: Dict[str, BranchSpan] = {}
    stack: List[EdgeDict] = []
    counter = 0

    for edge in edges:
        _reset_annotation(edge)

        # 1) Does this edge close the current top-of-stack branch?
        if stack and is_inverse(stack[-1], edge):
            opener = stack.pop()
            edge["branch_role"] = "close"
            edge["branch_id"] = opener["branch_id"]
            edge["branch_depth"] = opener["branch_depth"]
            edge["branch_flag"] = True
            span = span_by_id.get(opener["branch_id"])
            if span is not None:
                span.close_edge_id = edge.get("edge_id")
            continue

        # 2) Otherwise, does it open a new (ephemeral) branch?
        if is_opening(edge):
            counter += 1
            branch_id = f"b{counter}"
            depth = len(stack)  # number of enclosing unclosed pairs
            edge["branch_role"] = "open"
            edge["branch_id"] = branch_id
            edge["branch_depth"] = depth
            edge["branch_flag"] = True
            stack.append(edge)
            span = BranchSpan(
                branch_id=branch_id,
                open_edge_id=edge.get("edge_id"),
                depth=depth,
                variable=primary_variable(edge),
            )
            spans.append(span)
            span_by_id[branch_id] = span

    if demote_unmatched:
        # A branch must be a *completed* inverse pair. Openers that were never
        # closed are demoted back to plain spine edges and their spans dropped,
        # so the output contains only direct, resolved patient-state inverses.
        matched_ids = {s.branch_id for s in spans if s.matched}
        for edge in edges:
            if edge.get("branch_role") == "open" and edge.get("branch_id") not in matched_ids:
                _reset_annotation(edge)
        spans = [s for s in spans if s.matched]

    return spans


# ---------------------------------------------------------------------------
# Reconstruction + collapse (pure): rebuild spans from already-annotated edges
# and collapse matched excursions to isolate the persistent trajectory.
# ---------------------------------------------------------------------------

def edge_endpoints(edge: EdgeDict) -> "tuple[Optional[str], Optional[str]]":
    """Split an ``X_to_Y`` edge_id into (source_node_id, target_node_id)."""
    edge_id = edge.get("edge_id") or ""
    if "_to_" in edge_id:
        src, dst = edge_id.split("_to_", 1)
        return src, dst
    return None, None


def spans_from_edges(edges: List[EdgeDict]) -> List[BranchSpan]:
    """Rebuild the BranchSpan index from edges already annotated by
    :func:`match_branch_structure` (branch_id / branch_role / branch_depth).
    Deterministic — no LLM, so the DAG layer can reconstruct structure from the
    serialized pipeline output."""
    spans: Dict[str, BranchSpan] = {}
    order: List[str] = []
    for edge in edges:
        branch_id = edge.get("branch_id")
        if not branch_id:
            continue
        role = edge.get("branch_role")
        if role == "open":
            spans[branch_id] = BranchSpan(
                branch_id=branch_id,
                open_edge_id=edge.get("edge_id"),
                depth=edge.get("branch_depth", 0),
                variable=primary_variable(edge),
            )
            order.append(branch_id)
        elif role == "close" and branch_id in spans:
            spans[branch_id].close_edge_id = edge.get("edge_id")
    return [spans[b] for b in order]


def _vc_for_variable(edge: EdgeDict, variable: Optional[str]) -> Optional[Dict[str, Any]]:
    for change in edge.get("change", []) or []:
        if isinstance(change, dict) and change.get("variable") == variable:
            return change
    return None


def net_change(span: BranchSpan, open_edge: EdgeDict,
                     close_edge: EdgeDict) -> Optional[Dict[str, Any]]:
    """Residual change on the branch variable after a matched excursion resolves
    (e.g. HR 70->110 then 110->75 nets +5). Numeric when both ends are numeric."""
    variable = span.variable
    if not variable:
        return None
    open_vc = _vc_for_variable(open_edge, variable)
    close_vc = _vc_for_variable(close_edge, variable)
    from_value = open_vc.get("from_value") if open_vc else None
    to_value = close_vc.get("to_value") if close_vc else None
    delta = None
    direction = "unchanged"
    if isinstance(from_value, (int, float)) and isinstance(to_value, (int, float)):
        delta = to_value - from_value
        direction = "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
    unit = (open_vc or close_vc or {}).get("unit")
    return {
        "variable": variable,
        "variable_cui": (open_vc or close_vc or {}).get("variable_cui"),
        "direction": direction,
        "from_value": from_value,
        "to_value": to_value,
        "delta": delta,
        "unit": unit,
    }


def _summary_edge(src: str, dst: str, span: BranchSpan,
                  open_edge: EdgeDict, close_edge: EdgeDict) -> EdgeDict:
    net = net_change(span, open_edge, close_edge)
    return {
        "edge_id": f"{src}_to_{dst}",
        "content": (f"[collapsed branch {span.branch_id}] transient "
                    f"{span.variable or 'excursion'} opened and resolved."),
        "change": [net] if net else [],
        "branch_flag": False,
        "branch_role": "collapsed",
        "branch_id": span.branch_id,
        "branch_depth": span.depth,
        "collapsed": True,
    }


def collapse_matched_spans(nodes: List[Dict[str, Any]], edges: List[EdgeDict],
                           spans: Optional[List[BranchSpan]] = None,
                           only_branch_id: Optional[str] = None):
    """Collapse matched (resolved) excursions on the linear chain.

    Each outermost matched span [open..close] is replaced by a single summary
    edge from the excursion's entry node to its exit node, and the interior
    (transient) nodes are dropped. Nested spans inside a collapsed one are removed
    with it. Unresolved openers are left intact. Returns (new_nodes, new_edges).
    """
    spans = spans if spans is not None else spans_from_edges(edges)
    edge_pos = {e.get("edge_id"): i for i, e in enumerate(edges)}
    node_pos = {n.get("node_id"): i for i, n in enumerate(nodes)}

    matched = []
    for span in spans:
        if span.close_edge_id is None:
            continue
        if only_branch_id and span.branch_id != only_branch_id:
            continue
        open_pos = edge_pos.get(span.open_edge_id)
        close_pos = edge_pos.get(span.close_edge_id)
        if open_pos is None or close_pos is None or close_pos < open_pos:
            continue
        matched.append((open_pos, close_pos, span))
    matched.sort()

    # keep only outermost matched spans (skip those nested in an already-kept one)
    selected = {}
    last_close = -1
    for open_pos, close_pos, span in matched:
        if open_pos > last_close:
            selected[open_pos] = (close_pos, span)
            last_close = close_pos

    new_edges: List[EdgeDict] = []
    removed_nodes = set()
    i = 0
    while i < len(edges):
        if i in selected:
            close_pos, span = selected[i]
            open_edge, close_edge = edges[i], edges[close_pos]
            src, _ = edge_endpoints(open_edge)
            _, dst = edge_endpoints(close_edge)
            if src in node_pos and dst in node_pos:
                for p in range(node_pos[src] + 1, node_pos[dst]):
                    removed_nodes.add(nodes[p].get("node_id"))
            new_edges.append(_summary_edge(src, dst, span, open_edge, close_edge))
            i = close_pos + 1
        else:
            new_edges.append(edges[i])
            i += 1

    new_nodes = [n for n in nodes if n.get("node_id") not in removed_nodes]
    return new_nodes, new_edges
