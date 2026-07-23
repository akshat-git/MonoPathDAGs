# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the deterministic branch-matching structure.

The LLM-backed semantic predicates are replaced here with trivial deterministic
stubs so we test only the stack/nesting/collapse logic in
``src.graph.branch.match_branch_structure``.
"""

import os
import sys
import unittest

# Make the repo root importable when running this file directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.branch import (
    BranchSpan, match_branch_structure, primary_variable,
    is_reversible_state_edge, is_direct_state_inverse,
)


def vc_edge(eid, variable, domain, direction, frm=None, to=None):
    return {"edge_id": eid, "change": [{
        "variable": variable, "domain": domain, "direction": direction,
        "from_value": frm, "to_value": to}]}


def edge(edge_id, opens=False, inverts=None, change=None):
    """Build a test edge. `opens` drives is_opening; `inverts` names the edge_id
    this edge is the inverse of (drives is_inverse)."""
    e = {"edge_id": edge_id, "_open": opens, "_inverts": inverts}
    if change is not None:
        e["change"] = change
    return e


# Injected deterministic predicates.
def is_opening(e):
    return bool(e.get("_open"))


def is_inverse(open_edge, candidate):
    return candidate.get("_inverts") == open_edge.get("edge_id")


class MatchBranchStructureTests(unittest.TestCase):

    def test_single_matched_pair(self):
        edges = [
            edge("A_to_B", opens=True),           # opens b1
            edge("B_to_C"),                        # neutral spine edge
            edge("C_to_D", inverts="A_to_B"),      # closes b1
        ]
        spans = match_branch_structure(edges, is_opening, is_inverse)

        self.assertEqual(len(spans), 1)
        self.assertEqual(edges[0]["branch_role"], "open")
        self.assertEqual(edges[0]["branch_id"], "b1")
        self.assertEqual(edges[0]["branch_depth"], 0)
        self.assertTrue(edges[0]["branch_flag"])

        self.assertEqual(edges[1]["branch_role"], "none")
        self.assertFalse(edges[1]["branch_flag"])
        self.assertIsNone(edges[1]["branch_id"])

        self.assertEqual(edges[2]["branch_role"], "close")
        self.assertEqual(edges[2]["branch_id"], "b1")
        self.assertEqual(edges[2]["branch_depth"], 0)

        self.assertTrue(spans[0].matched)
        self.assertEqual(spans[0].open_edge_id, "A_to_B")
        self.assertEqual(spans[0].close_edge_id, "C_to_D")

    def test_nested_pairs(self):
        edges = [
            edge("A_to_B", opens=True),            # opens b1 (depth 0)
            edge("B_to_C", opens=True),            # opens b2 (depth 1)
            edge("C_to_D", inverts="B_to_C"),      # closes b2
            edge("D_to_E", inverts="A_to_B"),      # closes b1
        ]
        spans = match_branch_structure(edges, is_opening, is_inverse)
        by_id = {s.branch_id: s for s in spans}

        self.assertEqual(len(spans), 2)
        self.assertEqual(by_id["b1"].depth, 0)
        self.assertEqual(by_id["b2"].depth, 1)
        self.assertTrue(by_id["b1"].matched)
        self.assertTrue(by_id["b2"].matched)
        self.assertEqual(by_id["b2"].close_edge_id, "C_to_D")
        self.assertEqual(by_id["b1"].close_edge_id, "D_to_E")
        self.assertEqual(edges[1]["branch_depth"], 1)
        self.assertEqual(edges[2]["branch_depth"], 1)

    def test_unresolved_open_demoted_by_default(self):
        # A branch must be a completed inverse pair; an opener that never closes
        # is demoted back to a plain spine edge (default demote_unmatched=True).
        edges = [
            edge("A_to_B", opens=True),            # opens b1, never inverted
            edge("B_to_C"),
        ]
        spans = match_branch_structure(edges, is_opening, is_inverse)
        self.assertEqual(spans, [])
        self.assertEqual(edges[0]["branch_role"], "none")
        self.assertFalse(edges[0]["branch_flag"])

    def test_unresolved_open_retained_when_opted_out(self):
        edges = [edge("A_to_B", opens=True), edge("B_to_C")]
        spans = match_branch_structure(edges, is_opening, is_inverse,
                                       demote_unmatched=False)
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0].matched)
        self.assertEqual(edges[0]["branch_role"], "open")

    def test_no_branches(self):
        edges = [edge("A_to_B"), edge("B_to_C")]
        spans = match_branch_structure(edges, is_opening, is_inverse)

        self.assertEqual(spans, [])
        for e in edges:
            self.assertEqual(e["branch_role"], "none")
            self.assertFalse(e["branch_flag"])

    def test_only_top_of_stack_is_matched(self):
        # e3 inverts the OUTER opener (A_to_B) but the top of stack is B_to_C.
        # Proper bracket nesting means e3 must NOT close A_to_B here.
        edges = [
            edge("A_to_B", opens=True),            # b1 (depth 0)
            edge("B_to_C", opens=True),            # b2 (depth 1) -> top of stack
            edge("C_to_D", inverts="A_to_B"),      # inverts b1, not the top
        ]
        spans = match_branch_structure(edges, is_opening, is_inverse,
                                       demote_unmatched=False)
        by_id = {s.branch_id: s for s in spans}

        self.assertFalse(by_id["b1"].matched)
        self.assertFalse(by_id["b2"].matched)
        self.assertEqual(edges[2]["branch_role"], "none")

    def test_empty_input(self):
        self.assertEqual(match_branch_structure([], is_opening, is_inverse), [])


class HelperTests(unittest.TestCase):

    def test_primary_variable(self):
        e = edge("A_to_B", change=[
            {"variable": "heart rate", "direction": "increase"},
            {"variable": "spo2", "direction": "decrease"},
        ])
        self.assertEqual(primary_variable(e), "heart rate")

    def test_primary_variable_none(self):
        self.assertIsNone(primary_variable({"edge_id": "A_to_B"}))
        self.assertIsNone(primary_variable({"change": []}))

    def test_branch_span_to_dict(self):
        span = BranchSpan(branch_id="b1", open_edge_id="A_to_B",
                          close_edge_id="C_to_D", depth=2, variable="heart rate")
        d = span.to_dict()
        self.assertEqual(d, {
            "branch_id": "b1",
            "open_edge_id": "A_to_B",
            "close_edge_id": "C_to_D",
            "depth": 2,
            "variable": "heart rate",
            "matched": True,
        })


class DomainGateTests(unittest.TestCase):

    def test_reversible_state_domains_open(self):
        for dom in ("vital_sign", "lab", "symptom", "functional_status"):
            self.assertTrue(is_reversible_state_edge(
                vc_edge("A_to_B", "x", dom, "increase")), dom)

    def test_non_reversible_domains_excluded(self):
        for dom in ("medication", "diagnosis", "genetic", "imaging", "procedure"):
            self.assertFalse(is_reversible_state_edge(
                vc_edge("A_to_B", "x", dom, "increase")), dom)

    def test_appearance_resolution_not_reversible_move(self):
        # 'new'/'resolved' are not measurable up/down moves
        self.assertFalse(is_reversible_state_edge(vc_edge("A_to_B", "x", "lab", "new")))

    def test_direct_inverse_same_variable_opposite_direction(self):
        opener = vc_edge("A_to_B", "heart rate", "vital_sign", "increase", 70, 110)
        closer = vc_edge("C_to_D", "heart rate", "vital_sign", "decrease", 110, 75)
        self.assertTrue(is_direct_state_inverse(opener, closer))

    def test_treatment_cannot_close_state_branch(self):
        opener = vc_edge("A_to_B", "heart rate", "vital_sign", "increase", 70, 110)
        drug = vc_edge("C_to_D", "beta blocker", "medication", "new")
        self.assertFalse(is_direct_state_inverse(opener, drug))

    def test_different_variable_not_inverse(self):
        opener = vc_edge("A_to_B", "heart rate", "vital_sign", "increase")
        other = vc_edge("C_to_D", "temperature", "vital_sign", "decrease")
        self.assertFalse(is_direct_state_inverse(opener, other))

    def test_same_direction_not_inverse(self):
        opener = vc_edge("A_to_B", "heart rate", "vital_sign", "increase")
        same = vc_edge("C_to_D", "heart rate", "vital_sign", "increase")
        self.assertFalse(is_direct_state_inverse(opener, same))


if __name__ == "__main__":
    unittest.main()
