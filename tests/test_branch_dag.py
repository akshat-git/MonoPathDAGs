# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BranchDAG: collapse, main path, branch stack, causal export."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.branch_dag import BranchDAG


def node(nid, content=""):
    return {"node_id": nid, "content": content, "clinical_data": {}}


def hr(frm, to):
    return [{"variable": "heart rate", "variable_cui": "C0018810",
             "direction": "increase" if to > frm else "decrease",
             "from_value": frm, "to_value": to, "delta": to - frm, "unit": "bpm"}]


def edge(eid, role="none", bid=None, depth=0, vc=None):
    return {"edge_id": eid, "content": eid, "change": vc or [],
            "branch_flag": role in ("open", "close"),
            "branch_role": role, "branch_id": bid, "branch_depth": depth}


class BranchDAGTests(unittest.TestCase):

    def _simple(self):
        # A -b1open-> B --> C -b1close-> D --> E   (HR 70->110 ... 110->75)
        nodes = [node(x) for x in "ABCDE"]
        edges = [
            edge("A_to_B", role="open", bid="b1", depth=0, vc=hr(70, 110)),
            edge("B_to_C"),
            edge("C_to_D", role="close", bid="b1", depth=0, vc=hr(110, 75)),
            edge("D_to_E"),
        ]
        return BranchDAG.from_pipeline(nodes, edges)

    def test_roots_leaves_order(self):
        dag = self._simple()
        self.assertEqual(dag.node_order(), list("ABCDE"))
        self.assertEqual(dag.roots(), ["A"])
        self.assertEqual(dag.leaves(), ["E"])

    def test_collapse_all_removes_interior(self):
        collapsed = self._simple().collapse_all()
        self.assertEqual(collapsed.node_order(), ["A", "D", "E"])
        ids = [e["edge_id"] for e in collapsed.edges]
        self.assertEqual(ids, ["A_to_D", "D_to_E"])
        summary = collapsed.edges[0]
        self.assertTrue(summary["collapsed"])
        # net residual: HR 70 -> 75 = +5
        net = summary["change"][0]
        self.assertEqual(net["from_value"], 70)
        self.assertEqual(net["to_value"], 75)
        self.assertEqual(net["delta"], 5)

    def test_branch_stack_flat(self):
        stack = self._simple().branch_stack()
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack[0]["branch_id"], "b1")
        self.assertEqual(stack[0]["children"], [])

    def test_causal_export_shape(self):
        exp = self._simple().to_causal_export()
        self.assertEqual(len(exp["nodes"]), 5)
        self.assertEqual(len(exp["edges"]), 4)
        self.assertEqual([b["branch_id"] for b in exp["branches"]], ["b1"])
        self.assertEqual(exp["spine"]["nodes"], ["A", "D", "E"])
        self.assertTrue(exp["branches"][0]["matched"])

    def test_nested_branches_collapse_together(self):
        # b2 nested inside b1: A(b1 open) B(b2 open) C(b2 close) D(b1 close) E
        nodes = [node(x) for x in "ABCDE"]
        edges = [
            edge("A_to_B", role="open", bid="b1", depth=0, vc=hr(70, 110)),
            edge("B_to_C", role="open", bid="b2", depth=1, vc=hr(110, 130)),
            edge("C_to_D", role="close", bid="b2", depth=1, vc=hr(130, 108)),
            edge("D_to_E", role="close", bid="b1", depth=0, vc=hr(108, 72)),
        ]
        dag = BranchDAG.from_pipeline(nodes, edges)

        stack = dag.branch_stack()
        self.assertEqual(len(stack), 1)
        self.assertEqual(stack[0]["branch_id"], "b1")
        self.assertEqual([c["branch_id"] for c in stack[0]["children"]], ["b2"])

        collapsed = dag.collapse_all()
        self.assertEqual(collapsed.node_order(), ["A", "E"])
        self.assertEqual([e["edge_id"] for e in collapsed.edges], ["A_to_E"])

    def test_unresolved_branch_not_collapsed(self):
        # b1 opens but never closes -> persistent, must survive collapse
        nodes = [node(x) for x in "ABC"]
        edges = [
            edge("A_to_B", role="open", bid="b1", depth=0, vc=hr(70, 110)),
            edge("B_to_C"),
        ]
        collapsed = BranchDAG.from_pipeline(nodes, edges).collapse_all()
        self.assertEqual(collapsed.node_order(), list("ABC"))
        self.assertEqual([e["edge_id"] for e in collapsed.edges], ["A_to_B", "B_to_C"])


if __name__ == "__main__":
    unittest.main()
