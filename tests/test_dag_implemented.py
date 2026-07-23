# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for DAG_Implemented graph algorithms and the GraphImplemented.edges() fix."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph.graph_protocol import DAG_Implemented, GraphImplemented


class _Node:
    def __init__(self, nid):
        self.id = nid


class _Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        self.id = f"{source.id}_to_{target.id}"


def build(edge_pairs, node_ids):
    nodes = {i: _Node(i) for i in node_ids}
    edges = [_Edge(nodes[s], nodes[t]) for s, t in edge_pairs]
    return nodes, edges


class DAGImplementedTests(unittest.TestCase):

    def setUp(self):
        # A -> B -> C -> D  with a branch B -> D
        self.nodes, self.edges = build(
            [("A", "B"), ("B", "C"), ("C", "D"), ("B", "D")], list("ABCD"))
        self.dag = DAG_Implemented(nodes=self.nodes.values(), edges=self.edges)

    def test_edges_returns_values_not_dict(self):
        # regression: GraphImplemented.edges() used to return the dict itself
        edges = list(self.dag.edges())
        self.assertEqual(len(edges), 4)
        self.assertTrue(all(hasattr(e, "source") for e in edges))

    def test_topological_sort_order(self):
        order = [n.id for n in self.dag.topological_sort()]
        self.assertEqual(order[0], "A")
        self.assertEqual(order[-1], "D")
        # every edge respects the order
        pos = {nid: i for i, nid in enumerate(order)}
        for e in self.edges:
            self.assertLess(pos[e.source.id], pos[e.target.id])

    def test_roots_and_leaves(self):
        self.assertEqual({n.id for n in self.dag.roots()}, {"A"})
        self.assertEqual({n.id for n in self.dag.leaves()}, {"D"})

    def test_descendants_and_ancestors(self):
        a = self.nodes["A"]
        d = self.nodes["D"]
        self.assertEqual({n.id for n in self.dag.descendants(a)}, {"B", "C", "D"})
        self.assertEqual({n.id for n in self.dag.ancestors(d)}, {"A", "B", "C"})

    def test_cycle_detection(self):
        nodes, edges = build([("A", "B"), ("B", "A")], list("AB"))
        dag = DAG_Implemented(nodes=nodes.values(), edges=edges)
        with self.assertRaises(ValueError):
            dag.topological_sort()


if __name__ == "__main__":
    unittest.main()
