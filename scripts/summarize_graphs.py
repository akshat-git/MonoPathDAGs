#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize generated Monopath DAGs.

For each graph_*.json in a directory, report node/edge/branch counts and the
collapsed-spine size (via BranchDAG). Reusable across smoke and full runs.

Usage: python scripts/summarize_graphs.py <output_dir>
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph.branch_dag import BranchDAG


def summarize(out_dir: str) -> None:
    files = sorted(glob.glob(os.path.join(out_dir, "graph_*.json")))
    if not files:
        print("No graph JSON produced.")
        return

    print(f"{'graph':<16}{'nodes':>7}{'edges':>7}{'branches':>10}{'matched':>9}{'spine':>7}")
    for path in files:
        data = json.load(open(path))
        node_dicts = [n["customData"] for n in data.get("nodes", [])]
        edge_dicts = [e["data"] for e in data.get("edges", [])]
        dag = BranchDAG.from_pipeline(node_dicts, edge_dicts)
        spine = dag.collapse_all()
        matched = sum(1 for s in dag.spans if s.matched)
        name = os.path.basename(path)
        print(f"{name:<16}{len(node_dicts):>7}{len(edge_dicts):>7}"
              f"{len(dag.spans):>10}{matched:>9}{len(spine.nodes):>7}")


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else ".")
