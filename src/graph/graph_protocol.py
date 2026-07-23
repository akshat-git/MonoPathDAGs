# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0


"""
File for some graph procotols including the DAG one, 
  

"""


from typing import Protocol, Iterable,runtime_checkable,Set,List,Optional,Union,Dict
from collections import defaultdict
from uuid import UUID
from .node.node_protocol import NodeProtocol as Node
from .edge.edge_protocol import EdgeProtocol as Edge


__all__ = [ "Graph", "DAG"]


@runtime_checkable
class Graph(Protocol):
    """Protocol representing a directed or undirected graph structure."""

    def nodes(self) -> Iterable[Node]:
        """Return all nodes in the graph."""
        raise NotImplementedError

    def edges(self) -> Iterable[Edge]:
        """Return all edges in the graph as (source, target) pairs."""
        raise NotImplementedError

    def has_node(self, node: Node) -> bool:
        """Check if the graph contains the given node."""
        raise NotImplementedError

    def has_edge(self, source: Node, target: Node) -> bool:
        """Check if an edge exists from source to target."""
        raise NotImplementedError

    def neighbors(self, node: Node) -> Iterable[Node]:
        """Return all directly connected neighbors of the given node."""
        raise NotImplementedError

class GraphImplemented(Graph):
    def __init__(self,
                 nodes: Optional[Iterable[Node]] = None,
                 edges: Optional[Iterable[Edge]] = None,
                 root: Optional[Union[int, UUID]] = None):
        self._nodes: Dict[Union[int, UUID], Node] = {}
        self._edges: Dict[Union[int, UUID], Edge] = {}
        self._adjacency: Dict[Union[int, UUID], List[Node]] = defaultdict(list)
        self._root = root

        if nodes:
            for node in nodes:
                
                self._nodes[node.id] = node

        if edges:
            for edge in edges:
                edge_id = edge.id
                self._edges[edge_id] = edge
                self._adjacency[edge.source.id].append(edge.target)

    def nodes(self) -> Iterable[Node]:
        return self._nodes.values()

    def edges(self) -> Iterable[Edge]:
        return self._edges.values()

    def has_node(self, node: Node) -> bool:
        return node.id in self._nodes

    def has_edge(self, source: Node, target: Node) -> bool:
        return target in self._adjacency.get(source.id, [])

    def neighbors(self, node: Node) -> Iterable[Node]:
        return self._adjacency.get(node.id, [])
    def get_root_id(self):
        "get the roo id"
        return self._root
    def get_root_node(self):
        " get the root node"
        return self._nodes[self._root]

class DAG(Graph, Protocol):
    """Protocol representing a directed acyclic graph (DAG).
    ALL FUNCTIONS SHOULD BE READ ONLY, GRa
    
    """

    def topological_sort(self) -> List[Node]:
        """Return nodes in a valid topological order."""
        raise NotImplementedError

    def ancestors(self, node: Node) -> Set[Node]:
        """Return all nodes with a path to the given node (its ancestors)."""
        raise NotImplementedError

    def descendants(self, node: Node) -> Set[Node]:
        """Return all nodes reachable from the given node (its descendants)."""
        raise NotImplementedError

    def roots(self) -> Set[Node]:
        """Return all nodes with no incoming edges."""
        raise NotImplementedError

    def leaves(self) -> Set[Node]:
        """Return all nodes with no outgoing edges."""
        raise NotImplementedError
    

class DAG_Implemented(GraphImplemented):
    """Concrete read-only DAG: topological ordering + ancestor/descendant queries.

    Reuses GraphImplemented's node store and adjacency (built from edge
    source/target). Works on any Node objects exposing ``.id`` and Edge objects
    exposing ``.source.id`` / ``.target.id``.
    """

    def _adjacency_ids(self) -> Dict[Union[int, UUID], List[Union[int, UUID]]]:
        adj: Dict[Union[int, UUID], List[Union[int, UUID]]] = defaultdict(list)
        for src_id, targets in self._adjacency.items():
            for target in targets:
                adj[src_id].append(target.id)
        return adj

    def _reverse_adjacency_ids(self) -> Dict[Union[int, UUID], List[Union[int, UUID]]]:
        rev: Dict[Union[int, UUID], List[Union[int, UUID]]] = defaultdict(list)
        for src_id, targets in self._adjacency.items():
            for target in targets:
                rev[target.id].append(src_id)
        return rev

    def topological_sort(self) -> List[Node]:
        adj = self._adjacency_ids()
        indeg = {nid: 0 for nid in self._nodes}
        for src, targets in adj.items():
            for tgt in targets:
                indeg[tgt] = indeg.get(tgt, 0) + 1
        queue = [nid for nid in self._nodes if indeg.get(nid, 0) == 0]
        order: List[Node] = []
        while queue:
            nid = queue.pop(0)
            order.append(self._nodes[nid])
            for tgt in adj.get(nid, []):
                indeg[tgt] -= 1
                if indeg[tgt] == 0:
                    queue.append(tgt)
        if len(order) != len(self._nodes):
            raise ValueError("Graph is not a DAG (cycle detected).")
        return order

    def _reachable(self, start_id, adj) -> Set[Node]:
        seen: Set[Union[int, UUID]] = set()
        stack = list(adj.get(start_id, []))
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(adj.get(nid, []))
        return {self._nodes[nid] for nid in seen if nid in self._nodes}

    def descendants(self, node: Node) -> Set[Node]:
        return self._reachable(node.id, self._adjacency_ids())

    def ancestors(self, node: Node) -> Set[Node]:
        return self._reachable(node.id, self._reverse_adjacency_ids())

    def roots(self) -> Set[Node]:
        has_incoming = {t.id for targets in self._adjacency.values() for t in targets}
        return {n for nid, n in self._nodes.items() if nid not in has_incoming}

    def leaves(self) -> Set[Node]:
        return {n for nid, n in self._nodes.items() if not self._adjacency.get(nid)}