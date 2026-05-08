import json
from pathlib import Path
from dataclasses import asdict
from typing import Optional
import networkx as nx

from causality_graph.schema import KPINode, FeatureNode, ParameterNode, Edge, NodeType


class CausalityGraph:
    def __init__(self):
        self._g = nx.DiGraph()

    def add_kpi(self, node: KPINode) -> None:
        self._g.add_node(node.id, **asdict(node), node_type=NodeType.KPI.value)

    def add_feature(self, node: FeatureNode) -> None:
        self._g.add_node(node.id, **asdict(node), node_type=NodeType.FEATURE.value)

    def add_parameter(self, node: ParameterNode) -> None:
        self._g.add_node(node.id, **asdict(node), node_type=NodeType.PARAMETER.value)

    def add_edge(self, edge: Edge) -> None:
        self._g.add_edge(edge.from_id, edge.to_id, **asdict(edge))

    def get_node(self, node_id: str) -> Optional[dict]:
        if node_id not in self._g:
            return None
        return dict(self._g.nodes[node_id])

    def get_neighbors(self, node_id: str) -> list[dict]:
        result = []
        for neighbor_id in self._g.successors(node_id):
            data = dict(self._g.nodes[neighbor_id])
            data["id"] = neighbor_id
            result.append(data)
        return result

    def get_edges_from(self, node_id: str) -> list[dict]:
        result = []
        for _, to_id, data in self._g.out_edges(node_id, data=True):
            result.append(dict(data))
        return result

    def node_count(self) -> int:
        return self._g.number_of_nodes()

    def edge_count(self) -> int:
        return self._g.number_of_edges()

    def save(self, path: Path) -> None:
        data = nx.node_link_data(self._g)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CausalityGraph":
        data = json.loads(Path(path).read_text())
        g = cls()
        g._g = nx.node_link_graph(data)
        return g
