from typing import Optional

from causality_graph.store.graph import CausalityGraph
from causality_graph.store.embeddings import EmbeddingStore


class Retriever:
    def __init__(self, graph: CausalityGraph, embeddings: EmbeddingStore):
        self._graph = graph
        self._embeddings = embeddings

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 2,
        gen_filter: Optional[str] = None,
    ) -> dict:
        # 1. Semantic search for seed nodes
        hits = self._embeddings.search(query, top_k=top_k)
        seed_ids = [h["id"] for h in hits]

        # 2. k-hop expansion in NetworkX graph
        g = self._graph._g
        subgraph_nodes = set(n for n in seed_ids if n in g)
        for _ in range(hops):
            neighbors = set()
            for node in list(subgraph_nodes):
                neighbors.update(g.successors(node))
                neighbors.update(g.predecessors(node))
            subgraph_nodes.update(neighbors)

        # 3. Apply gen filter — include nodes with matching gen OR gen="both"
        if gen_filter:
            filtered = set()
            for node_id in subgraph_nodes:
                node_data = self._graph.get_node(node_id)
                if node_data is None:
                    continue
                gen = node_data.get("gen")
                if gen is None or gen in (gen_filter, "both"):
                    filtered.add(node_id)
            subgraph_nodes = filtered

        # 4. Collect nodes and edges
        nodes = []
        for node_id in subgraph_nodes:
            data = self._graph.get_node(node_id)
            if data:
                nodes.append({"id": node_id, **data})

        edges = []
        for from_id in subgraph_nodes:
            if from_id not in g:
                continue
            for _, to_id, edge_data in g.out_edges(from_id, data=True):
                if to_id in subgraph_nodes:
                    edges.append({"from": from_id, "to": to_id, **edge_data})

        return {"nodes": nodes, "edges": edges}
