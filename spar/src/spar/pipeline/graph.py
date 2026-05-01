from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.reranker.client import CrossEncoderClient
from spar.router.hybrid_router import HybridRouter
from spar.router.schemas import Route


def _route_selector(state: SparState) -> str:
    route = state["route_result"].route
    if route == Route.STRUCTURED_LOOKUP:
        return "structured_retrieve"
    if route == Route.DIAGNOSTIC:
        return "multi_hop_retrieve"
    return "rag_retrieve"


def build_graph(
    router: HybridRouter,
    reranker: CrossEncoderClient,
    acronyms_path: Path | None = None,
):
    nodes = Nodes.create(router=router, reranker=reranker, acronyms_path=acronyms_path)

    g: StateGraph = StateGraph(SparState)
    g.add_node("preprocess", nodes.preprocess)
    g.add_node("route", nodes.route)
    g.add_node("rag_retrieve", nodes.rag_retrieve)
    g.add_node("structured_retrieve", nodes.structured_retrieve)
    g.add_node("multi_hop_retrieve", nodes.multi_hop_retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_node("generate", nodes.generate)

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "route")
    g.add_conditional_edges(
        "route",
        _route_selector,
        {
            "rag_retrieve": "rag_retrieve",
            "structured_retrieve": "structured_retrieve",
            "multi_hop_retrieve": "multi_hop_retrieve",
        },
    )
    g.add_edge("rag_retrieve", "rerank")
    g.add_edge("structured_retrieve", "rerank")
    g.add_edge("multi_hop_retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)

    return g.compile()
