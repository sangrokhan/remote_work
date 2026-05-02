"""LangGraph StateGraph builder; wires Nodes into the RAG pipeline based on GraphConfig feature flags."""
from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, StateGraph

from spar.encoder.base import EncoderClient
from spar.llm.client import LLMClient
from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.nodes import Nodes
from spar.pipeline.state import SparState
from spar.reranker.client import CrossEncoderClient
from spar.retrieval.milvus_client import SparMilvusClient
from spar.router.hybrid_router import HybridRouter
from spar.router.schemas import Route

_RETRIEVE_NODES = ("rag_retrieve", "structured_retrieve", "multi_hop_retrieve")
_DEFAULT_CONFIG = next(
    (c for c in PRESET_CONFIGS if c.name == "full_retrieval"),
    None,
)
assert _DEFAULT_CONFIG is not None, (
    "PRESET_CONFIGS must contain a 'full_retrieval' entry (required as build_graph default)"
)


def _verify_selector(state: SparState) -> str:
    score = state.get("verify_score", 5.0)
    retry_count = state.get("retry_count", 0)
    tried = set(state.get("tried_strategies") or [])
    all_strategies = {"rag", "decomposed", "multi_hop", "structured"}
    remaining = all_strategies - tried
    if score < 3 and retry_count < 3 and remaining:
        return "tool_call"
    return END


def _route_selector(state: SparState) -> str:
    route_result = state["route_result"]
    if route_result.needs_decomposition:
        return "decompose"
    route = route_result.route
    if route == Route.STRUCTURED_LOOKUP:
        return "structured_retrieve"
    if route == Route.DIAGNOSTIC:
        return "multi_hop_retrieve"
    return "rag_retrieve"


def build_graph(
    router: HybridRouter,
    reranker: CrossEncoderClient,
    encoder: EncoderClient,
    milvus: SparMilvusClient,
    config: GraphConfig | None = None,
    acronyms_path: Path | None = None,
    llm: LLMClient | None = None,
):
    cfg = config if config is not None else _DEFAULT_CONFIG

    nodes = Nodes.create(
        router=router,
        reranker=reranker,
        encoder=encoder,
        milvus=milvus,
        acronyms_path=acronyms_path,
        llm=llm,
    )

    g: StateGraph = StateGraph(SparState)

    # determine entry point
    if cfg.use_query_expansion:
        entry = "preprocess"
    elif cfg.use_prepare_context:
        entry = "prepare_context"
    else:
        entry = "route"
    g.set_entry_point(entry)

    # optional pre-route nodes
    if cfg.use_query_expansion:
        g.add_node("preprocess", nodes.preprocess)
        g.add_node("rewrite_query", nodes.rewrite_query)
        g.add_edge("preprocess", "rewrite_query")
        next_node = "prepare_context" if cfg.use_prepare_context else "route"
        g.add_edge("rewrite_query", next_node)

    if cfg.use_prepare_context:
        g.add_node("prepare_context", nodes.prepare_context)
        g.add_edge("prepare_context", "route")

    # core nodes always present
    g.add_node("route", nodes.route)
    for name in _RETRIEVE_NODES:
        g.add_node(name, getattr(nodes, name))

    # decompose path always present (router can select it at runtime)
    g.add_node("decompose", nodes.decompose)
    g.add_node("decomposed_retrieve", nodes.decomposed_retrieve)
    g.add_edge("decompose", "decomposed_retrieve")

    _all_retrieve = (*_RETRIEVE_NODES, "decomposed_retrieve")

    g.add_conditional_edges(
        "route",
        _route_selector,
        {**{n: n for n in _RETRIEVE_NODES}, "decompose": "decompose"},
    )

    g.add_node("generate", nodes.generate)

    if cfg.use_verify_loop:
        g.add_node("tool_call", nodes.tool_call)
        g.add_node("verify", nodes.verify)

        for name in _all_retrieve:
            g.add_edge(name, "tool_call")

        if cfg.use_reranker:
            g.add_node("rerank", nodes.rerank)
            g.add_edge("tool_call", "rerank")
            g.add_edge("rerank", "generate")
        else:
            g.add_edge("tool_call", "generate")

        g.add_edge("generate", "verify")
        g.add_conditional_edges(
            "verify",
            _verify_selector,
            {"tool_call": "tool_call", END: END},
        )
    else:
        if cfg.use_reranker:
            g.add_node("rerank", nodes.rerank)
            for name in _all_retrieve:
                g.add_edge(name, "rerank")
            g.add_edge("rerank", "generate")
        else:
            for name in _all_retrieve:
                g.add_edge(name, "generate")

        g.add_edge("generate", END)

    return g.compile()
