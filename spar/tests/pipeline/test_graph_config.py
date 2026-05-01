# tests/pipeline/test_graph_config.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _deps():
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock()))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = MagicMock(return_value=[])
    return router, reranker, encoder, milvus


def test_build_graph_baseline_has_no_rerank_node():
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(name="baseline")
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "rerank" not in node_names
    assert "route" in node_names


def test_build_graph_full_retrieval_has_rerank():
    router, reranker, encoder, milvus = _deps()
    cfg = next(c for c in PRESET_CONFIGS if c.name == "full_retrieval")
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "rerank" in node_names
    assert "preprocess" in node_names
    assert "prepare_context" in node_names


def test_build_graph_default_preserves_existing_behavior():
    """No config arg = full_retrieval behavior (all nodes, stub generate)."""
    router, reranker, encoder, milvus = _deps()
    graph = build_graph(router, reranker, encoder, milvus)
    node_names = set(graph.get_graph().nodes.keys())
    assert "rerank" in node_names
    assert "preprocess" in node_names


def test_build_graph_baseline_entry_is_route():
    """With no expansion/context, graph starts at route."""
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(name="baseline")
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "preprocess" not in node_names
    assert "prepare_context" not in node_names
    assert "route" in node_names
    # verify route is the entry point (has edge from __start__)
    graph_data = graph.get_graph()
    start_edges = [e for e in graph_data.edges if e.source == "__start__"]
    assert start_edges[0].target == "route"


def test_build_graph_qexpand_only_no_prepare_context():
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(name="+qexpand", use_query_expansion=True)
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "preprocess" in node_names
    assert "prepare_context" not in node_names
    # verify preprocess is the entry point (has edge from __start__)
    graph_data = graph.get_graph()
    start_edges = [e for e in graph_data.edges if e.source == "__start__"]
    assert len(start_edges) == 1
    assert start_edges[0].target == "preprocess"
