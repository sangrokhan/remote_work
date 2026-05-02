from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from spar.pipeline.config import GraphConfig, PRESET_CONFIGS
from spar.pipeline.graph import build_graph


def _deps():
    router = MagicMock()
    router.route = AsyncMock(return_value=MagicMock(route=MagicMock(), needs_decomposition=False))
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=[])
    encoder = MagicMock()
    encoder.encode = MagicMock(return_value=np.zeros((1, 3)))
    milvus = MagicMock()
    milvus.hybrid_search = MagicMock(return_value=[])
    return router, reranker, encoder, milvus


def test_verify_loop_config_has_both_nodes():
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(
        name="test_verify",
        use_reranker=True,
        use_real_generate=True,
        use_verify_loop=True,
    )
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" in node_names
    assert "verify" in node_names


def test_verify_loop_false_has_no_verify_nodes():
    router, reranker, encoder, milvus = _deps()
    cfg = GraphConfig(name="no_verify", use_reranker=True)
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" not in node_names
    assert "verify" not in node_names


def test_preset_verify_loop_has_correct_nodes():
    router, reranker, encoder, milvus = _deps()
    cfg = next(c for c in PRESET_CONFIGS if c.name == "verify_loop")
    graph = build_graph(router, reranker, encoder, milvus, config=cfg)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" in node_names
    assert "verify" in node_names
    assert "rerank" in node_names
    assert "preprocess" in node_names


def test_default_graph_unchanged_no_verify_nodes():
    """기존 동작 보존 — default config(full_retrieval)에는 verify 노드 없음."""
    router, reranker, encoder, milvus = _deps()
    graph = build_graph(router, reranker, encoder, milvus)
    node_names = set(graph.get_graph().nodes.keys())
    assert "tool_call" not in node_names
    assert "verify" not in node_names


# ============================================================================
# Tests for _verify_selector routing logic and boundary conditions
# ============================================================================


def test_verify_selector_routes_to_tool_call_when_retry_available():
    """When score < 3, retry < 3, and strategies remain — route to tool_call."""
    from langgraph.graph import END
    from spar.pipeline.graph import _verify_selector

    state = {
        "verify_score": 1.0,
        "retry_count": 0,
        "tried_strategies": ["rag"],
    }
    result = _verify_selector(state)
    assert result == "tool_call"


def test_verify_selector_routes_to_end_when_retry_limit_reached():
    """When retry_count >= 3 — route to END regardless of score."""
    from langgraph.graph import END
    from spar.pipeline.graph import _verify_selector

    state = {
        "verify_score": 1.0,
        "retry_count": 3,
        "tried_strategies": ["rag"],
    }
    result = _verify_selector(state)
    assert result == END


def test_verify_selector_routes_to_end_when_strategies_exhausted():
    """When all strategies tried — route to END regardless of score."""
    from langgraph.graph import END
    from spar.pipeline.graph import _verify_selector

    state = {
        "verify_score": 1.0,
        "retry_count": 0,
        "tried_strategies": ["rag", "decomposed", "multi_hop", "structured"],
    }
    result = _verify_selector(state)
    assert result == END


def test_verify_selector_routes_to_end_when_score_sufficient():
    """When score >= 3 — route to END regardless of retry count or strategies."""
    from langgraph.graph import END
    from spar.pipeline.graph import _verify_selector

    state = {
        "verify_score": 4.0,
        "retry_count": 0,
        "tried_strategies": [],
    }
    result = _verify_selector(state)
    assert result == END
