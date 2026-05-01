from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from spar.retrieval.query_rewriter import (
    MAX_HISTORY_TURNS,
    QueryRewriteResult,
    build_context,
    extract_relevant_acronyms,
    format_history,
    rewrite_query,
)

SAMPLE_ACRONYMS = {
    "global": {
        "PDCP": {"expansion": "Packet Data Convergence Protocol", "variants": []},
        "RLC": {"expansion": "Radio Link Control", "variants": ["Rlc"]},
    }
}


def test_format_history_empty():
    assert format_history([]) == ""


def test_format_history_labels():
    history = [
        {"role": "user", "content": "What is PDCP?"},
        {"role": "assistant", "content": "PDCP handles header compression."},
    ]
    result = format_history(history)
    assert "User: What is PDCP?" in result
    assert "Assistant: PDCP handles header compression." in result


def test_format_history_max_turns():
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    result = format_history(history, max_turns=MAX_HISTORY_TURNS)
    lines = result.strip().splitlines()
    assert len(lines) == MAX_HISTORY_TURNS * 2


def test_extract_relevant_acronyms_match():
    result = extract_relevant_acronyms("explain PDCP timer", [], SAMPLE_ACRONYMS)
    assert "PDCP" in result
    assert result["PDCP"] == "Packet Data Convergence Protocol"


def test_extract_relevant_acronyms_variant_match():
    result = extract_relevant_acronyms("Rlc layer behavior", [], SAMPLE_ACRONYMS)
    assert "Rlc" in result


def test_extract_relevant_acronyms_no_match():
    result = extract_relevant_acronyms("explain timer", [], SAMPLE_ACRONYMS)
    assert result == {}


def test_extract_relevant_acronyms_from_history():
    history = [{"role": "user", "content": "What about RLC?"}]
    result = extract_relevant_acronyms("and that?", history, SAMPLE_ACRONYMS)
    assert "RLC" in result


def test_build_context_with_history_and_acronyms():
    history = [{"role": "user", "content": "What is PDCP?"}]
    result = build_context("how does it work in NR?", history, SAMPLE_ACRONYMS)
    assert "Conversation history:" in result
    assert "PDCP" in result
    assert "Relevant acronyms:" in result


def test_build_context_no_history():
    result = build_context("explain RLC", [], SAMPLE_ACRONYMS)
    assert "Conversation history:" not in result
    assert "RLC" in result


def test_build_context_no_match():
    result = build_context("explain timer", [], SAMPLE_ACRONYMS)
    assert result == ""


# --- rewrite_query async tests ---

def _make_client(raw_response: str) -> Any:
    mock = AsyncMock()
    mock.chat = AsyncMock(return_value=raw_response)
    return mock


@pytest.mark.asyncio
async def test_rewrite_query_basic():
    payload = {"rewritten": "What is the handover procedure?", "complexity": "simple", "rationale": "single concept"}
    with patch("spar.retrieval.query_rewriter.get_client", new_callable=AsyncMock, return_value=_make_client(json.dumps(payload))):
        result = await rewrite_query("What is it?", history=[], acronyms={})
    assert isinstance(result, QueryRewriteResult)
    assert result.rewritten == "What is the handover procedure?"
    assert result.complexity == "simple"
    assert result.original == "What is it?"


@pytest.mark.asyncio
async def test_rewrite_query_complex_classification():
    payload = {"rewritten": "Compare X2 and Xn handover latency", "complexity": "complex", "rationale": "comparison"}
    with patch("spar.retrieval.query_rewriter.get_client", new_callable=AsyncMock, return_value=_make_client(json.dumps(payload))):
        result = await rewrite_query("Compare them", history=[], acronyms={})
    assert result.complexity == "complex"


@pytest.mark.asyncio
async def test_rewrite_query_json_parse_failure_fallback():
    with patch("spar.retrieval.query_rewriter.get_client", new_callable=AsyncMock, return_value=_make_client("not json at all")):
        result = await rewrite_query("original query", history=[], acronyms={})
    assert result.rewritten == "original query"
    assert result.complexity == "simple"
    assert result.original == "original query"


@pytest.mark.asyncio
async def test_rewrite_query_llm_exception_fallback():
    bad_client = AsyncMock()
    bad_client.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
    with patch("spar.retrieval.query_rewriter.get_client", new_callable=AsyncMock, return_value=bad_client):
        result = await rewrite_query("original query", history=[], acronyms={})
    assert result.rewritten == "original query"
    assert result.complexity == "simple"
