from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from spar.retrieval.query_decomposer import QueryDecomposer, _MAX_SUB_QUESTIONS


def _make_mock_client(response: str) -> AsyncMock:
    client = AsyncMock()
    client.chat = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_decompose_returns_sub_questions():
    sub_qs = ["What is PDCP?", "How does RLC affect throughput?"]
    mock_client = _make_mock_client(json.dumps(sub_qs))

    with patch("spar.retrieval.query_decomposer.get_client", return_value=mock_client):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("What is PDCP and how does RLC affect throughput?")

    assert result == sub_qs


@pytest.mark.asyncio
async def test_decompose_caps_at_max():
    sub_qs = [f"Question {i}" for i in range(10)]
    mock_client = _make_mock_client(json.dumps(sub_qs))

    with patch("spar.retrieval.query_decomposer.get_client", return_value=mock_client):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("multi-part query")

    assert len(result) <= _MAX_SUB_QUESTIONS


@pytest.mark.asyncio
async def test_decompose_fallback_on_invalid_json():
    mock_client = _make_mock_client("not valid json {{{")

    with patch("spar.retrieval.query_decomposer.get_client", return_value=mock_client):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("original query")

    assert result == ["original query"]


@pytest.mark.asyncio
async def test_decompose_fallback_on_empty_list():
    mock_client = _make_mock_client("[]")

    with patch("spar.retrieval.query_decomposer.get_client", return_value=mock_client):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("original query")

    assert result == ["original query"]


@pytest.mark.asyncio
async def test_decompose_fallback_on_non_list():
    mock_client = _make_mock_client(json.dumps({"question": "something"}))

    with patch("spar.retrieval.query_decomposer.get_client", return_value=mock_client):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("original query")

    assert result == ["original query"]


@pytest.mark.asyncio
async def test_decompose_strips_empty_strings():
    sub_qs = ["Valid question", "", "  ", "Another valid question"]
    mock_client = _make_mock_client(json.dumps(sub_qs))

    with patch("spar.retrieval.query_decomposer.get_client", return_value=mock_client):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("query")

    assert "" not in result
    assert "  " not in result
    assert len(result) == 2


@pytest.mark.asyncio
async def test_decompose_fallback_on_llm_error():
    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    with patch("spar.retrieval.query_decomposer.get_client", return_value=mock_client):
        decomposer = QueryDecomposer()
        result = await decomposer.decompose("original query")

    assert result == ["original query"]
