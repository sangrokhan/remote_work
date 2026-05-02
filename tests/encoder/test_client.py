import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spar.encoder.client import CrossEncoderClient


@pytest.fixture
def client():
    return CrossEncoderClient(
        base_url="http://localhost:8002/rerank",
        model="test-reranker",
    )


def test_model_property(client):
    assert client.model == "test-reranker"


async def test_rerank_returns_scores(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.3},
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._http, "post", new=AsyncMock(return_value=mock_resp)):
        scores = await client.rerank("query", ["doc0", "doc1"])
    assert scores == [0.9, 0.3]


async def test_rerank_preserves_order(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "results": [
            {"index": 1, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.8},
        ]
    }
    mock_resp.raise_for_status = MagicMock()
    with patch.object(client._http, "post", new=AsyncMock(return_value=mock_resp)):
        scores = await client.rerank("query", ["doc0", "doc1"])
    assert scores[0] == 0.8
    assert scores[1] == 0.5
