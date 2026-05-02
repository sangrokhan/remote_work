from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spar.reranker.client import CrossEncoderClient, LocalCrossEncoderClient


class TestCrossEncoderClient:
    @pytest.fixture
    def mock_http(self):
        with patch("spar.reranker.client.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            yield mock_instance

    async def test_rerank_returns_scores_in_order(self, mock_http):
        mock_http.post.return_value = AsyncMock(
            raise_for_status=MagicMock(),
            json=MagicMock(return_value={
                "results": [
                    {"index": 1, "relevance_score": 0.7},
                    {"index": 0, "relevance_score": 0.9},
                ]
            }),
        )
        client = CrossEncoderClient(base_url="http://localhost:8002/rerank", model="bge")
        scores = await client.rerank("query", ["doc0", "doc1"])
        assert scores == [0.9, 0.7]

    async def test_rerank_raises_on_http_error(self, mock_http):
        import httpx
        mock_http.post.return_value = AsyncMock(
            raise_for_status=MagicMock(side_effect=httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock())),
        )
        client = CrossEncoderClient(base_url="http://localhost:8002/rerank", model="bge")
        with pytest.raises(Exception):
            await client.rerank("query", ["doc"])

    def test_model_property(self):
        with patch("spar.reranker.client.httpx.AsyncClient"):
            client = CrossEncoderClient(base_url="http://x", model="my-model")
        assert client.model == "my-model"


class TestLocalCrossEncoderClient:
    @pytest.fixture
    def mock_cross_encoder(self):
        with patch("spar.reranker.client._CrossEncoder") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = np.array([0.9, 0.3, 0.7])
            mock_cls.return_value = mock_instance
            yield mock_cls, mock_instance

    async def test_rerank_returns_float_list(self, mock_cross_encoder):
        _, mock_enc = mock_cross_encoder
        client = LocalCrossEncoderClient(model="BAAI/bge-reranker-v2-m3")
        scores = await client.rerank("query", ["doc0", "doc1", "doc2"])
        assert scores == pytest.approx([0.9, 0.3, 0.7])
        assert all(isinstance(s, float) for s in scores)

    async def test_rerank_empty_documents(self, mock_cross_encoder):
        client = LocalCrossEncoderClient(model="BAAI/bge-reranker-v2-m3")
        scores = await client.rerank("query", [])
        assert scores == []

    async def test_rerank_passes_pairs(self, mock_cross_encoder):
        _, mock_enc = mock_cross_encoder
        client = LocalCrossEncoderClient(model="BAAI/bge-reranker-v2-m3")
        mock_enc.predict.return_value = np.array([0.5])
        await client.rerank("my query", ["only doc"])
        mock_enc.predict.assert_called_once_with([("my query", "only doc")])

    def test_model_property(self, mock_cross_encoder):
        client = LocalCrossEncoderClient(model="BAAI/bge-reranker-v2-m3")
        assert client.model == "BAAI/bge-reranker-v2-m3"

    def test_raises_without_sentence_transformers(self):
        with patch("spar.reranker.client._CrossEncoder", None):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                LocalCrossEncoderClient(model="BAAI/bge-reranker-v2-m3")

    async def test_aclose_is_noop(self, mock_cross_encoder):
        client = LocalCrossEncoderClient(model="BAAI/bge-reranker-v2-m3")
        await client.aclose()
