from __future__ import annotations

import pytest
from unittest.mock import patch

from spar.reranker.client import CrossEncoderClient, LocalCrossEncoderClient
from spar.reranker.config import EncoderSettings
from spar.reranker.factory import EncoderFactory, EncoderRole


@pytest.fixture
def local_settings():
    return EncoderSettings(
        encoder_reranker_model="BAAI/bge-reranker-v2-m3",
        encoder_reranker_backend="local",
        encoder_reranker_device="cpu",
    )


@pytest.fixture
def remote_settings():
    return EncoderSettings(
        encoder_reranker_url="http://localhost:8002/rerank",
        encoder_reranker_model="BAAI/bge-reranker-v2-m3",
        encoder_reranker_backend="remote",
    )


def test_create_local_returns_local_client(local_settings):
    with patch("spar.reranker.client._CrossEncoder"):
        client = EncoderFactory.create(EncoderRole.RERANKER, local_settings)
    assert isinstance(client, LocalCrossEncoderClient)


def test_create_remote_returns_http_client(remote_settings):
    with patch("spar.reranker.client.httpx.AsyncClient"):
        client = EncoderFactory.create(EncoderRole.RERANKER, remote_settings)
    assert isinstance(client, CrossEncoderClient)


def test_create_local_uses_model_and_device(local_settings):
    with patch("spar.reranker.client._CrossEncoder") as mock_cls:
        client = EncoderFactory.create(EncoderRole.RERANKER, local_settings)
    assert client.model == "BAAI/bge-reranker-v2-m3"
    mock_cls.assert_called_once_with("BAAI/bge-reranker-v2-m3", device="cpu")


def test_create_unknown_role_raises(local_settings):
    with pytest.raises(ValueError, match="Unknown encoder role"):
        EncoderFactory.create("unknown", local_settings)  # type: ignore
