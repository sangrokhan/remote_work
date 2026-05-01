from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from spar.encoder.base import EncoderClient
from spar.encoder.registry import (
    RemoteSentenceTransformerEncoder,
    SentenceTransformerEncoder,
    get_encoder,
    reset_registry,
)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


# --- SentenceTransformerEncoder ---

@pytest.fixture
def encoder():
    with patch("spar.encoder.registry.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_model
        yield SentenceTransformerEncoder(model_name="BAAI/bge-small-en-v1.5", device="cpu")


def test_model_name(encoder):
    assert encoder.model_name == "BAAI/bge-small-en-v1.5"


def test_encode_returns_ndarray(encoder):
    result = encoder.encode(["hello world"])
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3)


def test_encode_normalize_flag(encoder):
    encoder.encode(["test"], normalize=True)
    encoder._model.encode.assert_called_once_with(["test"], normalize_embeddings=True)


def test_encode_normalize_false(encoder):
    encoder.encode(["test"], normalize=False)
    encoder._model.encode.assert_called_once_with(["test"], normalize_embeddings=False)


def test_implements_encoder_client(encoder):
    assert isinstance(encoder, EncoderClient)


# --- get_encoder singleton ---

async def test_get_encoder_returns_encoder_client():
    with patch("spar.encoder.registry.SentenceTransformer"):
        enc = await get_encoder()
    assert isinstance(enc, EncoderClient)


async def test_get_encoder_singleton():
    with patch("spar.encoder.registry.SentenceTransformer"):
        a = await get_encoder()
        b = await get_encoder()
    assert a is b


async def test_reset_registry_breaks_singleton():
    with patch("spar.encoder.registry.SentenceTransformer"):
        a = await get_encoder()
    reset_registry()
    with patch("spar.encoder.registry.SentenceTransformer"):
        b = await get_encoder()
    assert a is not b


async def test_get_encoder_uses_env_model(monkeypatch):
    monkeypatch.setenv("ENCODER_MODEL", "intfloat/e5-large-v2")
    with patch("spar.encoder.registry.SentenceTransformer") as mock_cls:
        mock_cls.return_value = MagicMock()
        enc = await get_encoder()
    assert enc.model_name == "intfloat/e5-large-v2"


async def test_get_encoder_uses_env_device(monkeypatch):
    monkeypatch.setenv("ENCODER_DEVICE", "cuda")
    with patch("spar.encoder.registry.SentenceTransformer") as mock_cls:
        mock_cls.return_value = MagicMock()
        await get_encoder()
    mock_cls.assert_called_once_with("BAAI/bge-small-en-v1.5", device="cuda")


async def test_get_encoder_uses_remote_url(monkeypatch):
    monkeypatch.setenv("ENCODER_URL", "http://embedder-host:8000/v1")
    with patch("spar.encoder.registry.httpx.Client") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        enc = await get_encoder()
    assert isinstance(enc, RemoteSentenceTransformerEncoder)
    assert mock_instance is not None
