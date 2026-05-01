from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from spar.encoder.client import SentenceTransformerEncoder


@pytest.fixture
def encoder():
    with patch("spar.encoder.client.SentenceTransformer") as mock_cls:
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


def test_encode_calls_underlying_model(encoder):
    encoder.encode(["test"], normalize=True)
    encoder._model.encode.assert_called_once_with(
        ["test"], normalize_embeddings=True
    )
