import pytest

from spar.encoder.client import CrossEncoderClient
from spar.encoder.config import EncoderSettings
from spar.encoder.factory import EncoderFactory, EncoderRole

_SETTINGS = EncoderSettings(
    encoder_reranker_url="http://test:8002/rerank",
    encoder_reranker_model="test-reranker",
)


def test_create_reranker():
    client = EncoderFactory.create(EncoderRole.RERANKER, _SETTINGS)
    assert isinstance(client, CrossEncoderClient)
    assert client.model == "test-reranker"


def test_create_unknown_role_raises():
    with pytest.raises(ValueError, match="Unknown encoder role"):
        EncoderFactory.create("unknown", _SETTINGS)  # type: ignore[arg-type]
