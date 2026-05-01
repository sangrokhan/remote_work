from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from spar.encoder.base import EncoderClient
from spar.router.embedding_router import EmbeddingRouter
from spar.router.schemas import Route


def _make_encoder(dim: int = 8) -> EncoderClient:
    """Stub encoder: returns deterministic vectors per unique text."""
    encoder = MagicMock(spec=EncoderClient)

    rng = np.random.default_rng(42)
    cache: dict[str, np.ndarray] = {}

    def _encode(texts: list[str], *, normalize: bool = True) -> np.ndarray:
        vecs = []
        for t in texts:
            if t not in cache:
                v = rng.random(dim).astype(np.float32)
                if normalize:
                    v /= np.linalg.norm(v)
                cache[t] = v
            vecs.append(cache[t])
        return np.array(vecs)

    encoder.encode.side_effect = _encode
    return encoder


@pytest.fixture(scope="module")
def router():
    return EmbeddingRouter(encoder=_make_encoder(), threshold=0.5)


def test_route_returns_result_or_none(router):
    result = router.route("How do I configure RACH parameters in NR?")
    assert result is None or result.layer == "embedding"


def test_low_confidence_returns_none():
    strict = EmbeddingRouter(encoder=_make_encoder(), threshold=0.99)
    result = strict.route("zzz xyz qwerty random nonsense")
    assert result is None


def test_route_result_has_confidence(router):
    result = router.route("What is Carrier Aggregation?")
    if result is not None:
        assert 0.0 <= result.confidence <= 1.0
        assert result.layer == "embedding"


def test_centroids_are_normalized(router):
    for route, centroid in router._centroids.items():
        norm = float(np.linalg.norm(centroid))
        assert abs(norm - 1.0) < 1e-5, f"Centroid for {route} not normalized: norm={norm}"
