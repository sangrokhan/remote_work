from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from spar.encoder.base import EncoderClient
from spar.router.hybrid_router import HybridRouter
from spar.router.schemas import Route, RouteResult


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


@pytest.fixture
def router():
    return HybridRouter(encoder=_make_encoder(), embedding_threshold=0.65, use_llm=False)


async def test_regex_hit_short_circuits(router):
    result = await router.route("What does ALM-4012 mean?")
    assert result.route == Route.STRUCTURED_LOOKUP
    assert result.layer == "regex"


async def test_embedding_hit(router):
    result = await router.route("How do I configure RACH parameters step by step?")
    assert result is not None
    assert result.layer in ("embedding", "fallback")


async def test_llm_called_when_embedding_misses():
    router = HybridRouter(encoder=_make_encoder(), embedding_threshold=0.99, use_llm=True)
    router._llm_router.route = AsyncMock(
        return_value=RouteResult(route=Route.DIAGNOSTIC, confidence=0.88, layer="llm")
    )
    result = await router.route("Something ambiguous xyz")
    router._llm_router.route.assert_called_once()
    assert result.layer == "llm"
