import pytest
from unittest.mock import AsyncMock

from spar.router.hybrid_router import HybridRouter
from spar.router.schemas import Route, RouteResult


@pytest.fixture
def router():
    return HybridRouter(embedding_threshold=0.65, use_llm=False)


async def test_regex_hit_short_circuits(router):
    result = await router.route("What does ALM-4012 mean?")
    assert result.route == Route.STRUCTURED_LOOKUP
    assert result.layer == "regex"


async def test_embedding_hit(router):
    result = await router.route("How do I configure RACH parameters step by step?")
    assert result.route == Route.PROCEDURAL
    assert result.layer == "embedding"


async def test_llm_called_when_embedding_misses():
    router = HybridRouter(embedding_threshold=0.99, use_llm=True)
    router._llm_router.route = AsyncMock(
        return_value=RouteResult(route=Route.DIAGNOSTIC, confidence=0.88, layer="llm")
    )
    result = await router.route("Something ambiguous xyz")
    router._llm_router.route.assert_called_once()
    assert result.layer == "llm"
