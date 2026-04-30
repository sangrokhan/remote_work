import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spar.llm.client import LLMClient
from spar.router.llm_router import LLMRouter
from spar.router.schemas import Route


def _make_mock_client(content: str) -> LLMClient:
    client = MagicMock(spec=LLMClient)
    client.chat = AsyncMock(return_value=content)
    return client


@pytest.fixture
def router():
    return LLMRouter()


async def test_llm_routes_procedural(router):
    mock_client = _make_mock_client(
        '{"route":"procedural","confidence":0.92,'
        '"entities":{},"product":"NR","release":"v7.1"}'
    )
    with patch("spar.router.llm_router.get_client", new=AsyncMock(return_value=mock_client)):
        result = await router.route("How to configure RACH in NR v7.1?")

    assert result.route == Route.PROCEDURAL
    assert result.confidence == 0.92
    assert result.product == "NR"
    assert result.release == "v7.1"


async def test_llm_falls_back_on_parse_error(router):
    mock_client = _make_mock_client("NOT_JSON")
    with patch("spar.router.llm_router.get_client", new=AsyncMock(return_value=mock_client)):
        result = await router.route("Some query")

    assert result.route == Route.DEFAULT_RAG
    assert result.layer == "llm_fallback"


async def test_llm_falls_back_on_network_error(router):
    with patch(
        "spar.router.llm_router.get_client",
        new=AsyncMock(side_effect=ConnectionError("vLLM unreachable")),
    ):
        result = await router.route("Some query")

    assert result.route == Route.DEFAULT_RAG
    assert result.layer == "llm_fallback"
