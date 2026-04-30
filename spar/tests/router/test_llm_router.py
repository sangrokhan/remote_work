import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spar.router.llm_router import LLMRouter
from spar.router.schemas import Route


@pytest.fixture
def router():
    return LLMRouter(base_url="http://localhost:8001/v1", model="stub")


async def test_llm_routes_procedural(router):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"route":"procedural","confidence":0.92,'
        '"entities":{},"product":"NR","release":"v7.1"}'
    )
    with patch.object(
        router._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
    ):
        result = await router.route("How to configure RACH in NR v7.1?")

    assert result.route == Route.PROCEDURAL
    assert result.confidence == 0.92
    assert result.product == "NR"
    assert result.release == "v7.1"


async def test_llm_falls_back_on_parse_error(router):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "NOT_JSON"
    with patch.object(
        router._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
    ):
        result = await router.route("Some query")

    assert result.route == Route.DEFAULT_RAG
    assert result.layer == "llm_fallback"
