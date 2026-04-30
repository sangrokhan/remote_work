import pytest
from unittest.mock import patch

from spar.llm.client import LLMClient
from spar.llm.config import LLMSettings
from spar.llm.factory import LLMRole
from spar.llm.registry import get_client, reset_registry

_TEST_SETTINGS = LLMSettings(
    llm_main_url="http://main:8000/v1",
    llm_main_model="main-model",
    llm_router_url="http://router:8001/v1",
    llm_router_model="router-model",
)


@pytest.fixture(autouse=True)
def clean_registry():
    reset_registry()
    yield
    reset_registry()


async def test_get_client_returns_llm_client():
    with patch("spar.llm.registry.get_settings", return_value=_TEST_SETTINGS):
        client = await get_client(LLMRole.MAIN)
    assert isinstance(client, LLMClient)
    assert client.model == "main-model"


async def test_get_client_singleton():
    with patch("spar.llm.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_client(LLMRole.MAIN)
        b = await get_client(LLMRole.MAIN)
    assert a is b


async def test_get_client_separate_per_role():
    with patch("spar.llm.registry.get_settings", return_value=_TEST_SETTINGS):
        main = await get_client(LLMRole.MAIN)
        router = await get_client(LLMRole.ROUTER)
    assert main is not router
    assert main.model == "main-model"
    assert router.model == "router-model"


async def test_reset_registry_clears_cache():
    with patch("spar.llm.registry.get_settings", return_value=_TEST_SETTINGS):
        a = await get_client(LLMRole.MAIN)
        reset_registry()
        b = await get_client(LLMRole.MAIN)
    assert a is not b
