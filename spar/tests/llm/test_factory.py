import pytest

from spar.llm.client import LLMClient
from spar.llm.config import LLMSettings
from spar.llm.factory import LLMFactory, LLMRole


@pytest.fixture
def settings():
    return LLMSettings(
        llm_main_url="http://main:8000/v1",
        llm_main_model="main-model",
        llm_main_api_key="key-main",
        llm_router_url="http://router:8001/v1",
        llm_router_model="router-model",
        llm_router_api_key="key-router",
    )


def test_create_main(settings):
    client = LLMFactory.create(LLMRole.MAIN, settings)
    assert isinstance(client, LLMClient)
    assert client.model == "main-model"


def test_create_router(settings):
    client = LLMFactory.create(LLMRole.ROUTER, settings)
    assert isinstance(client, LLMClient)
    assert client.model == "router-model"


def test_unknown_role_raises(settings):
    with pytest.raises(ValueError, match="Unknown LLM role"):
        LLMFactory.create("unknown", settings)  # type: ignore[arg-type]
