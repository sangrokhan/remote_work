import pytest

from spar.llm.client import LLMClient
from spar.llm.config import LLMSettings
from spar.llm.factory import LLMFactory, LLMRole
from spar.llm.fallback import FallbackLLMClient


@pytest.fixture
def settings():
    return LLMSettings(
        llm_main_url="http://main:8000/v1",
        llm_main_model="main-model",
        llm_main_api_key="key-main",
        llm_router_url="http://router:8001/v1",
        llm_router_model="router-model",
        llm_router_api_key="key-router",
        gemini_cli_fallback_enabled=False,
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


def test_create_with_gemini_fallback_wraps_primary(settings):
    settings_with_fallback = settings.model_copy(
        update={"gemini_cli_fallback_enabled": True}
    )
    client = LLMFactory.create(LLMRole.MAIN, settings_with_fallback)
    assert isinstance(client, FallbackLLMClient)
    assert client.model == "main-model"
