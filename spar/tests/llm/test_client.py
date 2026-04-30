import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from spar.llm.client import LLMClient


@pytest.fixture
def client():
    return LLMClient(base_url="http://localhost:8000/v1", model="test-model")


def test_model_property(client):
    assert client.model == "test-model"


async def test_chat_returns_content(client):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "answer text"
    with patch.object(client._openai.chat.completions, "create", new=AsyncMock(return_value=mock_resp)):
        result = await client.chat([{"role": "user", "content": "hello"}])
    assert result == "answer text"


async def test_chat_none_content_returns_empty(client):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = None
    with patch.object(client._openai.chat.completions, "create", new=AsyncMock(return_value=mock_resp)):
        result = await client.chat([{"role": "user", "content": "hello"}])
    assert result == ""


async def test_chat_passes_kwargs(client):
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "ok"
    create_mock = AsyncMock(return_value=mock_resp)
    with patch.object(client._openai.chat.completions, "create", new=create_mock):
        await client.chat([{"role": "user", "content": "q"}], temperature=0.5, max_tokens=256)
    _, kwargs = create_mock.call_args
    assert kwargs["temperature"] == 0.5
    assert kwargs["max_tokens"] == 256
