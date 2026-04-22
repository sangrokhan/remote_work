import pytest
from langgraph.core.base import BaseLLM


def test_base_llm_is_abstract():
    with pytest.raises(TypeError):
        BaseLLM(api_url="http://example.com", api_key="key")


def test_concrete_subclass_requires_generate():
    class Incomplete(BaseLLM):
        pass

    with pytest.raises(TypeError):
        Incomplete(api_url="", api_key="")


def test_concrete_subclass_works():
    class Concrete(BaseLLM):
        def generate(self, prompt: str, context: str) -> str:
            return f"ok: {context}"

    llm = Concrete(api_url="http://x", api_key="k")
    assert llm.generate("p", "ctx") == "ok: ctx"
    assert llm.api_url == "http://x"
    assert llm.api_key == "k"
