import pytest
from langgraph.core.base import BaseLLM
from langgraph.core.models.gauss_o4 import GaussO4
from langgraph.core.models.gauss_o4_think import GaussO4Think
from langgraph.core.models.gemma4_e4b_it import Gemma4E4BIt


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


def test_gauss_o4_generate():
    llm = GaussO4(api_url="http://example.com", api_key="key")
    result = llm.generate(prompt="분석하세요", context="사용자 질문")
    assert "[GaussO4]" in result
    assert "사용자 질문" in result


def test_gauss_o4_think_generate():
    llm = GaussO4Think(api_url="http://example.com", api_key="key")
    result = llm.generate(prompt="추론하세요", context="복잡한 문제")
    assert "[GaussO4-think]" in result
    assert "복잡한 문제" in result


def test_gemma4_generate():
    llm = Gemma4E4BIt(api_url="http://example.com", api_key="key")
    result = llm.generate(prompt="요약하세요", context="긴 텍스트")
    assert "[Gemma4-E4B-it]" in result
    assert "긴 텍스트" in result


def test_all_models_are_base_llm():
    for cls in [GaussO4, GaussO4Think, Gemma4E4BIt]:
        llm = cls(api_url="", api_key="")
        assert isinstance(llm, BaseLLM)
