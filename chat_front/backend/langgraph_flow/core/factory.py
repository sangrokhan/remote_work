"""
LLM factory for langgraph_flow. Exposes MODEL_REGISTRY, get_llm(), and list_models().
list_models() is called by the backend /models endpoint at startup.
"""
from __future__ import annotations

from langgraph_flow.core.base import BaseLLM
from langgraph_flow.core.models.gauss_o4 import GaussO4
from langgraph_flow.core.models.gauss_o4_think import GaussO4Think
from langgraph_flow.core.models.gemma4_e4b_it import Gemma4E4BIt

MODEL_REGISTRY: dict[str, type[BaseLLM]] = {
    "GaussO4": GaussO4,
    "GaussO4-think": GaussO4Think,
    "Gemma4-E4B-it": Gemma4E4BIt,
}


def get_llm(model_name: str, api_url: str = "", api_key: str = "") -> BaseLLM:
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {list(MODEL_REGISTRY)}")
    return cls(api_url=api_url, api_key=api_key)


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
