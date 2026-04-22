"""
LLM factory for the backend. Maps model name strings to concrete BaseLLM subclasses.
get_llm() is the single entry point used by flow services.
"""
from __future__ import annotations

from langgraph_flow.core.base import BaseLLM
from llm.models.gauss_o4 import GaussO4
from llm.models.gauss_o4_think import GaussO4Think
from llm.models.gemma4_e4b_it import Gemma4E4BIt

MODEL_REGISTRY: dict[str, type[BaseLLM]] = {
    "GaussO4": GaussO4,
    "GaussO4-think": GaussO4Think,
    "Gemma4-E4B-it": Gemma4E4BIt,
}


def get_llm(model_name: str) -> BaseLLM:
    cls = MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise ValueError(f"Unknown model: {model_name}")
    return cls()
