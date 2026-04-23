"""
LLM factory for the backend. MODEL_REGISTRY holds singleton instances created at import time.
get_llm() returns the cached instance — no per-request allocation.
"""
from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel

from llm.models.gauss_o4 import GaussO4
from llm.models.gauss_o4_think import GaussO4Think
from llm.models.gemma4_e4b_it import Gemma4E4BIt

MODEL_REGISTRY: dict[str, BaseLanguageModel] = {
    "GaussO4": GaussO4(),
    "GaussO4-think": GaussO4Think(),
    "Gemma4-E4B-it": Gemma4E4BIt(),
}


def get_llm(model_name: str) -> BaseLanguageModel:
    llm = MODEL_REGISTRY.get(model_name)
    if llm is None:
        raise ValueError(f"Unknown model: {model_name}")
    return llm


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
