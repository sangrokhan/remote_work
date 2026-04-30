from __future__ import annotations

from enum import Enum

from spar.llm.client import LLMClient
from spar.llm.config import LLMSettings


class LLMRole(str, Enum):
    MAIN = "main"
    ROUTER = "router"


class LLMFactory:
    @staticmethod
    def create(role: LLMRole, settings: LLMSettings) -> LLMClient:
        if role is LLMRole.MAIN:
            return LLMClient(
                base_url=settings.llm_main_url,
                model=settings.llm_main_model,
                api_key=settings.llm_main_api_key,
            )
        if role is LLMRole.ROUTER:
            return LLMClient(
                base_url=settings.llm_router_url,
                model=settings.llm_router_model,
                api_key=settings.llm_router_api_key,
            )
        raise ValueError(f"Unknown LLM role: {role!r}")
