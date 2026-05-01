from __future__ import annotations

from enum import Enum

from spar.llm.client import LLMClient
from spar.llm.config import LLMSettings
from spar.llm.fallback import FallbackLLMClient, LLMBackend
from spar.llm.gemini_cli import GeminiCliClient


class LLMRole(str, Enum):
    MAIN = "main"
    ROUTER = "router"


class LLMFactory:
    @staticmethod
    def create(role: LLMRole, settings: LLMSettings) -> LLMBackend:
        if role is LLMRole.MAIN:
            primary = LLMClient(
                base_url=settings.llm_main_url,
                model=settings.llm_main_model,
                api_key=settings.llm_main_api_key,
            )
        elif role is LLMRole.ROUTER:
            primary = LLMClient(
                base_url=settings.llm_router_url,
                model=settings.llm_router_model,
                api_key=settings.llm_router_api_key,
            )
        else:
            raise ValueError(f"Unknown LLM role: {role!r}")

        if settings.gemini_cli_fallback_enabled:
            gemini = GeminiCliClient(
                binary=settings.gemini_cli_binary,
                timeout=settings.gemini_cli_timeout_seconds,
            )
            return FallbackLLMClient(primary, gemini)
        return primary
