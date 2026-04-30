from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Main generation model
    llm_main_url: str = "http://localhost:8000/v1"
    llm_main_model: str = "qwen2.5-72b-instruct"
    llm_main_api_key: str = "dummy"

    # Router (small/fast) model
    llm_router_url: str = "http://localhost:8001/v1"
    llm_router_model: str = "qwen2.5-7b-instruct"
    llm_router_api_key: str = "dummy"


_settings: LLMSettings | None = None


def get_settings() -> LLMSettings:
    global _settings
    if _settings is None:
        _settings = LLMSettings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
