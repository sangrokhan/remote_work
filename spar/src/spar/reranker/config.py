from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EncoderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    encoder_reranker_url: str = "http://localhost:8002/rerank"
    encoder_reranker_model: str = "BAAI/bge-reranker-v2-m3"


_settings: EncoderSettings | None = None


def get_settings() -> EncoderSettings:
    global _settings
    if _settings is None:
        _settings = EncoderSettings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
