"""Reranker settings loaded from environment variables (endpoint URL, model, score threshold)."""
from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_ROOT_DIR = Path(__file__).resolve().parents[2]
_ENV_FILES = [str(_ROOT_DIR / ".env"), ".env"]


class EncoderSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILES,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    encoder_reranker_url: str = Field(
        default="http://localhost:8002/rerank",
        validation_alias=AliasChoices("ENCODER_RERANKER_URL", "RERANKER_URL"),
    )
    encoder_reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        validation_alias=AliasChoices("ENCODER_RERANKER_MODEL", "RERANKER_MODEL"),
    )
    encoder_reranker_backend: str = Field(
        default="local",
        validation_alias=AliasChoices("ENCODER_RERANKER_BACKEND", "RERANKER_BACKEND"),
    )
    encoder_reranker_device: str = Field(
        default="cpu",
        validation_alias=AliasChoices("ENCODER_RERANKER_DEVICE", "RERANKER_DEVICE"),
    )


_settings: EncoderSettings | None = None


def get_settings() -> EncoderSettings:
    global _settings
    if _settings is None:
        _settings = EncoderSettings()
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
