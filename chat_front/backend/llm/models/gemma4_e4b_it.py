"""Gemma4-E4B-it backend LLM."""
from __future__ import annotations

from llm.base import BaseLLM


class Gemma4E4BIt(BaseLLM):
    ENV_URL_KEY = "GEMMA4_E4B_IT_API_URL"
    ENV_KEY_KEY = "GEMMA4_E4B_IT_API_KEY"
    MODEL_NAME = "gemma-4-e4b-it"
