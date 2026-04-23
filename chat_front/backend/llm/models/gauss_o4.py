"""GaussO4 backend LLM."""
from __future__ import annotations

from llm.base import BaseLLM


class GaussO4(BaseLLM):
    ENV_URL_KEY = "GAUSS_O4_API_URL"
    ENV_KEY_KEY = "GAUSS_O4_API_KEY"
    MODEL_NAME = "gauss-o4-instruct"
