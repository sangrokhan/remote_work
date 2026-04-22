"""
Abstract base class for all LLM implementations in the backend.
Concrete models declare ENV_URL_KEY / ENV_KEY_KEY to load credentials from .env.
Explicit constructor args override env values (for request-time injection).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class BaseLLM(ABC):
    ENV_URL_KEY: str = ""
    ENV_KEY_KEY: str = ""

    def __init__(self, api_url: str = "", api_key: str = "") -> None:
        self.api_url = api_url or os.getenv(self.ENV_URL_KEY, "")
        self.api_key = api_key or os.getenv(self.ENV_KEY_KEY, "")

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str: ...
