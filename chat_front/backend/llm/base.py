from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, context: str) -> str: ...
