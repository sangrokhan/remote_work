"""
Abstract base class for all LLM implementations in the backend.
Inherits BaseLanguageModel so all models are LangChain-compatible.
Concrete models declare ENV_URL_KEY / ENV_KEY_KEY / MODEL_NAME,
create their own AsyncOpenAI client, and implement invoke() / ainvoke().
"""
from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, ClassVar, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from pydantic import Field, model_validator

load_dotenv()

LanguageModelInput = str | list[BaseMessage] | PromptValue


class BaseLLM(BaseLanguageModel):
    ENV_URL_KEY: ClassVar[str] = ""
    ENV_KEY_KEY: ClassVar[str] = ""
    MODEL_NAME: ClassVar[str] = ""

    api_url: str = Field(default="")
    api_key: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _load_env(cls, values: dict) -> dict:
        load_dotenv()
        if not values.get("api_url"):
            values["api_url"] = os.getenv(cls.ENV_URL_KEY, "")
        if not values.get("api_key"):
            values["api_key"] = os.getenv(cls.ENV_KEY_KEY, "")
        return values

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__.lower()

    @staticmethod
    def _to_str(input: LanguageModelInput) -> str:
        if isinstance(input, str):
            return input
        if isinstance(input, PromptValue):
            return input.to_string()
        return "\n".join(m.content for m in input if hasattr(m, "content"))

    def generate_prompt(self, prompts: List[PromptValue], stop=None, callbacks=None, **kwargs) -> LLMResult:
        generations = [[Generation(text=self.invoke(p.to_string()))] for p in prompts]
        return LLMResult(generations=generations)

    async def agenerate_prompt(self, prompts: List[PromptValue], stop=None, callbacks=None, **kwargs) -> LLMResult:
        generations = [[Generation(text=await self.ainvoke(p.to_string()))] for p in prompts]
        return LLMResult(generations=generations)

    @abstractmethod
    def invoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str: ...

    @abstractmethod
    async def ainvoke(self, input: LanguageModelInput, config: Optional[Any] = None, **kwargs: Any) -> str: ...
