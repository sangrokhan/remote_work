"""
Abstract base class for all LLM implementations.
Inherits BaseLanguageModel for type compatibility.
Creates a ChatOpenAI instance in model_post_init and delegates all calls to it.
Concrete models declare ENV_URL_KEY / ENV_KEY_KEY / MODEL_NAME.
"""
from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_openai import ChatOpenAI
from pydantic import Field, PrivateAttr, model_validator

load_dotenv()


class BaseLLM(BaseLanguageModel):
    ENV_URL_KEY: ClassVar[str] = ""
    ENV_KEY_KEY: ClassVar[str] = ""
    MODEL_NAME: ClassVar[str] = ""

    api_url: str = Field(default="")
    api_key: str = Field(default="")
    temperature: float = Field(default=0.7)
    default_headers: dict = Field(default_factory=dict)

    _llm: ChatOpenAI = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _load_env(cls, values: dict) -> dict:
        load_dotenv()
        if not values.get("api_url"):
            values["api_url"] = os.getenv(cls.ENV_URL_KEY, "")
        if not values.get("api_key"):
            values["api_key"] = os.getenv(cls.ENV_KEY_KEY, "")
        return values

    def model_post_init(self, __context: Any) -> None:
        if not self.api_url:
            raise ValueError(f"{self.__class__.__name__}: api_url is not set (check {self.ENV_URL_KEY} in backend/.env)")
        headers: Dict[str, str] = {"Authorization": f"Basic {self.api_key}"} if self.api_key else {}
        headers.update(self.default_headers)
        self._llm = ChatOpenAI(
            model=self.MODEL_NAME,
            base_url=self.api_url,
            api_key=self.api_key or "dummy",
            temperature=self.temperature,
            default_headers=headers or None,
        )

    @property
    def _llm_type(self) -> str:
        return self.__class__.__name__.lower()

    def generate_prompt(self, prompts: List[PromptValue], stop=None, callbacks=None, **kwargs) -> LLMResult:
        return self._llm.generate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)

    async def agenerate_prompt(self, prompts: List[PromptValue], stop=None, callbacks=None, **kwargs) -> LLMResult:
        return await self._llm.agenerate_prompt(prompts, stop=stop, callbacks=callbacks, **kwargs)

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any):
        return self._llm.invoke(input, config=config, **kwargs)

    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any):
        return await self._llm.ainvoke(input, config=config, **kwargs)
