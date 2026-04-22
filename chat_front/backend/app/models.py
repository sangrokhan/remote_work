from __future__ import annotations

from pydantic import BaseModel, Field


class RunWorkflowRequest(BaseModel):
    run_id: str
    input: str
    model: str = Field(default="GaussO4")
    response_mode: str = Field(default="normal")
    max_tokens: int = Field(default=1024)
    agentic_rag: bool = Field(default=False)
    api_url: str = Field(default="")
    api_key: str = Field(default="")
