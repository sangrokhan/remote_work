"""
Request model for POST /api/run.
Carries all per-request parameters from the frontend to the flow services.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RunWorkflowRequest(BaseModel):
    run_id: str
    input: str
    model: str = Field(default="Gemma4-E4B-it")
    response_mode: str = Field(default="normal")
    max_tokens: int = Field(default=1024)
    agentic_rag: bool = Field(default=False)
    api_url: str = Field(default="")
    api_key: str = Field(default="")
