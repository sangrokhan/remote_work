"""
Simple (non-agentic) flow service.

Single-turn: user input → LLM.generate() → result.
No graph traversal. Used when agentic_rag=false.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

from llm.factory import get_llm

if TYPE_CHECKING:
    from app.models import RunWorkflowRequest


class SimpleService:
    async def process(self, req: RunWorkflowRequest) -> AsyncGenerator[dict, None]:
        llm = get_llm(req.model)

        yield {"event": "node_started", "node": "llm", "name": "llm", "stage": "start", "message": "LLM 호출 중"}

        result = llm.generate(prompt=req.input, context="")

        yield {
            "event": "node_finished",
            "node": "llm",
            "name": "llm",
            "stage": "end",
            "message": result,
            "payload": {"final_output": result},
        }

        yield {"event": "workflow_complete", "message": "완료"}
