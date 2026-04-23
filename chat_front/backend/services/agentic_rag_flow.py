"""
Agentic RAG flow service.

Multi-hop graph: retriever → var_constructor → var_binder → planner → executor → (refiner | synthesizer).
Used when agentic_rag=true. LLM injected via RunnableConfig configurable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.graph import create_agentic_rag_graph
from llm.factory import get_llm

if TYPE_CHECKING:
    from app.models import RunWorkflowRequest


class AgenticService:
    async def process(self, req: RunWorkflowRequest) -> AsyncGenerator[dict, None]:
        llm = get_llm(req.model)
        graph = create_agentic_rag_graph()

        state = {
            "input": req.input,
            "agentic_rag": True,
            "planner_output": "",
            "executor_output": "",
            "refiner_output": "",
            "retriever_output": "",
            "var_bindings": "",
            "final_output": "",
            "hop_count": 0,
        }
        config = RunnableConfig(configurable={"llm": llm})

        async for event in graph.astream(state, config):
            yield event

        yield {"event": "workflow_complete", "message": "완료 (Agentic RAG)"}
