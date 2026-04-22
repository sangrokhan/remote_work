from __future__ import annotations

from typing import TYPE_CHECKING, AsyncGenerator

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.graph import create_agentic_rag_graph
from langgraph_flow.core.factory import get_llm

if TYPE_CHECKING:
    from backend.app.models import RunWorkflowRequest


async def run_agentic_rag_flow(req: RunWorkflowRequest) -> AsyncGenerator[dict, None]:
    llm = get_llm(req.model, req.api_url, req.api_key)
    graph = create_agentic_rag_graph(agentic_rag=True)

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

    async for event in graph.invoke(state, config):
        yield event

    yield {"event": "workflow_complete", "message": "완료 (Agentic RAG)"}
