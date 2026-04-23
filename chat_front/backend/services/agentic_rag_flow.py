"""
Agentic RAG flow service.

Graph: var_constructor → planner → (var_binder | synthesizer)
       var_binder → executor → retriever → (refiner | synthesizer)
       refiner → (var_binder | synthesizer)
Used when agentic_rag=true. LLM injected via RunnableConfig configurable.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncGenerator

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.graph import create_agentic_rag_graph
from langgraph_flow.agents.state import create_initial_state
from llm.factory import get_llm

if TYPE_CHECKING:
    from app.models import RunWorkflowRequest

logger = logging.getLogger(__name__)


class AgenticService:
    async def process(self, req: RunWorkflowRequest) -> AsyncGenerator[dict, None]:
        llm = get_llm(req.model)
        graph = create_agentic_rag_graph()
        state = create_initial_state(req.input)
        config = RunnableConfig(configurable={"llm": llm})

        logger.debug("AgenticService.process: model=%s input_len=%d", req.model, len(req.input))

        final_payload: dict = {}
        async for event in graph.invoke(state, config):
            if event.get("node") == "synthesizer" and event.get("event") == "node_finished":
                final_payload = event.get("payload", {})
            yield event

        logger.debug("AgenticService.process complete: final_payload keys=%s", list(final_payload.keys()))

        yield {
            "event": "workflow_complete",
            "message": "완료 (Agentic RAG)",
            "final_response": final_payload.get("final_response", ""),
            "steps": final_payload.get("retriever_history", []),
            "reference_features": final_payload.get("reference_features", []),
        }
