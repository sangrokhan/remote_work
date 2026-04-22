"""
Retriever node — performs BGE3-based document retrieval (stub: returns dummy result).
Entry point for the agentic RAG path. Writes retriever_output to state.
"""
from __future__ import annotations

import random
import time

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def retriever_node(state: AgentState, config: RunnableConfig) -> dict:
    time.sleep(random.uniform(0.5, 2.0))
    return {"retriever_output": f"[retriever-dummy] 검색 결과: {state.get('input', '')[:50]}"}
