"""
Retriever node — performs BGE3-based document retrieval (stub: returns dummy result).
Entry point for the agentic RAG path. Writes retriever_output to state.
"""
from __future__ import annotations

import asyncio
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


class RetrieverNode:
    async def invoke(self, state: AgentState, config: RunnableConfig) -> dict:
        await asyncio.sleep(random.uniform(0.5, 2.0))
        return {"retriever_output": f"[retriever-dummy] 검색 결과: {state.get('input', '')[:50]}"}


retriever_node = RetrieverNode()
