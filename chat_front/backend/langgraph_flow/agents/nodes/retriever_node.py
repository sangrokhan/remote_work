"""
Retriever node — logs executor's retrieval results into retriever_history.
Actual retrieval happens in executor_node._execute_retrieve_subtask.
This node deduplicates and accumulates the history, then clears retriever_outputs.
"""
from __future__ import annotations

from typing import Optional

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, update_state


class RetrieverNode:
    def __init__(self):
        self.name = "retriever"
        self.max_retries = 3

    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        return await self.pass_through(state)

    async def pass_through(self, state: AgentState) -> AgentState:
        retriever_outputs = state.get("retriever_outputs", [])
        retriever_history = state.get("retriever_history", [])

        logged_entries = set()
        for history in retriever_history:
            subtask_id = history.get("subtask_id")
            query = history.get("query", "")
            logged_entries.add((subtask_id, query[:100] if query else ""))

        new_entries = []
        for output in retriever_outputs:
            subtask_id = output.get("subtask_id")
            query = output.get("query", "")
            entry_key = (subtask_id, query[:100] if query else "")

            if entry_key not in logged_entries:
                new_entries.append({
                    "subtask_id": subtask_id,
                    "query": query,
                    "result": output.get("result"),
                    "status": output.get("status", "unknown"),
                })
                logged_entries.add(entry_key)

        return update_state(
            state,
            retriever_history=new_entries,
            retriever_outputs=[],
            next="refiner",
        )


retriever_node = RetrieverNode()
