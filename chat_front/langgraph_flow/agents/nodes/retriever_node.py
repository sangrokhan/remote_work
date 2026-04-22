from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def retriever_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: implement BGE3-based retrieval
    return {"retriever_output": "", "hop_count": state.get("hop_count", 0) + 1}
