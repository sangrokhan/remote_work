from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def var_constructor_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: construct query variables from retriever output
    return {"var_bindings": ""}
