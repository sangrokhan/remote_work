from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def var_binder_node(state: AgentState, config: RunnableConfig) -> dict:
    # TODO: bind variables into planner context
    return {}
