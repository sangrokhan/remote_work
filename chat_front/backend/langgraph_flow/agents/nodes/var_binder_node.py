"""
Var binder node — binds constructed variables into planner context.
Reads var_bindings, overwrites retriever_output so planner sees enriched context.
"""
from __future__ import annotations

import asyncio
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.var_binder import VAR_BINDER_PROMPT


class VarBinderNode:
    async def invoke(self, state: AgentState, config: RunnableConfig) -> dict:
        llm = config["configurable"]["llm"]
        await asyncio.sleep(random.uniform(0.5, 2.0))
        result = llm.generate(prompt=VAR_BINDER_PROMPT, context=state.get("var_bindings", ""))
        return {"retriever_output": result}


var_binder_node = VarBinderNode()
