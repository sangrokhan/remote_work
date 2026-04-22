"""
Var constructor node — constructs query variables from retriever output.
Reads retriever_output, writes var_bindings to state.
"""
from __future__ import annotations

import asyncio
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.var_constructor import VAR_CONSTRUCTOR_PROMPT


class VarConstructorNode:
    async def invoke(self, state: AgentState, config: RunnableConfig) -> dict:
        llm = config["configurable"]["llm"]
        await asyncio.sleep(random.uniform(0.5, 2.0))
        result = llm.generate(prompt=VAR_CONSTRUCTOR_PROMPT, context=state.get("retriever_output", ""))
        return {"var_bindings": result}


var_constructor_node = VarConstructorNode()
