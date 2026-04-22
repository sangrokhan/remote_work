"""
Executor node — executes each step of the planner's plan and generates results.
Writes executor_output to state.
"""
from __future__ import annotations

import asyncio
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.executor import EXECUTOR_PROMPT


class ExecutorNode:
    async def invoke(self, state: AgentState, config: RunnableConfig) -> dict:
        llm = config["configurable"]["llm"]
        await asyncio.sleep(random.uniform(1.0, 5.0))
        result = llm.generate(prompt=EXECUTOR_PROMPT, context=state.get("planner_output", ""))
        return {"executor_output": result, "hop_count": state.get("hop_count", 0) + 1}


executor_node = ExecutorNode()
