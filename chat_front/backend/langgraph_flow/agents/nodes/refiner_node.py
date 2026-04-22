"""
Refiner node — reviews executor output and improves quality before synthesis.
Writes refiner_output to state. May loop back to planner via refiner_route.
"""
from __future__ import annotations

import asyncio
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.refiner import REFINER_PROMPT


class RefinerNode:
    async def invoke(self, state: AgentState, config: RunnableConfig) -> dict:
        llm = config["configurable"]["llm"]
        await asyncio.sleep(random.uniform(1.0, 5.0))
        result = llm.generate(prompt=REFINER_PROMPT, context=state.get("executor_output", ""))
        return {"refiner_output": result, "hop_count": state.get("hop_count", 0) + 1}


refiner_node = RefinerNode()
