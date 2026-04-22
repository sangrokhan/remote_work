"""
Synthesizer node — terminal node. Combines refiner/executor output into the final answer.
Writes final_output to state. Always leads to END.
"""
from __future__ import annotations

import asyncio
import random

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.synthesizer import SYNTHESIZER_PROMPT


class SynthesizerNode:
    async def invoke(self, state: AgentState, config: RunnableConfig) -> dict:
        llm = config["configurable"]["llm"]
        await asyncio.sleep(random.uniform(1.0, 5.0))
        context = state.get("refiner_output") or state.get("executor_output", "")
        result = llm.generate(prompt=SYNTHESIZER_PROMPT, context=context)
        return {"final_output": result}


synthesizer_node = SynthesizerNode()
