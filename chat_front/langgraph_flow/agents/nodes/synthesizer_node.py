from __future__ import annotations

import random
import time

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.synthesizer import SYNTHESIZER_PROMPT


def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(1.0, 5.0))
    context = state.get("refiner_output") or state.get("executor_output", "")
    result = llm.generate(prompt=SYNTHESIZER_PROMPT, context=context)
    return {"final_output": result}
