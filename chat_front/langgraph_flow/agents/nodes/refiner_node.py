from __future__ import annotations

import random
import time

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.refiner import REFINER_PROMPT


def refiner_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(1.0, 5.0))
    result = llm.generate(prompt=REFINER_PROMPT, context=state.get("executor_output", ""))
    return {"refiner_output": result, "hop_count": state.get("hop_count", 0) + 1}
