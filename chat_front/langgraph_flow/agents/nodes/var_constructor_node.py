from __future__ import annotations

import random
import time

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState
from langgraph_flow.prompts.var_constructor import VAR_CONSTRUCTOR_PROMPT


def var_constructor_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    time.sleep(random.uniform(0.5, 2.0))
    result = llm.generate(prompt=VAR_CONSTRUCTOR_PROMPT, context=state.get("retriever_output", ""))
    return {"var_bindings": result}
