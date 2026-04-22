"""
Planner node — analyzes user input and produces a step-by-step execution plan.
Writes planner_output to state. LLM injected via config["configurable"]["llm"].
"""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


class PlannerNode:
    async def invoke(self, state: AgentState, config: RunnableConfig) -> dict:
        llm = config["configurable"]["llm"]
        result = llm.generate(
            prompt="입력을 분석하고 실행 계획을 수립하세요.",
            context=state["input"],
        )
        return {"planner_output": result, "hop_count": state.get("hop_count", 0) + 1}


planner_node = PlannerNode()
