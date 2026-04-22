from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def planner_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    result = llm.generate(
        prompt="입력을 분석하고 실행 계획을 수립하세요.",
        context=state["input"],
    )
    return {"planner_output": result, "hop_count": state.get("hop_count", 0) + 1}
