from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def executor_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    result = llm.generate(
        prompt="계획을 실행하고 결과를 생성하세요.",
        context=state.get("planner_output", ""),
    )
    return {"executor_output": result, "hop_count": state.get("hop_count", 0) + 1}
