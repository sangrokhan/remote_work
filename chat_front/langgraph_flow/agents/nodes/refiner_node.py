from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def refiner_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    result = llm.generate(
        prompt="실행 결과를 검토하고 개선점을 제안하세요.",
        context=state.get("executor_output", ""),
    )
    return {"refiner_output": result, "hop_count": state.get("hop_count", 0) + 1}
