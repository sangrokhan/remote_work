from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState


def synthesizer_node(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    result = llm.generate(
        prompt="모든 결과를 종합하여 최종 답변을 작성하세요.",
        context=state.get("refiner_output", state.get("executor_output", "")),
    )
    return {"final_output": result}
