from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DemoState(TypedDict):
    """State schema for the 4-node sequential workflow."""

    llm_input: str
    planner_output: str
    executor_output: str
    refiner_output: str
    final_output: str


def _planner(state: DemoState) -> dict:
    return {
        "planner_output": f"[planner] 계획 생성 완료: {state['llm_input']}",
    }


def _executor(state: DemoState) -> dict:
    return {
        "executor_output": f"[executor] 실행 결과: {state.get('planner_output', '')}",
    }


def _refiner(state: DemoState) -> dict:
    return {
        "refiner_output": f"[refiner] 정제 결과: {state.get('executor_output', '')}",
    }


def _synthesizer(state: DemoState) -> dict:
    return {
        "final_output": (
            f"[synthesizer] 최종 출력: "
            f"{state.get('planner_output', '')} -> "
            f"{state.get('executor_output', '')} -> "
            f"{state.get('refiner_output', '')}"
        )
    }


def build_workflow_graph() -> StateGraph:
    """Build a LangGraph StateGraph with 4 sequential nodes.

    Planner -> Executor -> Refiner -> Synthesizer -> END
    """
    builder = StateGraph(DemoState)

    builder.add_node("planner", _planner)
    builder.add_node("executor", _executor)
    builder.add_node("refiner", _refiner)
    builder.add_node("synthesizer", _synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "refiner")
    builder.add_edge("refiner", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile()


def run_demo_workflow(llm_input: str) -> dict:
    """Run the workflow and return the final graph state."""
    graph = build_workflow_graph()
    initial_state: DemoState = {
        "llm_input": llm_input,
        "planner_output": "",
        "executor_output": "",
        "refiner_output": "",
        "final_output": "",
    }
    return graph.invoke(initial_state)
