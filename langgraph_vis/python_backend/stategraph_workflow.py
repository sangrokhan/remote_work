from __future__ import annotations

import random
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class DemoState(TypedDict):
    """State schema for the 4-node workflow."""

    llm_input: str
    planner_output: str
    executor_output: str
    refiner_output: str
    final_output: str
    hop_count: int


def _planner(state: DemoState) -> dict:
    return {
        "planner_output": f"[planner] 계획 생성 완료: {state['llm_input']}",
        "hop_count": state.get("hop_count", 0) + 1,
    }


def _executor(state: DemoState) -> dict:
    return {
        "executor_output": f"[executor] 실행 결과: {state.get('planner_output', '')}",
        "hop_count": state.get("hop_count", 0) + 1,
    }


def _refiner(state: DemoState) -> dict:
    return {
        "refiner_output": f"[refiner] 정제 결과: {state.get('executor_output', '')}",
        "hop_count": state.get("hop_count", 0) + 1,
    }


def _synthesizer(state: DemoState) -> dict:
    return {
        "final_output": (
            f"[synthesizer] 최종 출력: "
            f"{state.get('planner_output', '')} -> "
            f"{state.get('executor_output', '')} -> "
            f"{state.get('refiner_output', '')}"
        ),
        "hop_count": state.get("hop_count", 0) + 1,
    }


def _executor_route(state: DemoState) -> str:
    if state.get("hop_count", 0) >= 6:
        return "to_synthesizer"
    return "to_refiner" if random.random() < 0.55 else "to_synthesizer"


def _refiner_route(state: DemoState) -> str:
    if state.get("hop_count", 0) >= 10:
        return "to_planner"
    return "to_synthesizer" if random.random() < 0.5 else "to_planner"


def build_workflow_graph() -> StateGraph:
    """Build a LangGraph StateGraph with conditional branches.

    Planner -> Executor -> (Refiner | Synthesizer)
    Refiner -> (Planner | Synthesizer)
    """
    builder = StateGraph(DemoState)

    builder.add_node("planner", _planner)
    builder.add_node("executor", _executor)
    builder.add_node("refiner", _refiner)
    builder.add_node("synthesizer", _synthesizer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges(
        "executor",
        _executor_route,
        {
            "to_refiner": "refiner",
            "to_synthesizer": "synthesizer",
        },
    )
    builder.add_conditional_edges(
        "refiner",
        _refiner_route,
        {
            "to_synthesizer": "synthesizer",
            "to_planner": "planner",
        },
    )
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
        "hop_count": 0,
    }
    return graph.invoke(initial_state)
