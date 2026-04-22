from __future__ import annotations

import random
import time
from typing import Generator, TypedDict

from langgraph.graph import END, START, StateGraph


class DemoState(TypedDict):
    """State schema for the 4-node workflow."""

    llm_input: str
    planner_output: str
    executor_output: str
    refiner_output: str
    final_output: str
    hop_count: int
    planner_delay: float
    executor_delay: float
    refiner_delay: float
    synthesizer_delay: float


def _random_node_delay() -> float:
    return random.uniform(1.0, 5.0)


def _sleep_node() -> float:
    delay = _random_node_delay()
    time.sleep(delay)
    return delay


def _planner(state: DemoState) -> dict:
    delay = _sleep_node()
    return {
        "planner_output": f"[planner] 계획 생성 완료: {state['llm_input']} ({delay:.1f}s)",
        "hop_count": state.get("hop_count", 0) + 1,
        "planner_delay": delay,
    }


def _executor(state: DemoState) -> dict:
    delay = _sleep_node()
    return {
        "executor_output": f"[executor] 실행 결과: {state.get('planner_output', '')} ({delay:.1f}s)",
        "hop_count": state.get("hop_count", 0) + 1,
        "executor_delay": delay,
    }


def _refiner(state: DemoState) -> dict:
    delay = _sleep_node()
    return {
        "refiner_output": f"[refiner] 정제 결과: {state.get('executor_output', '')} ({delay:.1f}s)",
        "hop_count": state.get("hop_count", 0) + 1,
        "refiner_delay": delay,
    }


def _synthesizer(state: DemoState) -> dict:
    delay = _sleep_node()
    return {
        "final_output": (
            f"[synthesizer] 최종 출력: "
            f"{state.get('planner_output', '')} -> "
            f"{state.get('executor_output', '')} -> "
            f"{state.get('refiner_output', '')} "
            f"({delay:.1f}s)"
        ),
        "hop_count": state.get("hop_count", 0) + 1,
        "synthesizer_delay": delay,
    }


def _executor_route(state: DemoState) -> str:
    if state.get("hop_count", 0) >= 6:
        return "to_synthesizer"
    return "to_refiner" if random.random() < 0.5 else "to_synthesizer"


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


def _emit_running_event(
    event_type: str,
    node_name: str,
    message: str,
    payload: dict | None = None,
    stage: str | None = None,
) -> dict:
    event: dict[str, object] = {
        "event": event_type,
        "node": node_name,
        "name": node_name,
        "message": message,
    }
    if stage:
        event["stage"] = stage
    if payload is not None:
        event["payload"] = payload
    return event


def run_demo_workflow_events(llm_input: str) -> Generator[dict, None, None]:
    """Execute the workflow and yield progress events per node."""
    state: DemoState = {
        "llm_input": llm_input,
        "planner_output": "",
        "executor_output": "",
        "refiner_output": "",
        "final_output": "",
        "hop_count": 0,
        "planner_delay": 0.0,
        "executor_delay": 0.0,
        "refiner_delay": 0.0,
        "synthesizer_delay": 0.0,
    }

    current = "planner"
    iteration = 0
    max_iterations = 20

    while iteration < max_iterations:
        iteration += 1

        if current == "planner":
            yield _emit_running_event(
                "node_started",
                "planner",
                "planner 실행됨",
                stage="start",
            )
            result = _planner(state)
            state.update(result)
            yield _emit_running_event(
                "node_finished",
                "planner",
                result["planner_output"],
                {"planner_delay": state["planner_delay"]},
                stage="end",
            )
            current = "executor"
            continue

        if current == "executor":
            yield _emit_running_event(
                "node_started",
                "executor",
                "executor 실행됨",
                stage="start",
            )
            result = _executor(state)
            state.update(result)
            yield _emit_running_event(
                "node_finished",
                "executor",
                result["executor_output"],
                {"executor_delay": state["executor_delay"]},
                stage="end",
            )
            route = _executor_route(state)
            yield _emit_running_event(
                "node_routed",
                "executor",
                f"next={route}",
                {"route": route},
                stage="routing",
            )
            current = "refiner" if route == "to_refiner" else "synthesizer"
            continue

        if current == "refiner":
            yield _emit_running_event(
                "node_started",
                "refiner",
                "refiner 실행됨",
                stage="start",
            )
            result = _refiner(state)
            state.update(result)
            yield _emit_running_event(
                "node_finished",
                "refiner",
                result["refiner_output"],
                {"refiner_delay": state["refiner_delay"]},
                stage="end",
            )
            route = _refiner_route(state)
            yield _emit_running_event(
                "node_routed",
                "refiner",
                f"next={route}",
                {"route": route},
                stage="routing",
            )
            current = "planner" if route == "to_planner" else "synthesizer"
            continue

        if current == "synthesizer":
            yield _emit_running_event(
                "node_started",
                "synthesizer",
                "synthesizer 실행됨",
                stage="start",
            )
            result = _synthesizer(state)
            state.update(result)
            yield _emit_running_event(
                "node_finished",
                "synthesizer",
                result["final_output"],
                {"synthesizer_delay": state["synthesizer_delay"]},
                stage="end",
            )
            yield _emit_running_event(
                "workflow_complete",
                "synthesizer",
                "workflow 실행 완료",
                {"final_output": state["final_output"], "hop_count": state["hop_count"]},
                stage="end",
            )
            break

    if iteration >= max_iterations:
        yield _emit_running_event(
            "workflow_error",
            "scheduler",
            "workflow 반복 횟수 상한으로 실행이 중단되었습니다.",
            {"hop_count": state["hop_count"]},
            stage="error",
        )
        return

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
        "planner_delay": 0.0,
        "executor_delay": 0.0,
        "refiner_delay": 0.0,
        "synthesizer_delay": 0.0,
    }
    return graph.invoke(initial_state)
