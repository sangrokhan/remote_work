from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import TYPE_CHECKING, AsyncGenerator, Generator, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from llm.factory import get_llm

if TYPE_CHECKING:
    from app.models import RunWorkflowRequest

logger = logging.getLogger("workflow_api")


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


def _planner(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="입력을 분석하고 실행 계획을 수립하세요.",
        context=state["llm_input"],
    )
    return {
        "planner_output": result,
        "hop_count": state.get("hop_count", 0) + 1,
        "planner_delay": delay,
    }


def _executor(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="계획을 실행하고 결과를 생성하세요.",
        context=state.get("planner_output", ""),
    )
    return {
        "executor_output": result,
        "hop_count": state.get("hop_count", 0) + 1,
        "executor_delay": delay,
    }


def _refiner(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="실행 결과를 검토하고 개선점을 제안하세요.",
        context=state.get("executor_output", ""),
    )
    return {
        "refiner_output": result,
        "hop_count": state.get("hop_count", 0) + 1,
        "refiner_delay": delay,
    }


def _synthesizer(state: DemoState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    delay = _sleep_node()
    result = llm.generate(
        prompt="모든 결과를 종합하여 최종 답변을 작성하세요.",
        context=state.get("refiner_output", state.get("executor_output", "")),
    )
    return {
        "final_output": result,
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


def run_demo_workflow_events(req: RunWorkflowRequest) -> Generator[dict, None, None]:
    """Execute the workflow and yield progress events per node."""
    logger.debug(
        "workflow starting: model=%s agentic_rag=%s response_mode=%s max_tokens=%s",
        req.model,
        req.agentic_rag,
        req.response_mode,
        req.max_tokens,
    )
    llm = get_llm(req.model, req.api_url, req.api_key)
    config = RunnableConfig(configurable={"llm": llm})
    llm_input = req.input
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
            result = _planner(state, config)
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
            result = _executor(state, config)
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
            result = _refiner(state, config)
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
            result = _synthesizer(state, config)
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
    from llm.models.gauss_o4 import GaussO4
    graph = build_workflow_graph()
    llm = GaussO4(api_url="", api_key="")
    config = RunnableConfig(configurable={"llm": llm})
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
    return graph.invoke(initial_state, config=config)


async def run_demo_workflow_events_async(req: RunWorkflowRequest) -> AsyncGenerator[dict, None]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _run() -> None:
        try:
            for event in run_demo_workflow_events(req):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                _emit_running_event("workflow_error", "scheduler", str(exc), stage="error"),
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_run, daemon=True).start()

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
