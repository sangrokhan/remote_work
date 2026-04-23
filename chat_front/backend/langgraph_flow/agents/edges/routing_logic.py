"""
Conditional edge functions for the AgenticRAG graph.

route_after_planner  — subtasks exist → var_binder; else → synthesizer
route_after_executor — current subtask has retriever result → refiner; else → synthesizer
route_after_refiner  — next runnable subtask exists → var_binder; all done → synthesizer
"""
from __future__ import annotations

import logging

from langgraph_flow.agents.state import AgentState

logger = logging.getLogger(__name__)


def route_after_planner(state: AgentState) -> str:
    if state.get("is_finished", False):
        route = "synthesizer"
    else:
        route = "var_binder" if state.get("subtasks") else "synthesizer"
    logger.debug("[ROUTE] planner → %s | subtasks=%d is_finished=%s",
                 route, len(state.get("subtasks", [])), state.get("is_finished"))
    return route


def route_after_executor(state: AgentState) -> str:
    retriever_history = state.get("retriever_history", [])
    current_executing_id = state.get("current_executing_subtask_id")
    has_result = any(h.get("subtask_id") == current_executing_id for h in retriever_history)
    route = "refiner" if has_result else "synthesizer"
    logger.debug("[ROUTE] executor → %s | current_subtask=%s has_result=%s",
                 route, current_executing_id, has_result)
    return route


def route_after_refiner(state: AgentState) -> str:
    current_step = state.get("current_step", 0) + 1
    if state.get("is_finished", False) or current_step >= state.get("max_steps", 10):
        logger.debug("[ROUTE] refiner → synthesizer | step=%d max=%d is_finished=%s",
                     current_step, state.get("max_steps", 10), state.get("is_finished"))
        return "synthesizer"

    subtasks = state.get("subtasks", [])
    if not subtasks:
        logger.debug("[ROUTE] refiner → synthesizer | no subtasks")
        return "synthesizer"

    completed = {
        s.get("id")
        for s in subtasks
        if s.get("verdict") is True or s.get("verdict") == "exceeded"
    }

    for i, subtask in enumerate(subtasks):
        task_id = subtask.get("id", i)
        if task_id in completed:
            continue
        deps = subtask.get("dependencies", [])
        if not deps or all(d in completed for d in deps):
            logger.debug("[ROUTE] refiner → var_binder | next_subtask=%s step=%d", task_id, current_step)
            return "var_binder"

    logger.debug("[ROUTE] refiner → synthesizer | all subtasks completed=%s", completed)
    return "synthesizer"


ROUTING_FUNCTIONS = {
    "route_after_planner": route_after_planner,
    "route_after_executor": route_after_executor,
    "route_after_refiner": route_after_refiner,
}
