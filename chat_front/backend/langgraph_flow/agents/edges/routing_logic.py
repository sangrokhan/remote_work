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
    has_success = any(
        h.get("subtask_id") == current_executing_id and h.get("status") == "success"
        for h in retriever_history
    )
    route = "refiner" if has_success else "synthesizer"
    logger.debug("[ROUTE] executor → %s | current_subtask=%s has_success=%s",
                 route, current_executing_id, has_success)
    return route


def route_after_refiner(state: AgentState) -> str:
    if state.get("is_finished", False):
        logger.debug("[ROUTE] refiner → synthesizer | is_finished=True")
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

    # 모든 subtask 결판났으면 종료. retry cap은 refiner가 책임 (subtask당 max 3 → exceeded).
    if all(s.get("id") in completed for s in subtasks):
        logger.debug("[ROUTE] refiner → synthesizer | all subtasks completed=%s", completed)
        return "synthesizer"

    # 안전망: retry_counts 합이 비정상적으로 커지면 강제 종료 (cap 깨졌을 때 무한 루프 방지)
    retry_counts = state.get("retry_counts") or {}
    total_attempts = sum(v for v in retry_counts.values() if isinstance(v, int))
    safety_cap = max(20, len(subtasks) * 5)
    if total_attempts > safety_cap:
        logger.warning("[ROUTE] refiner safety cap hit: total_attempts=%d cap=%d", total_attempts, safety_cap)
        return "synthesizer"

    for subtask in subtasks:
        subtask_id = subtask.get("id")
        if subtask_id in completed:
            continue
        deps = subtask.get("dependencies", [])
        if not deps or all(d in completed for d in deps):
            logger.debug("[ROUTE] refiner → var_binder | next_subtask=%s total_attempts=%d", subtask_id, total_attempts)
            return "var_binder"

    # dep 미해결 + 미완료 subtask 존재 → 데드락 (planner validator가 막아야 함)
    logger.warning("[ROUTE] refiner deadlock: no executable subtask but not all completed=%s", completed)
    return "synthesizer"


ROUTING_FUNCTIONS = {
    "route_after_planner": route_after_planner,
    "route_after_executor": route_after_executor,
    "route_after_refiner": route_after_refiner,
}
