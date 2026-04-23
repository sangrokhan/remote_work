"""
Conditional edge functions for the AgenticRAG graph.

route_after_planner  — subtasks exist → var_binder; else → synthesizer
route_after_executor — current subtask has retriever result → refiner; else → synthesizer
route_after_refiner  — next runnable subtask exists → var_binder; all done → synthesizer
"""
from __future__ import annotations

from langgraph_flow.agents.state import AgentState


def route_after_planner(state: AgentState) -> str:
    if state.get("is_finished", False):
        return "synthesizer"
    return "var_binder" if state.get("subtasks") else "synthesizer"


def route_after_executor(state: AgentState) -> str:
    retriever_history = state.get("retriever_history", [])
    current_executing_id = state.get("current_executing_subtask_id")
    has_result = any(h.get("subtask_id") == current_executing_id for h in retriever_history)
    return "refiner" if has_result else "synthesizer"


def route_after_refiner(state: AgentState) -> str:
    current_step = state.get("current_step", 0) + 1
    if state.get("is_finished", False) or current_step >= state.get("max_steps", 10):
        return "synthesizer"

    subtasks = state.get("subtasks", [])
    if not subtasks:
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
            return "var_binder"

    return "synthesizer"


ROUTING_FUNCTIONS = {
    "route_after_planner": route_after_planner,
    "route_after_executor": route_after_executor,
    "route_after_refiner": route_after_refiner,
}
