"""
Conditional edge functions for the AgenticRAG graph.

route_after_planner   — routes to var_binder (subtasks exist) or synthesizer (none)
route_after_executor  — routes to refiner or synthesizer based on retrieval result
route_after_refiner   — loops back to var_binder or exits to synthesizer
"""
from __future__ import annotations

import random

from langgraph_flow.agents.state import AgentState


def route_after_planner(state: AgentState) -> str:
    if state.get("hop_count", 0) >= 6:
        return "synthesizer"
    return "var_binder" if random.random() < 0.5 else "synthesizer"


def route_after_executor(state: AgentState) -> str:
    if state.get("hop_count", 0) >= 6:
        return "synthesizer"
    return "refiner" if random.random() < 0.5 else "synthesizer"


def route_after_refiner(state: AgentState) -> str:
    if state.get("hop_count", 0) >= 10:
        return "synthesizer"
    return "var_binder" if random.random() < 0.5 else "synthesizer"
