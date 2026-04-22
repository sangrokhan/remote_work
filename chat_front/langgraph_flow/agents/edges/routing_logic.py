from __future__ import annotations

import random

from langgraph_flow.agents.state import AgentState


def start_route(state: AgentState) -> str:
    return "retriever" if state.get("agentic_rag") else "planner"


def executor_route(state: AgentState) -> str:
    if state.get("hop_count", 0) >= 6:
        return "to_synthesizer"
    return "to_refiner" if random.random() < 0.5 else "to_synthesizer"


def refiner_route(state: AgentState) -> str:
    if state.get("hop_count", 0) >= 10:
        return "to_planner"
    return "to_synthesizer" if random.random() < 0.5 else "to_planner"
