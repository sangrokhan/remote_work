"""
AgentState — shared state TypedDict passed between all LangGraph nodes.
Every node reads from and writes to this dict; LangGraph merges partial returns.
"""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    input: str
    agentic_rag: bool
    planner_output: str
    executor_output: str
    refiner_output: str
    retriever_output: str
    var_bindings: str
    final_output: str
    hop_count: int
