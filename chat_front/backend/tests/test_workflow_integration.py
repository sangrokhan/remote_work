from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig

from llm.models.gauss_o4 import GaussO4
from langgraph_flow.agents.state import AgentState
from langgraph_flow.agents.nodes.planner_node import planner_node
from langgraph_flow.agents.nodes.executor_node import executor_node
from langgraph_flow.agents.nodes.refiner_node import refiner_node
from langgraph_flow.agents.nodes.synthesizer_node import synthesizer_node


@pytest.fixture
def gauss_config():
    llm = GaussO4(api_url="", api_key="")
    return RunnableConfig(configurable={"llm": llm})


@pytest.fixture
def base_state() -> AgentState:
    return AgentState(
        input="테스트 입력",
        agentic_rag=False,
        planner_output="",
        executor_output="",
        refiner_output="",
        retriever_output="",
        var_bindings="",
        final_output="",
        hop_count=0,
    )


def test_planner_uses_llm(gauss_config, base_state):
    result = planner_node(base_state, gauss_config)
    assert "[GaussO4]" in result["planner_output"]
    assert result["hop_count"] == 1


def test_executor_uses_llm(gauss_config, base_state):
    base_state["planner_output"] = "[GaussO4] plan"
    result = executor_node(base_state, gauss_config)
    assert "[GaussO4]" in result["executor_output"]
    assert result["hop_count"] == 1


def test_refiner_uses_llm(gauss_config, base_state):
    base_state["executor_output"] = "[GaussO4] exec"
    result = refiner_node(base_state, gauss_config)
    assert "[GaussO4]" in result["refiner_output"]
    assert result["hop_count"] == 1


def test_synthesizer_uses_llm(gauss_config, base_state):
    base_state["refiner_output"] = "[GaussO4] refined"
    result = synthesizer_node(base_state, gauss_config)
    assert "[GaussO4]" in result["final_output"]
