from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableConfig

from llm.models.gauss_o4 import GaussO4
from stategraph_workflow import DemoState, _planner, _executor, _refiner, _synthesizer


@pytest.fixture
def gauss_config():
    llm = GaussO4(api_url="", api_key="")
    return RunnableConfig(configurable={"llm": llm})


@pytest.fixture
def base_state() -> DemoState:
    return DemoState(
        llm_input="테스트 입력",
        planner_output="",
        executor_output="",
        refiner_output="",
        final_output="",
        hop_count=0,
        planner_delay=0.0,
        executor_delay=0.0,
        refiner_delay=0.0,
        synthesizer_delay=0.0,
    )


def test_planner_uses_llm(gauss_config, base_state):
    result = _planner(base_state, gauss_config)
    assert "[GaussO4]" in result["planner_output"]
    assert result["hop_count"] == 1


def test_executor_uses_llm(gauss_config, base_state):
    base_state["planner_output"] = "[GaussO4] plan"
    result = _executor(base_state, gauss_config)
    assert "[GaussO4]" in result["executor_output"]


def test_refiner_uses_llm(gauss_config, base_state):
    base_state["executor_output"] = "[GaussO4] exec"
    result = _refiner(base_state, gauss_config)
    assert "[GaussO4]" in result["refiner_output"]


def test_synthesizer_uses_llm(gauss_config, base_state):
    base_state["refiner_output"] = "[GaussO4] refined"
    result = _synthesizer(base_state, gauss_config)
    assert "[GaussO4]" in result["final_output"]
