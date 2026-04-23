"""
Var constructor node — constructs binding context from user query.
InputState → AgentState 변환 및 바인딩 컨텍스트 추출 전담.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, InputState, update_state
from langgraph_flow.prompts.var_constructor import CONSTRUCTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class VarConstructorNode:
    """Variable Constructor 노드 클래스 - 바인딩 컨텍스트 추출 및 InputState 변환 전담"""

    def __init__(self):
        self.name = "var_constructor"

    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        # InputState에서 AgentState로 변환 (진입점)
        state = self._ensure_agent_state(state)
        return await construct_binding_context(state)

    def _ensure_agent_state(self, state: AgentState) -> AgentState:
        """
        InputState를 AgentState로 변환

        LangGraph Studio WebUI에서 InputState(user_query만 포함)가 입력되면
        전체 AgentState로 변환하여 내부 필드들을 초기화
        """
        # user_query 추출 (InputState와 AgentState 모두 user_query 필드 사용)
        # socket_id 기반 변환 로직 제거 — stateless 구조 사용
        return state


async def construct_binding_context(
    state: AgentState,
    llm: Optional[BaseLanguageModel] = None,
) -> AgentState:
    """
    바인딩 컨텍스트를 추출하는 함수

    LLM이 주입되지 않은 경우 기본 구조 반환 후 planner로 이동
    """
    # LLM이 제공되지 않은 경우 (노드에 LLM 주입 없음) 기본 응답 반환
    if llm is None:
        default_context = {
            "query_entities": {"main_concept": ["task_0.main_concept"]},
            "previous_features": [],
            "explicit_dependencies": [],
        }
        return update_state(state, binding_context=default_context, next="planner")

    try:
        # 바인딩 컨텍스트 추출을 위한 메시지 구성
        messages = [
            SystemMessage(content=CONSTRUCTOR_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {state.get('user_query', '')}"),
        ]

        # LLM 호출 — ainvoke는 str 반환
        response = await llm.ainvoke(messages)
        content = response if isinstance(response, str) else getattr(response, "content", "{}")
        binding_context = json.loads(content)

        # 기본 구조 보장
        if "query_entities" not in binding_context:
            binding_context["query_entities"] = {"features": [], "keywords": []}
        if "previous_features" not in binding_context:
            binding_context["previous_features"] = []
        if "explicit_dependencies" not in binding_context:
            binding_context["explicit_dependencies"] = []

        return update_state(state, binding_context=binding_context, next="planner")

    except Exception as e:
        # 에러 발생 시 planner로 이동
        logger.error("VarConstructor error: %s", e)
        return update_state(state, next="planner")


var_constructor_node = VarConstructorNode()
