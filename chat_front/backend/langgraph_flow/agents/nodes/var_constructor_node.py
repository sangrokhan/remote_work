# langgraph_agenticrag/src/agents/nodes/var_constructor_node.py

import json
from typing import Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, InputState, create_initial_state, update_state
from langgraph_flow.prompts.var_constructor import CONSTRUCTOR_SYSTEM_PROMPT


class VarConstructorNode:
    """Variable Constructor 노드 클래스 - 바인딩 컨텍스트 추출 및 InputState 변환 전담"""

    def __init__(self):
        self.name = "var_constructor"

    async def invoke(self, state: AgentState,
                     config: Optional[RunnableConfig] = None) -> AgentState:
        """
        Variable Constructor 노드 실행
        
        Args:
            state: 현재 에이전트 상태 (InputState에서 변환될 수 있음)
            config: 설정 정보 (선택적)
            
        Returns:
            업데이트된 상태
        """
        # InputState에서 AgentState로 변환 (진입점)
        state = self._ensure_agent_state(state)

        return await construct_binding_context(state)

    def _ensure_agent_state(self, state: AgentState) -> AgentState:
        """
        InputState를 AgentState로 변환
        
        LangGraph Studio WebUI에서 InputState(user_query만 포함)가 입력되면
        전체 AgentState로 변환하여 내부 필드들을 초기화
        
        Args:
            state: 현재 상태 (InputState 또는 AgentState)
            
        Returns:
            AgentState (모든 필드가 초기화된 상태)
        """
        # user_query 추출 (InputState와 AgentState 모두 user_query 필드 사용)
        user_query = state.get("user_query", "")
        # socket_id = state.get("socket_id")

        # # socket_id가 없으면 InputState에서 들어온 것으로 간주
        # if not socket_id:
        #     print(f"=== DEBUG: Converting InputState to AgentState ===")
        #     print(f"Input user_query: '{user_query}'")
        #     print(f"State keys: {list(state.keys())}")
        #     return create_initial_state(user_query, "studio_session")

        # 이미 AgentState인 경우 그대로 반환
        return state


async def construct_binding_context(state: AgentState,
                                    llm: Optional[BaseLanguageModel] = None) -> AgentState:
    """
    바인딩 컨텍스트를 추출하는 함수
    
    Args:
        state: 현재 에이전트 상태
        llm: 언어 모델 (선택적, 실제 구현에서는 주입됨)
        
    Returns:
        업데이트된 상태
    """
    # LLM이 제공되지 않은 경우 (테스트용) 기본 응답 반환
    if llm is None:
        default_context = {
            "query_entities": {"main_concept": ["task_0.main_concept"]},
            "previous_features": [],
            "explicit_dependencies": []
        }

        updated_state = update_state(
            state,
            binding_context=default_context,
            next="planner"
        )
        return updated_state

    try:
        # 바인딩 컨텍스트 추출을 위한 메시지 구성
        messages = [
            SystemMessage(content=CONSTRUCTOR_SYSTEM_PROMPT),
            HumanMessage(content=f"User Query: {state.get('user_query', '')}")
        ]

        # LLM 호출
        response = await llm.ainvoke(messages)

        # 응답 파싱
        content = response.content or "{}"
        binding_context = json.loads(content)

        # 기본 구조 보장
        if "query_entities" not in binding_context:
            binding_context["query_entities"] = {"features": [], "keywords": []}
        if "previous_features" not in binding_context:
            binding_context["previous_features"] = []
        if "explicit_dependencies" not in binding_context:
            binding_context["explicit_dependencies"] = []

        # 상태 업데이트
        updated_state = update_state(
            state,
            binding_context=binding_context,
            next="planner"
        )

        return updated_state

    except Exception as e:
        # 에러 발생 시 planner로 이동
        print(f"VarConstructor error: {e}")
        return update_state(state, next="planner")


# 노드 인스턴스 생성
var_constructor_node = VarConstructorNode()
