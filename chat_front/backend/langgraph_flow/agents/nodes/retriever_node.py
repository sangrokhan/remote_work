# langgraph_agenticrag/src/agents/nodes/retriever_node.py

import asyncio
import json
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langgraph_flow.agents.state import AgentState, update_state
from tools.registry import ToolRegistry


class RetrieverNode:
    """
    Retriever 노드 클래스
    
    - Executor에서 이미 retriever tool을 실행하고 retriever_outputs에 저장
    - 이 노드는 단순히 상태를 전달하고 retriever_history에 로깅만 수행
    - 실제 검색은 executor_node의 _execute_retrieve_subtask에서 수행됨
    """

    def __init__(self):
        self.name = "retriever"
        self.max_retries = 3

    async def invoke(self, state: AgentState,
                     config: Optional[RunnableConfig] = None) -> AgentState:
        """
        Retriever 노드 실행
        
        Args:
            state: 현재 에이전트 상태
            config: 설정 정보 (선택적)
            
        Returns:
            업데이트된 상태
        """
        return await self.pass_through(state)

    async def pass_through(self, state: AgentState) -> AgentState:
        """
        상태 전달 및 로깅
        
        executor에서 이미 수행된 검색 결과를 retriever_history에 로깅
        중복 로깅 방지를 위해 이미 로깅된 (subtask_id, query) 조합을 확인
        
        원본 Agentic RAG 방식:
        - state.retriever_history.append(tool_msg) - 한 번만 append
        - LangGraph reducer 방식: (old or []) + (new or [])
        
        Args:
            state: 현재 에이전트 상태
            
        Returns:
            업데이트된 상태
        """
        retriever_outputs = state.get("retriever_outputs", [])
        retriever_history = state.get("retriever_history", [])

        # 이미 로깅된 (subtask_id, query) 조합 집합 - 더 정확한 중복 체크
        logged_entries = set()
        for history in retriever_history:
            subtask_id = history.get("subtask_id")
            query = history.get("query", "")
            # (subtask_id, query) 조합으로 중복 체크
            logged_entries.add((subtask_id, query[:100] if query else ""))  # query 앞부분만 사용

        new_entries = []
        for output in retriever_outputs:
            subtask_id = output.get("subtask_id")
            query = output.get("query", "")
            entry_key = (subtask_id, query[:100] if query else "")

            if entry_key not in logged_entries:
                new_entries.append({
                    "subtask_id": subtask_id,
                    "query": query,
                    "result": output.get("result"),
                    "status": output.get("status", "unknown")
                })
                logged_entries.add(entry_key)  # 집합에 추가하여 중복 방지
                print(f"=== DEBUG: Retriever Node - 새로운 항목 추가: subtask_id={subtask_id} ===")
            else:
                print(f"=== DEBUG: Retriever Node - 중복 항목 건너뜀: subtask_id={subtask_id} ===")

        print(f"=== DEBUG: Retriever Node - {len(new_entries)}개의 새로운 검색 결과 로깅 ===")

        # 상태 업데이트 - refiner로 이동
        # retriever_outputs를 비워서 다음 iteration에서 깨끗하게 시작
        # 원본 코드: state.retriever_outputs = []; state.retriever_outputs.append(tool_msg)
        updated_state = update_state(
            state,
            retriever_history=new_entries,  # 새로운 항목만 반환 (reducer가 누적)
            retriever_outputs=[],  # ⭐ 비워서 다음 iteration에서 깨끗하게 시작
            next="refiner"
        )

        return updated_state


# 노드 인스턴스 생성
retriever_node = RetrieverNode()
