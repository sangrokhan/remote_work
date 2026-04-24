# langgraph_agenticrag/src/agents/nodes/executor_node.py

import asyncio
import json
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph_flow.agents.state import AgentState, update_state
from tools.registry import ToolRegistry
from langgraph_flow.prompts.executor import EXECUTOR_SYSTEM_PROMPT

# 디버그를 위한 전역 카운터
executor_call_count = 0


class ExecutorNode:
    """Executor 노드 클래스"""

    def __init__(self):
        self.name = "executor"
        self.max_retries = 3
        self.system_prompt = EXECUTOR_SYSTEM_PROMPT

    async def invoke(self, state: AgentState,
                     config: Optional[RunnableConfig] = None) -> AgentState:
        """
        Executor 노드 실행
        
        Args:
            state: 현재 에이전트 상태
            config: 설정 정보 (선택적)
            
        Returns:
            업데이트된 상태
        """
        global executor_call_count
        executor_call_count += 1

        print(f"=== DEBUG: Executor invoke 호출 #{executor_call_count} ===")
        print(f"Current Step: {state.get('current_step', 0)}")
        print(f"Subtasks Count: {len(state.get('subtasks', []))}")
        print(f"Execution History Keys: {list(state.get('execution_history', {}).keys())}")
        print(f"Subtask Results Count: {len(state.get('subtask_results', []))}")

        tool_registry = ToolRegistry() if config is None else config.get("tool_registry",
                                                                         ToolRegistry())
        llm = None
        if config is not None:
            # dict 형식의 config 처리
            if isinstance(config, dict):
                llm = config.get("llm")
                if llm is None and "configurable" in config:
                    llm = config["configurable"].get("llm")
            # RunnableConfig 객체 처리
            elif hasattr(config, 'configurable') and config.configurable:
                llm = config.configurable.get("llm")
            # 다른 객체 형식 처리
            elif hasattr(config, 'get'):
                llm = config.get("llm")

        return await self.execute_subtasks(state, tool_registry, llm)

    async def execute_subtasks(self, state: AgentState, tool_registry: ToolRegistry,
                               llm: BaseLanguageModel = None) -> AgentState:
        """
        Subtask들을 실행하는 함수
        
        Args:
            state: 현재 에이전트 상태
            tool_registry: 툴 레지스트리
            llm: 언어 모델 (선택적)
            
        Returns:
            업데이트된 상태
        """
        subtasks = state.get("subtasks", [])
        execution_history = state.get("execution_history", {}).copy()
        retry_counts = state.get("retry_counts", {}).copy()
        retriever_outputs = []
        retriever_history = list(state.get("retriever_history", []))
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", 10)
        is_finished = state.get("is_finished", False)

        # var_binder에서 해결한 resolved_bindings 가져오기
        resolved_bindings = state.get("resolved_bindings", {})
        current_executing_subtask_id = state.get("current_executing_subtask_id")

        print(f"=== DEBUG: execute_subtasks ===")
        print(f"Total Subtasks: {len(subtasks)}")
        print(f"Current Executing Subtask ID: {current_executing_subtask_id}")
        print(f"Resolved Bindings: {resolved_bindings}")

        # 종료 조건 체크
        if is_finished or current_step >= max_steps:
            return update_state(state, is_finished=True, next="synthesizer")

        if not subtasks:
            return update_state(state, next="refiner")

        # var_binder에서 선택한 subtask 사용
        if current_executing_subtask_id is not None:
            # 선택된 subtask 찾기
            subtask = None
            for s in subtasks:
                if s.get("id") == current_executing_subtask_id:
                    subtask = s
                    break

            if subtask is None:
                print("=== DEBUG: 선택된 subtask를 찾을 수 없음 - synthesizer로 이동 ===")
                return update_state(state, next="synthesizer")
        else:
            # fallback: 실행 가능한 subtask 찾기
            executable_subtasks = self._get_executable_subtasks(subtasks, state)
            if not executable_subtasks:
                print("=== DEBUG: 실행 가능한 subtask 없음 - synthesizer로 이동 ===")
                return update_state(state, next="synthesizer")
            subtask = executable_subtasks[0]

        print(f"=== DEBUG: 실행할 Subtask ID: {subtask['id']} ===")

        subtask_id = subtask["id"]

        try:
            # task_type 기반 툴 라우팅
            task_type = subtask.get("task_type", "THINK")

            if task_type == "RETRIEVE":
                # RETRIEVE 타입은 retriever_outputs에 저장
                result = await self._execute_retrieve_subtask(subtask, tool_registry, state,
                                                              resolved_bindings, llm)
                retriever_outputs.append({
                    "subtask_id": subtask_id,
                    "query": subtask.get("goal", subtask.get("description", "")),
                    "result": result,
                    "status": "success"
                })
            else:
                # THINK 타입은 execution_history에 저장
                result = await self._execute_think_subtask(subtask, tool_registry, state,
                                                           resolved_bindings, llm)
                retriever_history.append({
                    "subtask_id": subtask_id,
                    "query": subtask.get("goal", subtask.get("description", "")),
                    "result": {"results": [str(result)]} if result else {"results": []},
                    "status": "success",
                })

            # 실행 결과 저장
            if subtask_id not in execution_history:
                execution_history[subtask_id] = []

            execution_history[subtask_id].append({
                "result": result,
                "status": "success",
                "retry_count": retry_counts.get(subtask_id, 0)
            })

        except Exception as e:
            # 재시도 로직
            current_retries = retry_counts.get(subtask_id, 0)
            if current_retries < 3:
                retry_counts[subtask_id] = current_retries + 1
                print(f"=== DEBUG: Subtask {subtask_id} 재시도 {current_retries + 1}/3 ===")
            else:
                # 최대 재시도 횟수 초과
                if subtask_id not in execution_history:
                    execution_history[subtask_id] = []

                execution_history[subtask_id].append({
                    "result": str(e),
                    "status": "failed",
                    "retry_count": current_retries,
                    "error": True
                })
                print(f"=== DEBUG: Subtask {subtask_id} 최대 재시도 초과 ===")

        # 상태 업데이트 - 현재 실행 중인 subtask ID 저장
        updated_state = update_state(
            state,
            execution_history=execution_history,
            retry_counts=retry_counts,
            retriever_outputs=retriever_outputs,
            retriever_history=retriever_history,
            current_executing_subtask_id=subtask_id,
            next="retriever"
        )

        return updated_state

    def _get_executable_subtasks(self, subtasks: List[Dict], state: AgentState) -> List[Dict]:
        """
        실행 가능한 subtask들 필터링
        subtasks[i].verdict를 기준으로 완료 여부를 판단합니다.
        
        Args:
            subtasks: 전체 subtask 리스트
            state: 현재 상태 (verdict 확인용)
            
        Returns:
            실행 가능한 subtask 리스트
        """
        executable = []

        for i, subtask in enumerate(subtasks):
            task_id = subtask.get("id", i)

            # 이미 완료된 subtask는 건너뜀
            # verdict가 True이거나 "exceeded"이면 완료로 간주
            verdict = subtask.get("verdict", False)
            if verdict is True or verdict == "exceeded":
                continue

            # 의존관계 확인
            dependencies = subtask.get("dependencies", [])
            if not dependencies:
                # 의존이 없는 subtask는 즉시 실행 가능
                executable.append(subtask)
                continue

            # 모든 의존 subtask가 완료되었는지 확인
            all_dependencies_met = True
            for dep_index in dependencies:
                if dep_index >= len(subtasks):
                    all_dependencies_met = False
                    break
                dep_verdict = subtasks[dep_index].get("verdict", False)
                if not (dep_verdict is True or dep_verdict == "exceeded"):
                    all_dependencies_met = False
                    break

            if all_dependencies_met:
                executable.append(subtask)

        return executable

    async def _execute_retrieve_subtask(self, subtask: Dict, tool_registry: ToolRegistry,
                                        state: AgentState, resolved_bindings: dict,
                                        llm: BaseLanguageModel = None) -> Any:
        """
        RETRIEVE 타입 subtask 실행
        
        Args:
            subtask: 실행할 subtask
            tool_registry: 툴 레지스트리
            state: 현재 상태
            resolved_bindings: var_binder에서 해결된 바인딩
            llm: 언어 모델 (선택적)
            
        Returns:
            실행 결과
        """
        goal = subtask.get("goal", subtask.get("description", ""))

        # var_binder에서 해결한 resolved_bindings 적용
        if resolved_bindings:
            updated_goal = goal
            for key, value in resolved_bindings.items():
                # 다양한 placeholder 형식 지원
                placeholders = [
                    f"${{{key}}}",  # ${feature_id}
                    f"$task_0.{key}",  # $task_0.feature_id (직접 참조)
                    f"${{task_0.{key}}}"  # ${task_0.feature_id}
                ]
                for placeholder in placeholders:
                    if placeholder in updated_goal:
                        updated_goal = updated_goal.replace(placeholder, str(value))
                        print(f"=== DEBUG: Placeholder '{placeholder}' → '{value}'로 대체 ===")

            # unresolved가 남아있으면 원래 placeholder 유지
            if "unresolved_" in updated_goal:
                print(f"=== DEBUG: WARNING - unresolved binding이 남아있음 ===")
                print(f"=== DEBUG: updated_goal: {updated_goal} ===")

            goal = updated_goal

        # retriever 툴 실행
        tool = tool_registry.get_tool("retriever")
        if tool is None:
            raise ValueError("Retriever tool not found in registry")

        # 툴 실행 파라미터 구성
        tool_args = {"query": goal}

        # top_k 파라미터가 있는 경우 추가
        if "top_k" in subtask:
            tool_args["top_k"] = subtask["top_k"]
        else:
            tool_args["top_k"] = 5  # 기본값

        print(f"=== DEBUG: Retriever 호출 - query: {goal[:99]}... ===")

        # 툴 실행
        return await tool.ainvoke(tool_args)

    async def _execute_think_subtask(self, subtask: Dict, tool_registry: ToolRegistry,
                                     state: AgentState, resolved_bindings: dict,
                                     llm: BaseLanguageModel = None) -> Any:
        """
        THINK 타입 subtask 실행
        
        Args:
            subtask: 실행할 subtask
            tool_registry: 툴 레지스트리
            state: 현재 상태
            resolved_bindings: var_binder에서 해결된 바인딩
            llm: 언어 모델 (선택적)
            
        Returns:
            실행 결과
        """
        goal = subtask.get("goal", subtask.get("description", ""))

        # var_binder에서 해결한 resolved_bindings 적용
        if resolved_bindings:
            updated_goal = goal
            for key, value in resolved_bindings.items():
                placeholder = f"${{{key}}}"
                if placeholder in updated_goal:
                    updated_goal = updated_goal.replace(placeholder, str(value))
            goal = updated_goal

        # description 기반으로 적절한 툴 선택
        tool_name = self._select_tool_for_subtask(goal)
        tool = tool_registry.get_tool(tool_name)

        if tool is None:
            # THINK 타입의 경우 기본적으로 LLM 기반 처리
            if llm:
                # LLM을 사용하여 직접 처리
                messages = [
                    SystemMessage(
                        content="당신은 THINK 타입의 작업을 처리하는 어시스턴트입니다. 주어진 작업을 분석하고 적절한 응답을 생성하세요."),
                    HumanMessage(content=f"작업: {goal}")
                ]
                response = await llm.bind(temperature=0.7).ainvoke(messages)
                content = response.content or "{}"
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```", 2)[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.rsplit("```", 1)[0].strip()
                if not content:
                    content = "{}"
                return content
            else:
                raise ValueError(f"Tool '{tool_name}' not found in registry and no LLM available")

        # 툴 실행 파라미터 구성
        tool_args = {"query": goal}

        # 툴 실행
        return await tool.ainvoke(tool_args)

    def _select_tool_for_subtask(self, goal: str) -> str:
        """
        subtask 설명에 따라 적절한 툴 선택
        
        Args:
            goal: subtask 목표
            
        Returns:
            툴 이름
        """
        # 기존 executor.py는 LLM을 통해 직접 decision을 생성하므로
        # 툴 선택이 아닌 LLM 기반 처리를 위해 retriever를 기본값으로 사용
        return "retriever"


# 노드 인스턴스 생성
executor_node = ExecutorNode()
