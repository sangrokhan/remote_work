"""
Executor node — executes subtasks from planner output.
Routes RETRIEVE subtasks to retriever tool, THINK subtasks to LLM.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.prompts.executor import EXECUTOR_SYSTEM_PROMPT
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ExecutorNode:
    def __init__(self) -> None:
        self.name = "executor"
        self.max_retries = 3

    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        configurable = (config or {}).get("configurable", {})
        llm: Optional[BaseLanguageModel] = configurable.get("llm")
        tool_registry: ToolRegistry = configurable.get("tool_registry") or ToolRegistry()
        return await self._execute_subtasks(state, tool_registry, llm)

    async def _execute_subtasks(
        self,
        state: AgentState,
        tool_registry: ToolRegistry,
        llm: Optional[BaseLanguageModel],
    ) -> AgentState:
        subtasks = state.get("subtasks", [])
        execution_history = state.get("execution_history", {}).copy()
        retry_counts = state.get("retry_counts", {}).copy()
        retriever_outputs = []
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", 10)
        is_finished = state.get("is_finished", False)

        # var_binder에서 해결한 resolved_bindings 가져오기
        resolved_bindings = state.get("resolved_bindings", {})
        current_executing_subtask_id = state.get("current_executing_subtask_id")

        # 종료 조건 체크
        if is_finished or current_step >= max_steps:
            return update_state(state, is_finished=True, next="synthesizer")

        if not subtasks:
            return update_state(state, next="refiner")

        # var_binder에서 선택한 subtask 사용
        if current_executing_subtask_id is not None:
            # 선택된 subtask 찾기
            subtask = next((s for s in subtasks if s.get("id") == current_executing_subtask_id), None)
            if subtask is None:
                logger.warning("executor: selected subtask id=%s not found", current_executing_subtask_id)
                return update_state(state, next="synthesizer")
        else:
            # fallback: 실행 가능한 subtask 찾기
            executable = self._get_executable_subtasks(subtasks, state)
            if not executable:
                logger.debug("executor: no executable subtasks")
                return update_state(state, next="synthesizer")
            subtask = executable[0]

        subtask_id = subtask["id"]
        logger.debug("executor: running subtask id=%s", subtask_id)

        try:
            # task_type 기반 툴 라우팅
            task_type = subtask.get("task_type", "THINK")
            if task_type == "RETRIEVE":
                # RETRIEVE 타입은 retriever_outputs에 저장
                result = await self._execute_retrieve_subtask(subtask, tool_registry, resolved_bindings)
                retriever_outputs.append({
                    "subtask_id": subtask_id,
                    "query": subtask.get("goal", subtask.get("description", "")),
                    "result": result,
                    "status": "success",
                })
            else:
                result = await self._execute_think_subtask(subtask, tool_registry, resolved_bindings, llm)
                retriever_outputs.append({
                    "subtask_id": subtask_id,
                    "query": subtask.get("goal", subtask.get("description", "")),
                    "result": result,
                    "status": "success",
                })

            # 실행 결과 저장
            execution_history.setdefault(subtask_id, []).append({
                "result": result,
                "status": "success",
                "retry_count": retry_counts.get(subtask_id, 0),
            })

        except Exception as exc:
            # 재시도 로직
            current_retries = retry_counts.get(subtask_id, 0)
            if current_retries < self.max_retries:
                retry_counts[subtask_id] = current_retries + 1
                logger.warning("executor: subtask %s retry %d/%d", subtask_id, current_retries + 1, self.max_retries)
            else:
                # 최대 재시도 횟수 초과
                execution_history.setdefault(subtask_id, []).append({
                    "result": str(exc),
                    "status": "failed",
                    "retry_count": current_retries,
                    "error": True,
                })
                logger.error("executor: subtask %s exceeded max retries", subtask_id)

        # 상태 업데이트 - 현재 실행 중인 subtask ID 저장
        return update_state(
            state,
            execution_history=execution_history,
            retry_counts=retry_counts,
            retriever_outputs=retriever_outputs,
            current_executing_subtask_id=subtask_id,
            next="retriever",
        )

    def _get_executable_subtasks(self, subtasks: List[Dict], state: AgentState) -> List[Dict]:
        """실행 가능한 subtask들 필터링 — verdict 기준으로 완료 여부 판단"""
        executable = []
        for subtask in subtasks:
            # 이미 완료된 subtask는 건너뜀
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
            if all(
                subtasks[dep].get("verdict") in (True, "exceeded")
                for dep in dependencies
                if dep < len(subtasks)
            ):
                executable.append(subtask)
        return executable

    async def _execute_retrieve_subtask(
        self,
        subtask: Dict,
        tool_registry: ToolRegistry,
        resolved_bindings: Dict[str, Any],
    ) -> Any:
        """RETRIEVE 타입 subtask 실행"""
        goal = self._resolve_bindings(subtask.get("goal", subtask.get("description", "")), resolved_bindings)

        # retriever 툴 실행
        tool = tool_registry.get("retriever")
        if tool is None:
            raise ValueError("retriever tool not registered")

        # 툴 실행 파라미터 구성
        tool_args = {"query": goal, "top_k": subtask.get("top_k", 5)}
        return await tool.ainvoke(tool_args)

    async def _execute_think_subtask(
        self,
        subtask: Dict,
        tool_registry: ToolRegistry,
        resolved_bindings: Dict[str, Any],
        llm: Optional[BaseLanguageModel],
    ) -> Any:
        """THINK 타입 subtask 실행"""
        goal = self._resolve_bindings(subtask.get("goal", subtask.get("description", "")), resolved_bindings)

        # description 기반으로 적절한 툴 선택
        tool = tool_registry.get(self._select_tool(goal))

        if tool is None:
            # THINK 타입의 경우 기본적으로 LLM 기반 처리
            if llm is None:
                raise ValueError("no tool and no LLM available for THINK subtask")
            messages = [SystemMessage(content=EXECUTOR_SYSTEM_PROMPT), HumanMessage(content=f"작업: {goal}")]
            return await llm.ainvoke(messages)

        # 툴 실행
        return await tool.ainvoke({"query": goal})

    @staticmethod
    def _resolve_bindings(text: str, resolved_bindings: Dict[str, Any]) -> str:
        """var_binder에서 해결한 resolved_bindings 적용"""
        for key, value in resolved_bindings.items():
            # 다양한 placeholder 형식 지원
            for placeholder in [f"${{{key}}}", f"$task_0.{key}", f"${{task_0.{key}}}"]:
                text = text.replace(placeholder, str(value))
        return text

    @staticmethod
    def _select_tool(goal: str) -> str:
        """subtask 설명에 따라 적절한 툴 선택"""
        return "retriever"


executor_node = ExecutorNode()
