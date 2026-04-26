# langgraph_agenticrag/src/agents/nodes/executor_node.py

import asyncio
import json
import logging
import re
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph_flow.agents.state import AgentState, update_state
from tools.registry import ToolRegistry
from langgraph_flow.prompts.executor import EXECUTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

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
                resolved_query, result = await self._execute_retrieve_subtask(
                    subtask, tool_registry, state, resolved_bindings, llm
                )
                retriever_outputs.append({
                    "subtask_id": subtask_id,
                    "query": resolved_query,  # resolved goal for correct dedup in merge_retriever_history
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

        # ID 기반 lookup (var_binder/routing_logic과 동일 의미). 위치 인덱스 사용 금지.
        subtasks_by_id = {s.get("id"): s for s in subtasks if s.get("id") is not None}

        for subtask in subtasks:
            # 이미 완료된 subtask는 건너뜀
            # verdict가 True이거나 "exceeded"이면 완료로 간주
            verdict = subtask.get("verdict", False)
            if verdict is True or verdict == "exceeded":
                continue

            # 의존관계 확인 (ID 매칭)
            dependencies = subtask.get("dependencies", [])
            if not dependencies:
                executable.append(subtask)
                continue

            all_dependencies_met = True
            for dep_id in dependencies:
                dep_subtask = subtasks_by_id.get(dep_id)
                if dep_subtask is None:
                    all_dependencies_met = False
                    break
                dep_verdict = dep_subtask.get("verdict", False)
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
        original_goal = goal

        # ===== DIAGNOSTIC LOGGING =====
        logger.info("=" * 80)
        logger.info("[Executor:RETRIEVE] ENTER subtask_id=%s", subtask.get("id"))
        logger.info("[Executor:RETRIEVE] subtask.goal         = %r", goal)
        logger.info("[Executor:RETRIEVE] subtask.task_type    = %r", subtask.get("task_type"))
        logger.info("[Executor:RETRIEVE] subtask.dependencies = %r", subtask.get("dependencies"))
        logger.info("[Executor:RETRIEVE] subtask.bindings     = %r", subtask.get("bindings"))
        logger.info("[Executor:RETRIEVE] subtask.verdict      = %r", subtask.get("verdict"))
        logger.info("[Executor:RETRIEVE] subtask.top_k        = %r", subtask.get("top_k"))
        logger.info("[Executor:RETRIEVE] resolved_bindings    = %r", resolved_bindings)
        sr_summary = [
            {"id": r.get("id"),
             "verdict": r.get("verdict"),
             "ref_features": r.get("reference_features", []),
             "subtask_answer_preview": (r.get("subtask_answer", "") or "")[:120]}
            for r in state.get("subtask_results", [])
        ]
        logger.info("[Executor:RETRIEVE] subtask_results[] = %s", sr_summary)
        # ================================

        # var_binder에서 해결한 resolved_bindings 적용
        if resolved_bindings:
            updated_goal = goal
            for key, value in resolved_bindings.items():
                # 1) key 자체가 placeholder literal일 때만 직접 대체 (e.g. "$subtask_0.feature_id").
                # bare 단어 키(`feature_id`, `feature_name` 등)가 다른 placeholder 내부 substring을
                # blanket replace 하지 않도록 `$` 접두사 가드. 그 외 키는 path 3 정규식이 처리.
                if key.startswith("$") and key in updated_goal:
                    logger.info("[Executor:RETRIEVE] key-match  '%s' → '%s'", key, value)
                    updated_goal = updated_goal.replace(key, str(value))
                # 2) bare ${key}
                bare = f"${{{key}}}"
                if bare in updated_goal:
                    logger.info("[Executor:RETRIEVE] placeholder-match '%s' → '%s'", bare, value)
                    updated_goal = updated_goal.replace(bare, str(value))
                # 3) $subtask_N.{key} or ${subtask_N.{key}} for any N
                pattern = re.compile(rf'\$\{{?subtask_\d+\.{re.escape(key)}\}}?')
                def _sub(m, v=value):
                    logger.info("[Executor:RETRIEVE] placeholder-match '%s' → '%s'", m.group(0), v)
                    return str(v)
                updated_goal = pattern.sub(_sub, updated_goal)

            if updated_goal == goal:
                logger.warning("[Executor:RETRIEVE] NO substitution applied "
                               "(resolved_bindings non-empty but no placeholder in goal)")
                # safety net: enrich query with concrete binding values so retriever
                # embeds feature_id / feature_name / literals even if goal lacks placeholders
                merged = {**subtask.get("bindings", {}), **resolved_bindings}
                concrete = {
                    k: v for k, v in merged.items()
                    if isinstance(v, str) and not v.startswith("$") and not v.startswith("unresolved_")
                }
                if concrete:
                    extras = " ".join(str(v) for v in concrete.values())
                    updated_goal = f"{goal} [{extras}]"
                    logger.info("[Executor:RETRIEVE] enriched query with bindings: %s", concrete)
                else:
                    logger.warning("[Executor:RETRIEVE] no concrete bindings available to enrich query")
            goal = updated_goal

        # fallback: auto-resolve remaining $subtask_N.field from state subtask_results
        # verdict=True인 entry 중 attempt 최대값만 사용 (실패 attempt 제외)
        if "$subtask_" in goal:
            subtask_results = state.get("subtask_results", [])
            results_by_id: Dict[str, Dict] = {}
            for r in subtask_results:
                if r.get("verdict") is not True:
                    continue
                sid = str(r.get("id", ""))
                if not sid:
                    continue
                existing = results_by_id.get(sid)
                if existing is None or r.get("attempt", 0) > existing.get("attempt", 0):
                    results_by_id[sid] = r
            for match in re.finditer(r'\$subtask_(\d+)\.(\w+)', goal):
                placeholder, subtask_id, field = match.group(0), match.group(1), match.group(2)
                result = results_by_id.get(subtask_id)
                if not result:
                    logger.warning("[Executor:RETRIEVE] auto-resolve miss: %s (no result for subtask_id=%s)",
                                   placeholder, subtask_id)
                    continue
                payload = result.get("result") or {}
                ref_features = payload.get("reference_features") or result.get("reference_features", [])
                # 모든 reference_features 엔트리에서 field 값 수집 (순서 보존 dedupe).
                # 단수면 단일 문자열, 복수면 공백 join — var_binder fallback과 동일 정책.
                collected: List[str] = []
                seen_vals: set = set()
                for feat in ref_features:
                    if field in feat:
                        raw = feat[field]
                        if raw is None:
                            continue
                        v = str(raw).strip()
                        if not v or v in seen_vals:
                            continue
                        seen_vals.add(v)
                        collected.append(v)
                if collected:
                    value = collected[0] if len(collected) == 1 else " ".join(collected)
                else:
                    value = (
                        payload.get(field)
                        or result.get(field)
                        or payload.get("subtask_answer")
                        or result.get("subtask_answer", "")
                    )
                if value:
                    goal = goal.replace(placeholder, str(value))
                    logger.info("[Executor:RETRIEVE] auto-resolved %s → %s", placeholder, value)
                else:
                    logger.warning("[Executor:RETRIEVE] auto-resolve failed: %s (field=%s missing in result)",
                                   placeholder, field)
            if "$subtask_" in goal:
                logger.warning("[Executor:RETRIEVE] subtask_id=%s unresolved placeholders remain: %s",
                               subtask.get("id"), goal)

        logger.info("[Executor:RETRIEVE] FINAL query sent to retriever = %r", goal)
        logger.info("=" * 80)

        # retriever 툴 실행
        tool = tool_registry.get_tool("retriever")
        if tool is None:
            raise ValueError("Retriever tool not found in registry")

        tool_args = {"query": goal, "top_k": subtask.get("top_k", 5)}

        logger.info(
            "[Retriever] subtask_id=%s original_goal=%s resolved_query=%s",
            subtask.get("id"), original_goal, goal
        )

        # 툴 실행
        return goal, await tool.ainvoke(tool_args)

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
                # bare 단어 키가 다른 placeholder 내부 substring을 침범하지 않도록 `$` 접두사 가드.
                if key.startswith("$") and key in updated_goal:
                    updated_goal = updated_goal.replace(key, str(value))
                # bare ${key}
                bare = f"${{{key}}}"
                if bare in updated_goal:
                    updated_goal = updated_goal.replace(bare, str(value))
                # $subtask_N.{key} or ${subtask_N.{key}} for any N
                pattern = re.compile(rf'\$\{{?subtask_\d+\.{re.escape(key)}\}}?')
                updated_goal = pattern.sub(str(value), updated_goal)
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
