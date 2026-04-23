"""
Var binder node — 실행 가능한 subtask 선택 및 바인딩 해결 전담.
subtask_results에서 이전 결과를 참조해 $task_N.field 패턴을 구체적 값으로 변환.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.prompts.var_binder import BINDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class VarBinderNode:
    """Variable Binder 노드 클래스 - 바인딩 해결 전담"""

    def __init__(self):
        self.name = "var_binder"

    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        # 노드에 LLM 주입 없음 — stateless 구조
        return await resolve_bindings(state)


async def resolve_bindings(
    state: AgentState,
    llm: Optional[BaseLanguageModel] = None,
) -> AgentState:
    """
    바인딩을 해결하는 함수

    subtasks에서 실행 가능한 subtask를 찾아 bindings 해결 후 executor로 이동.
    LLM이 없으면 fallback 직접 해결 사용.
    """
    subtasks = state.get("subtasks", [])
    subtask_results = state.get("subtask_results", [])
    execution_history = state.get("execution_history", {})

    # 실행 가능한 subtask 찾기 (verdict가 False인 것 중 의존성이 해결된 것)
    executable_subtask = _get_next_executable_subtask(subtasks, subtask_results, execution_history)

    if not executable_subtask:
        # 실행할 subtask가 없으면 synthesizer로
        return update_state(state, next="synthesizer")

    # 바인딩 정보 추출
    bindings = executable_subtask.get("bindings", {})

    # 바인딩이 없으면 바로 executor로
    if not bindings:
        return update_state(
            state,
            current_executing_subtask_id=executable_subtask["id"],
            resolved_bindings={},
            next="executor",
        )

    # subtask_results가 비어있으면 폴백 사용
    if not subtask_results:
        resolved_bindings = _resolve_bindings_fallback(bindings, [])
        return update_state(
            state,
            current_executing_subtask_id=executable_subtask["id"],
            resolved_bindings=resolved_bindings,
            next="executor",
        )

    # LLM이 제공된 경우 LLM 기반 binding 해결
    if llm:
        try:
            resolved_bindings = await _resolve_bindings_with_llm(
                bindings,
                subtask_results,
                {
                    "current_subtask": executable_subtask,
                    "user_query": state.get("user_query", ""),
                    "dependencies": executable_subtask.get("dependencies", []),
                },
                llm,
            )
            return update_state(
                state,
                current_executing_subtask_id=executable_subtask["id"],
                resolved_bindings=resolved_bindings,
                next="executor",
            )
        except Exception as e:
            # 에러 시 폴백 사용
            logger.error("VarBinder LLM error: %s", e)

    # 폴백: 직접 binding 해결
    resolved_bindings = _resolve_bindings_fallback(bindings, subtask_results)

    return update_state(
        state,
        current_executing_subtask_id=executable_subtask["id"],
        resolved_bindings=resolved_bindings,
        next="executor",
    )


def _get_next_executable_subtask(
    subtasks: List[Dict],
    subtask_results: List[Dict],
    execution_history: Dict,
) -> Optional[Dict]:
    """
    실행 가능한 다음 subtask를 찾는 함수

    완료된 subtask(verdict=True 또는 subtask_results에 있는 것) 제외,
    의존성이 모두 완료된 첫 번째 subtask 반환.
    """
    completed_ids = set()
    for result in subtask_results:
        task_id = result.get("id", result.get("task_id"))
        if task_id is not None:
            completed_ids.add(task_id)

    for subtask in subtasks:
        task_id = subtask.get("id")

        # 이미 완료된 subtask는 건너뜀
        if task_id in completed_ids:
            continue

        # verdict가 True이면 완료된 것으로 간주
        if subtask.get("verdict", False) is True:
            continue

        # 의존성 확인
        dependencies = subtask.get("dependencies", [])
        if not dependencies:
            return subtask

        # 모든 의존성이 완료되었는지 확인
        if all(dep_id in completed_ids for dep_id in dependencies):
            return subtask

    return None


async def _resolve_bindings_with_llm(
    bindings: dict,
    subtask_results: list,
    subtask_context: dict,
    llm: BaseLanguageModel,
) -> dict:
    """
    LLM으로 추상 바인딩을 구체적 값으로 해결

    $task_0.feature_id → subtask_results[0]["reference_features"][0]["feature_id"]
    """
    logger.debug("_resolve_bindings_with_llm: bindings=%s results_count=%d", bindings, len(subtask_results))

    if not bindings:
        return {}

    # subtask_results를 dict 형태로 변환 (task_id → result)
    subtask_results_dict = {}
    for result in subtask_results:
        task_id = result.get("id", result.get("task_id"))
        if task_id is not None:
            subtask_results_dict[str(task_id)] = result

    messages = [
        SystemMessage(content=BINDER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Bindings to resolve: {json.dumps(bindings, ensure_ascii=False)}\n\n"
                f"Previous results: {json.dumps(subtask_results_dict, ensure_ascii=False, indent=2)}\n\n"
                f"Current context: {json.dumps(subtask_context, ensure_ascii=False)}"
            )
        ),
    ]

    response = await llm.ainvoke(messages)
    # ainvoke는 str 반환, 혹은 AIMessage — 둘 다 처리
    content = response if isinstance(response, str) else getattr(response, "content", "{}")
    resolved_bindings = json.loads(content)
    logger.debug("_resolve_bindings_with_llm: resolved=%s", resolved_bindings)
    return resolved_bindings


def _resolve_bindings_fallback(bindings: dict, subtask_results: list) -> dict:
    """
    폴백: subtask_results에서 직접 바인딩 해결

    $task_0.feature_id → subtask_results[0]["reference_features"][0]["feature_id"]
    $task_0.feature_name → subtask_results[0]["reference_features"][0]["feature_name"]
    """
    if not bindings:
        return {}

    resolved = {}
    for binding_key, binding_ref in bindings.items():
        logger.debug("binding '%s': '%s' 해결 중", binding_key, binding_ref)

        if isinstance(binding_ref, str) and binding_ref.startswith("$task_"):
            # $task_{id}.{field} 파싱
            parts = binding_ref.replace("$task_", "").split(".")
            if len(parts) == 2:
                task_id, field_name = parts
                task_id_int = int(task_id)
                found_value = None

                for result in subtask_results:
                    result_id = result.get("id", result.get("task_id"))
                    if result_id != task_id_int:
                        continue

                    # reference_features에서 먼저 찾기
                    for ref in result.get("reference_features", []):
                        if field_name in ref:
                            found_value = ref[field_name]
                            logger.debug("reference_features에서 찾음: %s", found_value)
                            break

                    # 없으면 텍스트에서 패턴 추출
                    if found_value is None:
                        text = result.get("subtask_answer", "") + result.get("refined_text", "")
                        if field_name == "feature_id":
                            m = re.search(r'FGR-[A-Z]{2}\d{4}', text)
                            if m:
                                found_value = m.group(0)
                                logger.debug("텍스트에서 feature_id 추출: %s", found_value)
                        elif field_name == "feature_name":
                            for line in text.split('\n'):
                                if 'feature' in line.lower() and 'name' in line.lower():
                                    found_value = line.split(':')[-1].strip() if ':' in line else line.strip()
                                    break

                    if found_value:
                        break

                if found_value:
                    resolved[binding_key] = found_value
                else:
                    # 최종 폴백: unresolved 표시
                    resolved[binding_key] = f"unresolved_{task_id}_{field_name}"
                    logger.debug("binding 해결 실패: unresolved_%s_%s", task_id, field_name)
        else:
            resolved[binding_key] = binding_ref

    logger.debug("fallback 해결 결과: %s", resolved)
    return resolved


var_binder_node = VarBinderNode()
