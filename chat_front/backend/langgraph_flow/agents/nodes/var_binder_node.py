# langgraph_agenticrag/src/agents/nodes/var_binder_node.py

import json
import re
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.prompts.var_binder import BINDER_SYSTEM_PROMPT


class VarBinderNode:
    """Variable Binder 노드 클래스 - 바인딩 해결 전담"""

    def __init__(self):
        self.name = "var_binder"

    async def invoke(self, state: AgentState,
                     config: Optional[RunnableConfig] = None) -> AgentState:
        """
        Variable Binder 노드 실행
        
        Args:
            state: 현재 에이전트 상태
            config: 설정 정보 (선택적)
            
        Returns:
            업데이트된 상태
        """
        # config에서 LLM 가져오기
        llm = None
        if config and "llm" in config:
            llm = config["llm"]

        return await resolve_bindings(state, llm)


async def resolve_bindings(state: AgentState,
                           llm: Optional[BaseLanguageModel] = None) -> AgentState:
    """
    바인딩을 해결하는 함수
    
    Args:
        state: 현재 에이전트 상태
        llm: 언어 모델 (선택적, 실제 구현에서는 주입됨)
        
    Returns:
        업데이트된 상태
    """
    # subtasks에서 실행 가능한 subtask 찾기
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
            next="executor"
        )

    # subtask_results가 비어있으면 폴백 사용
    if not subtask_results:
        resolved_bindings = _resolve_bindings_fallback(bindings, [])
        return update_state(
            state,
            current_executing_subtask_id=executable_subtask["id"],
            resolved_bindings=resolved_bindings,
            next="executor"
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
                    "dependencies": executable_subtask.get("dependencies", [])
                },
                llm
            )

            return update_state(
                state,
                current_executing_subtask_id=executable_subtask["id"],
                resolved_bindings=resolved_bindings,
                next="executor"
            )
        except Exception as e:
            print(f"VarBinder LLM error: {e}")
            # 에러 시 폴백 사용

    # 폴백: 직접 binding 해결
    resolved_bindings = _resolve_bindings_fallback(bindings, subtask_results)

    return update_state(
        state,
        current_executing_subtask_id=executable_subtask["id"],
        resolved_bindings=resolved_bindings,
        next="executor"
    )


def _get_next_executable_subtask(subtasks: List[Dict], subtask_results: List[Dict],
                                 execution_history: Dict) -> Optional[Dict]:
    """
    실행 가능한 다음 subtask를 찾는 함수
    
    Args:
        subtasks: 전체 subtask 리스트
        subtask_results: 완료된 subtask 결과들
        execution_history: 실행 이력
        
    Returns:
        실행 가능한 subtask 또는 None
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
        all_deps_completed = all(dep_id in completed_ids for dep_id in dependencies)
        if all_deps_completed:
            return subtask

    return None


async def _resolve_bindings_with_llm(bindings: dict, subtask_results: list,
                                     subtask_context: dict, llm: BaseLanguageModel) -> dict:
    """
    Resolve abstract bindings to concrete values using LLM
    
    - subtask_results에서 이전 subtask의 reference_features를 찾음
    - $task_0.feature_id → subtask_results[0]["reference_features"][0]["feature_id"]
    """
    print(f"=== DEBUG: _resolve_bindings_with_llm 호출 ===")
    print(f"Bindings: {bindings}")
    print(f"Subtask Results Count: {len(subtask_results)}")

    if not bindings:
        return {}

    # subtask_results를 dict 형태로 변환 (task_id → result)
    subtask_results_dict = {}
    for result in subtask_results:
        task_id = result.get("id", result.get("task_id"))
        if task_id is not None:
            subtask_results_dict[str(task_id)] = result

    resolution_messages = [
        SystemMessage(content=BINDER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Bindings to resolve: {json.dumps(bindings, ensure_ascii=False)}\n\nPrevious results: {json.dumps(subtask_results_dict, ensure_ascii=False, indent=2)}\n\nCurrent context: {json.dumps(subtask_context, ensure_ascii=False)}")
    ]

    print("=== DEBUG: Binding resolution LLM 호출 시작 ===")
    response = await llm.ainvoke(resolution_messages)
    print("=== DEBUG: Binding resolution LLM 호출 완료 ===")

    content = response.content or "{}"
    print(f"=== DEBUG: Binding resolution 응답: {content[:200]}... ===")

    resolved_bindings = json.loads(content)
    print(f"=== DEBUG: Resolved Bindings: {resolved_bindings} ===")
    return resolved_bindings


def _resolve_bindings_fallback(bindings: dict, subtask_results: list) -> dict:
    """
    Fallback method: Resolve abstract bindings to concrete values from previous subtask results
    
    - $task_0.feature_id → subtask_results[0]["reference_features"][0]["feature_id"]
    - $task_0.feature_name → subtask_results[0]["reference_features"][0]["feature_name"]
    """
    if not bindings:
        return {}

    resolved = {}
    for binding_key, binding_ref in bindings.items():
        print(f"=== DEBUG: Binding '{binding_key}': '{binding_ref}' 해결 중 ===")

        if isinstance(binding_ref, str) and binding_ref.startswith("$task_"):
            # Parse $task_{id}.{field} format
            parts = binding_ref.replace("$task_", "").split(".")
            if len(parts) == 2:
                task_id, field_name = parts
                task_id_int = int(task_id)

                # subtask_results에서 해당 task_id 찾기
                found_value = None

                # reference_features에서 찾기 
                for result in subtask_results:
                    result_id = result.get("id", result.get("task_id"))
                    if result_id == task_id_int:
                        ref_features = result.get("reference_features", [])
                        if ref_features and len(ref_features) > 0:
                            for ref in ref_features:
                                if field_name in ref:
                                    found_value = ref[field_name]
                                    print(f"=== DEBUG: reference_features에서 찾음: {found_value} ===")
                                    break

                        # reference_features에 없으면 subtask_answer에서 추출 시도
                        if found_value is None:
                            subtask_answer = result.get("subtask_answer", "")
                            refined_text = result.get("refined_text", "")

                            # feature_id 패턴 찾기 (예: FGR-BC0311)
                            if field_name == "feature_id" or field_name == "feature_id":
                                feature_id_match = re.search(r'FGR-[A-Z]{2}\d{4}',
                                                             subtask_answer + refined_text)
                                if feature_id_match:
                                    found_value = feature_id_match.group(0)
                                    print(f"=== DEBUG: 텍스트에서 feature_id 추출: {found_value} ===")

                            # feature_name 패턴 찾기
                            elif field_name == "feature_name":
                                # 일반적인 feature name 패턴 찾기
                                lines = (subtask_answer + refined_text).split('\n')
                                for line in lines:
                                    if 'feature' in line.lower() and 'name' in line.lower():
                                        found_value = line.split(':')[
                                            -1].strip() if ':' in line else line.strip()
                                        break

                        if found_value:
                            break

                if found_value:
                    resolved[binding_key] = found_value
                else:
                    # 최종 폴백: unresolved 표시
                    resolved[binding_key] = f"unresolved_{task_id}_{field_name}"
                    print(f"=== DEBUG: Binding 해결 실패 - unresolved_{task_id}_{field_name} ===")
        else:
            resolved[binding_key] = binding_ref

    print(f"=== DEBUG: Fallback으로 해결된 Bindings: {resolved} ===")
    return resolved


# 노드 인스턴스 생성
var_binder_node = VarBinderNode()
