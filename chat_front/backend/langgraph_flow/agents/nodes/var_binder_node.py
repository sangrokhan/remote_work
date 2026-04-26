# langgraph_agenticrag/src/agents/nodes/var_binder_node.py

import json
import logging
import re
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.agents._subtask_utils import (
    pick_latest_successful,
    result_payload as _result_payload,
)
from langgraph_flow.prompts.var_binder import BINDER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


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
    prev_resolved = state.get("resolved_bindings") or {}

    # 실행 가능한 subtask 찾기 (verdict가 False인 것 중 의존성이 해결된 것)
    executable_subtask = _get_next_executable_subtask(subtasks, subtask_results)

    if not executable_subtask:
        logger.info("[VarBinder] no executable subtask → synthesizer")
        return update_state(state, next="synthesizer")

    # 바인딩 정보 추출
    bindings = executable_subtask.get("bindings", {})
    logger.info(
        "[VarBinder] selected subtask_id=%s bindings=%s subtask_results_count=%d",
        executable_subtask.get("id"), list(bindings.keys()), len(subtask_results)
    )

    # 바인딩이 없으면 바로 executor로
    if not bindings:
        logger.info("[VarBinder] subtask_id=%s no bindings → executor directly", executable_subtask["id"])
        return update_state(
            state,
            current_executing_subtask_id=executable_subtask["id"],
            resolved_bindings=prev_resolved,
            next="executor"
        )

    # subtask_results가 비어있으면 폴백 사용
    if not subtask_results:
        logger.warning("[VarBinder] subtask_id=%s no subtask_results → fallback resolution", executable_subtask["id"])
        new_resolved = _resolve_bindings_fallback(bindings, [])
        resolved_bindings = {**prev_resolved, **new_resolved}
        logger.info("[VarBinder] fallback resolved=%s merged_keys=%s", new_resolved, list(resolved_bindings.keys()))
        return update_state(
            state,
            current_executing_subtask_id=executable_subtask["id"],
            resolved_bindings=resolved_bindings,
            next="executor"
        )

    # LLM이 제공된 경우 LLM 기반 binding 해결
    if llm:
        try:
            new_resolved = await _resolve_bindings_with_llm(
                bindings,
                subtask_results,
                {
                    "current_subtask": executable_subtask,
                    "user_query": state.get("user_query", ""),
                    "dependencies": executable_subtask.get("dependencies", [])
                },
                llm
            )
            resolved_bindings = {**prev_resolved, **new_resolved}
            logger.info("[VarBinder] LLM resolved subtask_id=%s new=%s merged_keys=%s",
                        executable_subtask["id"], new_resolved, list(resolved_bindings.keys()))
            return update_state(
                state,
                current_executing_subtask_id=executable_subtask["id"],
                resolved_bindings=resolved_bindings,
                next="executor"
            )
        except Exception as e:
            logger.error("[VarBinder] LLM resolution error subtask_id=%s: %s → fallback", executable_subtask["id"], e)

    # 폴백: 직접 binding 해결
    new_resolved = _resolve_bindings_fallback(bindings, subtask_results)
    resolved_bindings = {**prev_resolved, **new_resolved}
    logger.info("[VarBinder] fallback resolved subtask_id=%s new=%s merged_keys=%s",
                executable_subtask["id"], new_resolved, list(resolved_bindings.keys()))

    return update_state(
        state,
        current_executing_subtask_id=executable_subtask["id"],
        resolved_bindings=resolved_bindings,
        next="executor"
    )


def _get_next_executable_subtask(subtasks: List[Dict], subtask_results: List[Dict]) -> Optional[Dict]:
    """
    실행 가능한 다음 subtask를 찾는 함수

    Args:
        subtasks: 전체 subtask 리스트
        subtask_results: 완료된 subtask 결과들 (verdict True/exceeded 기준)

    Returns:
        실행 가능한 subtask 또는 None
    """
    completed_ids = set()
    for result in subtask_results:
        subtask_id = result.get("id")
        verdict = result.get("verdict")
        # verdict=True 또는 exceeded만 완료로 간주. 실패 attempt는 dep 해결에서 제외.
        if subtask_id is not None and (verdict is True or verdict == "exceeded"):
            completed_ids.add(subtask_id)

    logger.debug("[VarBinder] completed_ids=%s", completed_ids)
    for subtask in subtasks:
        subtask_id = subtask.get("id")

        # 이미 완료된 subtask는 건너뜀
        if subtask_id in completed_ids:
            logger.debug("[VarBinder] skip subtask_id=%s (in completed_ids)", subtask_id)
            continue

        # verdict가 True이면 완료된 것으로 간주
        if subtask.get("verdict", False) is True:
            logger.debug("[VarBinder] skip subtask_id=%s (verdict=True)", subtask_id)
            continue

        # 의존성 확인
        dependencies = subtask.get("dependencies", [])
        if not dependencies:
            logger.debug("[VarBinder] selected subtask_id=%s (no dependencies)", subtask_id)
            return subtask

        # 모든 의존성이 완료되었는지 확인
        all_deps_completed = all(dep_id in completed_ids for dep_id in dependencies)
        if all_deps_completed:
            logger.debug("[VarBinder] selected subtask_id=%s (all deps met: %s)", subtask_id, dependencies)
            return subtask
        else:
            pending = [d for d in dependencies if d not in completed_ids]
            logger.debug("[VarBinder] skip subtask_id=%s (pending deps: %s)", subtask_id, pending)

    logger.info("[VarBinder] no executable subtask found")
    return None


async def _resolve_bindings_with_llm(bindings: dict, subtask_results: list,
                                     subtask_context: dict, llm: BaseLanguageModel) -> dict:
    """
    Resolve abstract bindings to concrete values using LLM

    - subtask_results에서 이전 subtask의 reference_features를 찾음
    - $subtask_0.feature_id → subtask_results[0]["reference_features"][0]["feature_id"]
    """
    print(f"=== DEBUG: _resolve_bindings_with_llm 호출 ===")
    print(f"Bindings: {bindings}")
    print(f"Subtask Results Count: {len(subtask_results)}")

    if not bindings:
        return {}

    # envelope 평면화: LLM 프롬프트엔 결과 payload만 노출 (id/attempt/verdict 제외)
    latest = pick_latest_successful(subtask_results, key_as_str=True)
    subtask_results_dict: Dict[str, Dict] = {k: _result_payload(v) for k, v in latest.items()}

    resolution_messages = [
        SystemMessage(content=BINDER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Bindings to resolve: {json.dumps(bindings, ensure_ascii=False)}\n\nPrevious results: {json.dumps(subtask_results_dict, ensure_ascii=False, indent=2)}\n\nCurrent context: {json.dumps(subtask_context, ensure_ascii=False)}")
    ]

    print("=== DEBUG: Binding resolution LLM 호출 시작 ===")
    response = await llm.bind(temperature=0.1).ainvoke(resolution_messages)
    print("=== DEBUG: Binding resolution LLM 호출 완료 ===")

    content = response.content or "{}"
    print(f"=== DEBUG: Binding resolution 응답: {content[:200]}... ===")
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```", 2)[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.rsplit("```", 1)[0].strip()
    if not content:
        content = "{}"
    resolved_bindings = json.loads(content)
    print(f"=== DEBUG: Resolved Bindings: {resolved_bindings} ===")
    return resolved_bindings


def _resolve_bindings_fallback(bindings: dict, subtask_results: list) -> dict:
    """
    Fallback method: Resolve abstract bindings to concrete values from previous subtask results

    - $subtask_0.feature_id → subtask_results[0]["reference_features"][0]["feature_id"]
    - $subtask_0.feature_name → subtask_results[0]["reference_features"][0]["feature_name"]
    """
    if not bindings:
        return {}

    resolved = {}
    for binding_key, binding_ref in bindings.items():
        print(f"=== DEBUG: Binding '{binding_key}': '{binding_ref}' 해결 중 ===")

        if isinstance(binding_ref, str) and binding_ref.startswith("$subtask_"):
            # Parse $subtask_{id}.{field} format
            parts = binding_ref.replace("$subtask_", "").split(".")
            if len(parts) == 2:
                subtask_id, field_name = parts
                subtask_id_int = int(subtask_id)

                # subtask_results에서 해당 subtask_id 찾기 (verdict=True인 최신 attempt만)
                found_value = None

                # id별 verdict=True 최신 attempt 미리 추출
                latest_by_id: Dict[int, Dict] = {}
                for r in subtask_results:
                    if r.get("verdict") is not True:
                        continue
                    rid = r.get("id")
                    if rid is None:
                        continue
                    if rid not in latest_by_id or r.get("attempt", 0) > latest_by_id[rid].get("attempt", 0):
                        latest_by_id[rid] = r

                # envelope payload 평면화 후 lookup
                # 여러 reference_features 엔트리에서 field_name 값 모두 수집 (순서 보존 dedupe).
                # 단수면 기존처럼 단일 문자열, 복수면 공백 join — 다운스트림 substitution과 호환.
                matched = latest_by_id.get(subtask_id_int)
                if matched is not None:
                    payload = _result_payload(matched)
                    ref_features = payload["reference_features"]
                    collected: List[str] = []
                    seen_vals: set = set()
                    if ref_features:
                        for ref in ref_features:
                            if field_name in ref:
                                val = ref[field_name]
                                if val is None:
                                    continue
                                val_str = str(val).strip()
                                if not val_str or val_str in seen_vals:
                                    continue
                                seen_vals.add(val_str)
                                collected.append(val_str)

                    if collected:
                        found_value = collected[0] if len(collected) == 1 else " ".join(collected)
                        print(f"=== DEBUG: reference_features에서 찾음 (n={len(collected)}): {found_value} ===")

                    # reference_features에 없으면 subtask_answer에서 추출 시도
                    if found_value is None:
                        subtask_answer = payload["subtask_answer"]
                        refined_text = payload["refined_text"]
                        haystack = subtask_answer + refined_text

                        # feature_id 패턴 찾기 (예: FGR-BC0311) — 모든 매치 dedupe 후 join
                        if field_name == "feature_id":
                            matches = re.findall(r'FGR-[A-Z]{2}\d{4}', haystack)
                            unique_ids: List[str] = []
                            seen_ids: set = set()
                            for m in matches:
                                if m not in seen_ids:
                                    seen_ids.add(m)
                                    unique_ids.append(m)
                            if unique_ids:
                                found_value = unique_ids[0] if len(unique_ids) == 1 else " ".join(unique_ids)
                                print(f"=== DEBUG: 텍스트에서 feature_id 추출 (n={len(unique_ids)}): {found_value} ===")

                        # feature_name 패턴 찾기
                        elif field_name == "feature_name":
                            names: List[str] = []
                            seen_names: set = set()
                            for line in haystack.split('\n'):
                                if 'feature' in line.lower() and 'name' in line.lower():
                                    name = line.split(':')[-1].strip() if ':' in line else line.strip()
                                    if name and name not in seen_names:
                                        seen_names.add(name)
                                        names.append(name)
                            if names:
                                found_value = names[0] if len(names) == 1 else " ".join(names)

                if found_value:
                    resolved[binding_key] = found_value
                else:
                    # 최종 폴백: unresolved 표시
                    resolved[binding_key] = f"unresolved_{subtask_id}_{field_name}"
                    print(f"=== DEBUG: Binding 해결 실패 - unresolved_{subtask_id}_{field_name} ===")
        else:
            resolved[binding_key] = binding_ref

    print(f"=== DEBUG: Fallback으로 해결된 Bindings: {resolved} ===")
    return resolved


# 노드 인스턴스 생성
var_binder_node = VarBinderNode()
