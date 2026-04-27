# langgraph_agenticrag/src/agents/nodes/refiner_node.py

import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.agents._subtask_utils import (
    format_features,
    pick_latest_successful,
    result_payload,
    truncate,
)
from langgraph_flow.prompts.refiner import REFINER_PROMPT_TEMPLATE

GOAL_PREVIEW_MAX = 80
ANSWER_PREVIEW_MAX = 300


class RefinerNode:
    """Refiner 노드 클래스"""

    def __init__(self):
        self.name = "refiner"
        self.max_retries = 3
        self.system_prompt = REFINER_PROMPT_TEMPLATE

    async def invoke(self, state: AgentState,
                     config: Optional[RunnableConfig] = None) -> AgentState:
        """
        Refiner 노드 실행
        
        Args:
            state: 현재 에이전트 상태
            config: 설정 정보 (선택적)
            
        Returns:
            업데이트된 상태
        """
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

        return await self.refine_results(state, llm)

    async def refine_results(self, state: AgentState, llm: BaseLanguageModel = None) -> AgentState:
        """
        실행 결과를 정제하는 함수
        
        refiner가 다음을 반환:
        - refined_text: 정제된 검색 결과
        - subtask_answer: subtask에 대한 답변
        - reference_features: [{"feature_id": "...", "feature_name": "..."}]
        - verdict: 성공 여부
        - retry_reason: 재시도 사유 (verdict=false인 경우)
        
        이 정보는 다음 subtask의 binding resolution에 활용됨.
        
        retriever_history를 기반으로 동작
        
        Args:
            state: 현재 에이전트 상태
            llm: 언어 모델 (선택적)
            
        Returns:
            업데이트된 상태
        """
        retry_counts = state.get("retry_counts", {}).copy()
        subtasks = state.get("subtasks", [])
        subtask_results = state.get("subtask_results", [])
        reference_features = state.get("reference_features", [])

        retriever_history = state.get("retriever_history", [])

        # 현재 실행 중인 subtask ID 사용 (executor에서 설정)
        latest_subtask_id = state.get("current_executing_subtask_id")
        print(f"=== DEBUG: Refiner - Executor에서 전달받은 Subtask ID: {latest_subtask_id} ===")

        # 현재 subtask의 결과가 retriever_history에 있는지 확인
        has_result = any(
            h.get("subtask_id") == latest_subtask_id
            for h in retriever_history
        )

        if not has_result:
            print(f"=== DEBUG: Refiner - Subtask {latest_subtask_id}의 검색 결과 없음 - synthesizer로 이동 ===")
            return update_state(state, next="synthesizer")

        # retriever_history에서 현재 subtask의 결과만 추출
        retriever_outputs = [
            h for h in retriever_history
            if h.get("subtask_id") == latest_subtask_id
        ]
        print(f"=== DEBUG: Refiner - retriever_history에서 추출한 결과: {len(retriever_outputs)}개 ===")

        try:
            # Retriever 출력 결과 수집
            retriever_results = self._collect_retriever_results(retriever_outputs)

            # LLM을 사용한 정제 및 verdict 결정
            if llm is None:
                # 테스트용 기본 정제 - 항상 성공으로 처리
                refined_output = self._default_refine(retriever_results)
                verdict = True

                # 원본 데이터에서 feature 추출 (LLM 없어도 수행)
                raw_extracted_features = self._extract_features_from_raw_results(retriever_outputs)
                print(f"=== DEBUG: LLM 없음 - 원본 데이터에서 추출된 features: {raw_extracted_features} ===")

                refined_data = {
                    "refined_text": refined_output,
                    "subtask_answer": refined_output[:500] if len(
                        refined_output) > 500 else refined_output,
                    "reference_features": raw_extracted_features,
                    "verdict": True
                }
            else:
                # LLM을 사용한 정제 및 verdict 결정
                refined_data = await self._llm_refine_with_verdict(retriever_results, state, llm)

            verdict = refined_data.get("verdict", True)
            reference_features_found = refined_data.get("reference_features", [])
            subtask_answer = refined_data.get("subtask_answer", "")
            refined_text = refined_data.get("refined_text", "")

            updated_subtasks = []
            for subtask in subtasks:
                subtask_copy = subtask.copy()
                if subtask_copy.get("id") == latest_subtask_id:
                    subtask_copy["verdict"] = verdict
                    subtask_copy["subtask_answer"] = subtask_answer
                    subtask_copy["refined_text"] = refined_text
                    subtask_copy["reference_features"] = reference_features_found
                    print(f"=== DEBUG: Subtask {latest_subtask_id} verdict 설정: {verdict} ===")
                    print(
                        f"=== DEBUG: Subtask {latest_subtask_id} reference_features: {reference_features_found} ===")

                    # verdict=false 시 retry_reason 구조화 적용 → 다음 시도 goal 재작성
                    if not verdict:
                        rr_raw = refined_data.get("retry_reason")
                        rr_dict = self._normalize_retry_reason(rr_raw)

                        # excluded_doc_ids 누적 (attempt 간 union)
                        existing_excluded = subtask_copy.get("excluded_doc_ids") or []
                        new_excluded = rr_dict.get("excluded_doc_ids") or []
                        merged_excluded = list(
                            dict.fromkeys([*existing_excluded, *new_excluded])
                        )
                        if merged_excluded:
                            subtask_copy["excluded_doc_ids"] = merged_excluded
                            if new_excluded:
                                print(
                                    f"=== DEBUG: Subtask {latest_subtask_id} excluded_doc_ids "
                                    f"누적: {merged_excluded} ==="
                                )

                        # seen_feature_ids 누적 (id only, dedup, excluded 제거) — 다음 시도 dedup 용
                        prior_seen = subtask_copy.get("seen_feature_ids") or []
                        current_ref = refined_data.get("reference_features") or []
                        excluded_set = set(merged_excluded)
                        merged_seen: List[str] = []
                        seen_set: set = set()
                        for src in (prior_seen, current_ref):
                            for item in src or []:
                                if isinstance(item, str):
                                    fid = item.strip()
                                elif isinstance(item, dict):
                                    fid = str(item.get("feature_id") or "").strip()
                                else:
                                    fid = ""
                                if not fid or fid in seen_set or fid in excluded_set:
                                    continue
                                merged_seen.append(fid)
                                seen_set.add(fid)
                        subtask_copy["seen_feature_ids"] = merged_seen
                        subtask_copy["retry_reason"] = rr_dict

                        new_goal = (rr_dict.get("suggested_next_goal") or "").strip()
                        if new_goal:
                            if not subtask_copy.get("original_goal"):
                                subtask_copy["original_goal"] = subtask_copy.get(
                                    "goal", subtask_copy.get("description", "")
                                )
                            print(
                                f"=== DEBUG: Subtask {latest_subtask_id} goal 재작성: "
                                f"{subtask_copy.get('goal')!r} → {new_goal!r} ==="
                            )
                            subtask_copy["goal"] = new_goal
                updated_subtasks.append(subtask_copy)

            # ⭐ current_step 증가 (원본 코드의 iteration_count += 1과 동일)
            current_step = state.get("current_step", 0) + 1

            # 상태 업데이트 변수 초기화
            update_kwargs = {
                "subtasks": updated_subtasks,
                "retry_counts": retry_counts,
                "current_step": current_step,  # 증가된 step 저장
                "next": "synthesizer"
            }

            print(f"=== DEBUG: Current Step 증가: {current_step} ===")

            # 시도 번호 캡처 (retry_counts 증가 전 — 이번 refiner 호출이 N번째 시도인지)
            current_attempt = retry_counts.get(latest_subtask_id, 0)

            # 재시도 로직 - 원본 agent_service.py와 동일한 로직
            if not verdict:
                # 재시도 횟수 증가
                current_retries = retry_counts.get(latest_subtask_id, 0)
                retry_counts[latest_subtask_id] = current_retries + 1
                print(f"=== DEBUG: Subtask {latest_subtask_id} 재시도 {current_retries + 1}/3 ===")

                # 최대 재시도 초과 시 exceeded로 설정 (원본과 동일)
                if current_retries + 1 >= 3:
                    for subtask in updated_subtasks:
                        if subtask.get("id") == latest_subtask_id:
                            subtask["verdict"] = "exceeded"
                            subtask["subtask_answer"] = "최대 재시도 횟수를 초과하여 요청을 중단합니다."
                            subtask["retry_reason"] = "최대 재시도 횟수(3회) 초과"
                            print(
                                f"=== DEBUG: Subtask {latest_subtask_id} 최대 재시도 초과 - exceeded 설정 ===")

            # verdict=True의 경우 reference_features 머지 (reducer가 기존 state와 merge)
            if verdict and reference_features_found:
                update_kwargs["reference_features"] = reference_features_found

            # 모든 시도(success/failure/exceeded)를 attempt 키와 함께 subtask_results에 push
            # 같은 (id, attempt) 조합만 dedupe → retry 시 이전 시도 결과 보존
            # subtask config (goal/action/bindings 등)는 봉투에서 제외 — concept 4(subtasks)와 분리
            current_subtask_state = next(
                (s for s in updated_subtasks if s.get("id") == latest_subtask_id),
                None,
            )
            if current_subtask_state is not None:
                entry_verdict = current_subtask_state.get("verdict")
                result_payload = {
                    "subtask_answer": current_subtask_state.get("subtask_answer", ""),
                    "refined_text": current_subtask_state.get("refined_text", ""),
                    "reference_features": current_subtask_state.get("reference_features", []),
                }
                result_entry = {
                    "id": latest_subtask_id,
                    "attempt": current_attempt,
                    "verdict": entry_verdict,
                    "result": result_payload,
                }
                if not entry_verdict and isinstance(refined_data, dict):
                    rr = refined_data.get("retry_reason", "")
                    if rr:
                        result_entry["retry_reason"] = rr
                if entry_verdict == "exceeded" and current_subtask_state.get("retry_reason"):
                    result_entry.setdefault("retry_reason", current_subtask_state["retry_reason"])
                update_kwargs["subtask_results"] = [result_entry]
                print(
                    f"=== DEBUG: subtask_results push: id={latest_subtask_id} "
                    f"attempt={current_attempt} verdict={entry_verdict} ==="
                )

            updated_state = update_state(state, **update_kwargs)

            return updated_state

        except Exception as e:
            print(f"Refiner error: {e}")
            import traceback

            traceback.print_exc()
            # 재시도 로직
            current_retries = retry_counts.get("refiner", 0)
            if current_retries < 3:
                retry_counts["refiner"] = current_retries + 1
                return update_state(
                    state,
                    retry_counts=retry_counts,
                    next="refiner"  # 다시 시도
                )
            else:
                # 최대 재시도 횟수 초과 시 다음 노드로 이동
                return update_state(state, next="synthesizer")

    def _normalize_retry_reason(self, raw: Any) -> Dict[str, Any]:
        """
        retry_reason 입력을 표준 dict 형태로 정규화.

        지원 입력:
        - dict: 누락 필드 기본값으로 채워서 반환
        - str: 자유 텍스트를 missing_info에 보존, suggested_next_goal은 비움
                (다음 시도가 사실상 무의미하므로 LLM이 재구성 못한 케이스로 취급)
        - None / 기타: 빈 표준 dict

        반환 키:
            failure_type, missing_info, irrelevant_aspects,
            query_hints (List[str]), excluded_doc_ids (List[str]),
            suggested_next_goal
        """
        defaults = {
            "failure_type": "unknown",
            "missing_info": "",
            "irrelevant_aspects": "",
            "query_hints": [],
            "excluded_doc_ids": [],
            "suggested_next_goal": "",
        }
        if isinstance(raw, dict):
            out = dict(defaults)
            for k in defaults:
                if k in raw and raw[k] is not None:
                    out[k] = raw[k]
            # 리스트 필드 형식 보정
            if not isinstance(out["query_hints"], list):
                out["query_hints"] = [str(out["query_hints"])] if out["query_hints"] else []
            if not isinstance(out["excluded_doc_ids"], list):
                out["excluded_doc_ids"] = (
                    [str(out["excluded_doc_ids"])] if out["excluded_doc_ids"] else []
                )
            return out
        if isinstance(raw, str) and raw.strip():
            return {**defaults, "missing_info": raw.strip()}
        return dict(defaults)

    def _collect_retriever_results(self, retriever_outputs: List[Dict]) -> List[Dict]:
        """
        Retriever 출력 결과 수집
        
        Args:
            retriever_outputs: Retriever 출력 결과
            
        Returns:
            수집된 결과 리스트
        """
        all_results = []

        for output in retriever_outputs:
            result = output.get("result", {})
            if isinstance(result, dict):
                # result가 dict인 경우 results 필드에서 텍스트 추출
                results_list = result.get("results", [])
                if isinstance(results_list, list):
                    for item in results_list:
                        if isinstance(item, str):
                            all_results.append({
                                "text": item,
                                "subtask_id": output.get("subtask_id", "unknown")
                            })
                        elif isinstance(item, dict) and "text" in item:
                            all_results.append({
                                "text": item["text"],
                                "subtask_id": output.get("subtask_id", "unknown")
                            })
                elif isinstance(results_list, str):
                    all_results.append({
                        "text": results_list,
                        "subtask_id": output.get("subtask_id", "unknown")
                    })
            elif isinstance(result, list):
                # result가 list인 경우
                for item in result:
                    if isinstance(item, str):
                        all_results.append({
                            "text": item,
                            "subtask_id": output.get("subtask_id", "unknown")
                        })
                    elif isinstance(item, dict) and "text" in item:
                        all_results.append({
                            "text": item["text"],
                            "subtask_id": output.get("subtask_id", "unknown")
                        })

        return all_results

    def _default_refine(self, results: List[Dict]) -> str:
        """
        기본 정제 함수 (테스트용)
        
        Args:
            results: 실행 결과 리스트
            
        Returns:
            정제된 결과 문자열
        """
        if not results:
            return "No results to refine"

        refined_parts = []
        for result in results:
            text = result.get("text", "")
            if text:
                refined_parts.append(text)

        return "\n\n".join(refined_parts) if refined_parts else "No results to refine"

    def _extract_features_from_raw_results(self, retriever_outputs: List[Dict]) -> List[
        Dict[str, str]]:
        """
        원본 검색 결과(retriever_outputs)에서 직접 feature_id와 feature_name 추출
        
        변환되지 않은 원본 데이터를 사용하여 정보 손실 방지
        
        Args:
            retriever_outputs: 원본 retriever 출력 결과
            
        Returns:
            추출된 feature 리스트 [{"feature_id": "...", "feature_name": "..."}]
        """
        import re

        features = []
        seen_ids = set()

        for output in retriever_outputs:
            result = output.get("result", {})

            # result에서 results 리스트 추출
            if isinstance(result, dict):
                results_list = result.get("results", [])
            elif isinstance(result, list):
                results_list = result
            else:
                continue

            if not isinstance(results_list, list):
                continue

            for item in results_list:
                if not isinstance(item, str):
                    continue

                # feature_id 패턴 찾기 (예: FGR-BC0311, FGR-RS0751)
                feature_id_matches = re.findall(r'FGR-[A-Z]{2}\d{4}', item)

                # feature_name 패턴 찾기 (JSON 형식)
                # 형식 1: "feature_name": "..."
                feature_name_matches = re.findall(r'"feature_name":\s*"([^"]+)"', item)

                for i, feature_id in enumerate(feature_id_matches):
                    if feature_id not in seen_ids:
                        feature_name = ""
                        if i < len(feature_name_matches):
                            feature_name = feature_name_matches[i].strip()

                        features.append({
                            "feature_id": feature_id,
                            "feature_name": feature_name
                        })
                        seen_ids.add(feature_id)
                        print(f"=== DEBUG: 원본 데이터에서 feature 추출: {feature_id} - {feature_name} ===")

        return features

    def _extract_features_from_results(self, results: List[Dict]) -> List[Dict[str, str]]:
        """
        검색 결과에서 feature_id와 feature_name 직접 추출 (fallback)
        
        LLM이 추출하지 못한 경우 검색 결과 텍스트에서 정규식으로 추출
        
        Args:
            results: 검색 결과 리스트
            
        Returns:
            추출된 feature 리스트 [{"feature_id": "...", "feature_name": "..."}]
        """
        import re

        features = []
        seen_ids = set()

        for result in results:
            text = result.get("text", "")
            if not text:
                continue

            # feature_id 패턴 찾기 (예: FGR-BC0311, FGR-RS0751)
            feature_id_matches = re.findall(r'FGR-[A-Z]{2}\d{4}', text)

            # feature_name 패턴 찾기 (다양한 형식 지원)
            # 형식 1: "feature_name": "..."
            feature_name_matches = re.findall(r'"feature_name":\s*"([^"]+)"', text)
            # 형식 2: Feature Name: ...
            feature_name_matches += re.findall(r'Feature\s*Name:\s*([^\n]+)', text, re.IGNORECASE)
            # 형식 3: Name: ... (feature_id 근처)

            for i, feature_id in enumerate(feature_id_matches):
                if feature_id not in seen_ids:
                    feature_name = ""
                    if i < len(feature_name_matches):
                        feature_name = feature_name_matches[i].strip()

                    features.append({
                        "feature_id": feature_id,
                        "feature_name": feature_name
                    })
                    seen_ids.add(feature_id)

        return features

    async def _llm_refine_with_verdict(self, results: List[Dict], state: AgentState,
                                       llm: BaseLanguageModel) -> dict:
        """
        LLM을 사용한 결과 정제 및 verdict 결정
        
        Args:
            results: 실행 결과 리스트 (변환된 형식)
            state: 현재 상태
            llm: 언어 모델
            
        Returns:
            정제된 결과 dict (refined_text, subtask_answer, reference_features, verdict, retry_reason)
        """
        # 현재 subtask 정보 가져오기
        latest_subtask_id = state.get("current_executing_subtask_id", 0)
        subtasks = state.get("subtasks", [])
        current_subtask = None
        for subtask in subtasks:
            if subtask.get("id") == latest_subtask_id:
                current_subtask = subtask
                break

        # ⭐ retriever_history에서 현재 subtask의 결과 추출
        retriever_history = state.get("retriever_history", [])
        retriever_outputs = [
            h for h in retriever_history
            if h.get("subtask_id") == latest_subtask_id
        ]
        raw_extracted_features = self._extract_features_from_raw_results(retriever_outputs)
        print(f"=== DEBUG: 원본 데이터에서 추출된 features: {raw_extracted_features} ===")

        # 검색 결과 텍스트 포맷팅
        results_text = self._format_results_for_refiner(results)

        # 다른 subtask 컨텍스트 — 격리된 refine 방지 (cross-subtask 정합성)
        plan_overview_text = self._format_plan_overview(subtasks, latest_subtask_id)
        other_results_text = self._format_other_subtask_results(state, latest_subtask_id)
        cumulative_features_text = self._format_cumulative_features(
            state.get("reference_features", [])
        )

        user_content = f"""사용자 질문: {state.get('user_query', '')}

전체 실행 계획 (Plan Overview):
{plan_overview_text}

이전 Subtask 결과 (cross-reference 용):
{other_results_text}

지금까지 누적된 reference_features (중복 제외):
{cumulative_features_text}

현재 Subtask: {json.dumps(current_subtask, ensure_ascii=False, indent=2) if current_subtask else 'N/A'}

검색 결과:
{results_text}

위 검색 결과를 분석하고, subtask의 목표를 달성했는지 판단하여 JSON 형식으로 반환해주세요.
reference_features 필드에 검색 결과에서 사용된 모든 feature의 feature_id와 feature_name을 반드시 포함하세요.
이전 Subtask 결과와 누적 reference_features를 고려해 중복은 제외하되 누락 없이 모든 신규 feature를 포함하세요.

형식: {{"feature_id": "FGR-XXXX", "feature_name": "..."}}"""

        # LLM 호출
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ]

        try:
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
            print(f"=== DEBUG: Refiner LLM 응답 ===")
            print(content[:500] if len(content) > 500 else content)

            # JSON 파싱 시도
            try:
                result_dict = json.loads(content)
                # 필수 필드 확인
                refined_data = {
                    "refined_text": result_dict.get("refined_text", ""),
                    "subtask_answer": result_dict.get("subtask_answer", ""),
                    "reference_features": result_dict.get("reference_features", []),
                    "verdict": result_dict.get("verdict", True),
                    "retry_reason": result_dict.get("retry_reason", "")
                }

                # reference_features 추출 우선순위:
                # 1. 원본 데이터에서 직접 추출 (가장 신뢰할 수 있음)
                # 2. LLM이 추출한 값
                # 3. 변환된 결과에서 추출 (fallback)
                if raw_extracted_features:
                    refined_data["reference_features"] = raw_extracted_features
                    print(f"=== DEBUG: 원본 데이터에서 추출된 features 사용: {raw_extracted_features} ===")
                elif not refined_data["reference_features"]:
                    extracted_features = self._extract_features_from_results(results)
                    if extracted_features:
                        refined_data["reference_features"] = extracted_features
                        print(f"=== DEBUG: Fallback으로 추출된 features: {extracted_features} ===")

                return refined_data
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 기본값 + 원본에서 추출
                print(f"=== DEBUG: JSON 파싱 실패, 기본값 + 원본 추출 사용 ===")
                # 원본 데이터에서 추출 우선
                features_to_use = raw_extracted_features if raw_extracted_features else self._extract_features_from_results(
                    results)
                return {
                    "refined_text": content,
                    "subtask_answer": content,
                    "reference_features": features_to_use,
                    "verdict": True
                }
        except Exception as e:
            print(f"LLM refine error: {e}")
            import traceback

            traceback.print_exc()
            # 에러 시에도 원본 데이터에서 features 추출 시도
            features_to_use = raw_extracted_features if raw_extracted_features else self._extract_features_from_results(
                results)
            return {
                "refined_text": str(e),
                "subtask_answer": "",
                "reference_features": features_to_use,
                "verdict": False
            }

    def _format_plan_overview(self, subtasks: List[Dict], current_id: Any) -> str:
        if not subtasks:
            return "(plan 없음)"
        lines = []
        for s in subtasks:
            sid = s.get("id", "?")
            verdict = s.get("verdict", False)
            if sid == current_id:
                status = "current"
            elif verdict is True:
                status = "done"
            elif verdict == "exceeded":
                status = "exceeded"
            else:
                status = "pending"
            goal = truncate((s.get("goal") or s.get("description") or "").strip(), GOAL_PREVIEW_MAX)
            lines.append(f"  [{sid}] ({status}) {goal}")
        return "\n".join(lines)

    def _format_other_subtask_results(self, state: AgentState, current_id: Any) -> str:
        latest_per_id = pick_latest_successful(
            state.get("subtask_results", []), exclude_id=current_id
        )
        if not latest_per_id:
            return "(이전 완료 subtask 없음)"
        lines = []
        for sid in sorted(latest_per_id.keys()):
            payload = result_payload(latest_per_id[sid])
            answer = truncate(
                (payload["subtask_answer"] or payload["refined_text"]).strip(),
                ANSWER_PREVIEW_MAX,
            )
            ref_compact = format_features(payload["reference_features"], sep=", ")
            lines.append(f"[Subtask {sid}]")
            lines.append(f"  answer: {answer or '(빈 답변)'}")
            lines.append(f"  features: {ref_compact}")
        return "\n".join(lines)

    def _format_cumulative_features(self, features: List[Dict]) -> str:
        return format_features(features, line_prefix="  - ", empty="(누적 feature 없음)")

    def _format_results_for_refiner(self, results: List[Dict]) -> str:
        """
        Refiner용 결과 포맷팅
        
        Args:
            results: 실행 결과 리스트
            
        Returns:
            포맷팅된 결과 문자열
        """
        refined_parts = []

        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            if text:
                refined_parts.append(f"[Document {i}/{len(results)}]")
                refined_parts.append(text)
                refined_parts.append("")

        return "\n".join(refined_parts) if refined_parts else "No results to refine"


# 노드 인스턴스 생성
refiner_node = RefinerNode()
