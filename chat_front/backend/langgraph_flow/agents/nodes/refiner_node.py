"""
Refiner node — 실행 결과 정제 및 verdict 결정.
retriever_history에서 현재 subtask 결과를 추출해 LLM으로 정제하고 subtask_results에 저장.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.prompts.refiner import REFINER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class RefinerNode:
    """Refiner 노드 클래스"""

    def __init__(self):
        self.name = "refiner"
        self.max_retries = 3
        self.system_prompt = REFINER_PROMPT_TEMPLATE

    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        # synthesizer_node와 동일한 패턴으로 LLM 추출
        configurable = (config or {}).get("configurable", {})
        llm: Optional[BaseLanguageModel] = configurable.get("llm")
        return await self.refine_results(state, llm)

    async def refine_results(self, state: AgentState, llm: Optional[BaseLanguageModel] = None) -> AgentState:
        """
        실행 결과를 정제하는 함수

        반환 구조:
        - refined_text: 정제된 검색 결과
        - subtask_answer: subtask에 대한 답변
        - reference_features: [{"feature_id": "...", "feature_name": "..."}]
        - verdict: 성공 여부
        - retry_reason: 재시도 사유 (verdict=false인 경우)

        이 정보는 다음 subtask의 binding resolution에 활용됨.
        """
        retry_counts = state.get("retry_counts", {}).copy()
        subtasks = state.get("subtasks", [])
        subtask_results = state.get("subtask_results", [])
        reference_features = state.get("reference_features", [])
        retriever_history = state.get("retriever_history", [])

        # executor에서 설정한 현재 실행 중인 subtask ID 사용
        latest_task_id = state.get("current_executing_subtask_id")
        logger.debug("Refiner - executor에서 전달받은 Subtask ID: %s", latest_task_id)

        # 현재 subtask의 결과가 retriever_history에 있는지 확인
        has_result = any(h.get("subtask_id") == latest_task_id for h in retriever_history)

        if not has_result:
            logger.debug("Refiner - Subtask %s 검색 결과 없음 - synthesizer로 이동", latest_task_id)
            return update_state(state, next="synthesizer")

        # retriever_history에서 현재 subtask의 결과만 추출
        retriever_outputs = [h for h in retriever_history if h.get("subtask_id") == latest_task_id]
        logger.debug("Refiner - retriever_history에서 추출한 결과: %d개", len(retriever_outputs))

        try:
            retriever_results = self._collect_retriever_results(retriever_outputs)

            if llm is None:
                # LLM 없음 — 기본 정제, 항상 성공으로 처리
                refined_output = self._default_refine(retriever_results)
                raw_extracted_features = self._extract_features_from_raw_results(retriever_outputs)
                logger.debug("LLM 없음 - 원본 데이터에서 추출된 features: %s", raw_extracted_features)
                refined_data = {
                    "refined_text": refined_output,
                    "subtask_answer": refined_output[:500] if len(refined_output) > 500 else refined_output,
                    "reference_features": raw_extracted_features,
                    "verdict": True,
                }
            else:
                refined_data = await self._llm_refine_with_verdict(retriever_results, state, llm)

            verdict = refined_data.get("verdict", True)
            reference_features_found = refined_data.get("reference_features", [])
            subtask_answer = refined_data.get("subtask_answer", "")
            refined_text = refined_data.get("refined_text", "")

            # subtask verdict 업데이트
            updated_subtasks = []
            for subtask in subtasks:
                subtask_copy = subtask.copy()
                if subtask_copy.get("id") == latest_task_id:
                    subtask_copy["verdict"] = verdict
                    subtask_copy["subtask_answer"] = subtask_answer
                    subtask_copy["refined_text"] = refined_text
                    subtask_copy["reference_features"] = reference_features_found
                    logger.debug("Subtask %s verdict=%s, features=%s", latest_task_id, verdict, reference_features_found)
                updated_subtasks.append(subtask_copy)

            # ⭐ current_step 증가 (원본 코드의 iteration_count += 1과 동일)
            current_step = state.get("current_step", 0) + 1
            logger.debug("Current Step 증가: %d", current_step)

            update_kwargs: Dict[str, Any] = {
                "subtasks": updated_subtasks,
                "retry_counts": retry_counts,
                "current_step": current_step,
                "next": "synthesizer",
            }

            # 재시도 로직
            if not verdict:
                current_retries = retry_counts.get(latest_task_id, 0)
                retry_counts[latest_task_id] = current_retries + 1
                logger.debug("Subtask %s 재시도 %d/3", latest_task_id, current_retries + 1)

                # 최대 재시도 초과 시 exceeded로 설정
                if current_retries + 1 >= 3:
                    for subtask in updated_subtasks:
                        if subtask.get("id") == latest_task_id:
                            subtask["verdict"] = "exceeded"
                            subtask["subtask_answer"] = "최대 재시도 횟수를 초과하여 요청을 중단합니다."
                            subtask["retry_reason"] = "최대 재시도 횟수(3회) 초과"
                            logger.debug("Subtask %s exceeded 설정", latest_task_id)

                    # exceeded subtask도 subtask_results에 추가
                    exceeded_subtask = next(
                        (s.copy() for s in updated_subtasks if s.get("id") == latest_task_id), None
                    )
                    if exceeded_subtask:
                        update_kwargs["subtask_results"] = [exceeded_subtask]
                        logger.debug("exceeded subtask 결과 추가됨")

            # 완료된 subtask를 subtask_results에 추가 (binding resolution용)
            if verdict:
                completed_subtask = next(
                    (s.copy() for s in updated_subtasks if s.get("id") == latest_task_id and s.get("verdict")),
                    None,
                )
                if completed_subtask:
                    update_kwargs["subtask_results"] = [completed_subtask]
                    logger.debug("subtask_results에 추가됨: Task %s, features=%s",
                                 latest_task_id, completed_subtask.get("reference_features", []))

                if reference_features_found:
                    update_kwargs["reference_features"] = reference_features + reference_features_found

            return update_state(state, **update_kwargs)

        except Exception as e:
            logger.error("Refiner error: %s\n%s", e, traceback.format_exc())
            # 재시도 로직
            current_retries = retry_counts.get("refiner", 0)
            if current_retries < 3:
                retry_counts["refiner"] = current_retries + 1
                return update_state(state, retry_counts=retry_counts, next="refiner")
            else:
                return update_state(state, next="synthesizer")

    def _collect_retriever_results(self, retriever_outputs: List[Dict]) -> List[Dict]:
        """Retriever 출력 결과 수집 — dict/list result 모두 처리"""
        all_results = []
        for output in retriever_outputs:
            result = output.get("result", {})
            subtask_id = output.get("subtask_id", "unknown")
            if isinstance(result, dict):
                results_list = result.get("results", [])
                if isinstance(results_list, str):
                    all_results.append({"text": results_list, "subtask_id": subtask_id})
                elif isinstance(results_list, list):
                    for item in results_list:
                        if isinstance(item, str):
                            all_results.append({"text": item, "subtask_id": subtask_id})
                        elif isinstance(item, dict) and "text" in item:
                            all_results.append({"text": item["text"], "subtask_id": subtask_id})
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, str):
                        all_results.append({"text": item, "subtask_id": subtask_id})
                    elif isinstance(item, dict) and "text" in item:
                        all_results.append({"text": item["text"], "subtask_id": subtask_id})
        return all_results

    def _default_refine(self, results: List[Dict]) -> str:
        """기본 정제 함수 (LLM 없이 fallback)"""
        if not results:
            return "No results to refine"
        parts = [r.get("text", "") for r in results if r.get("text")]
        return "\n\n".join(parts) if parts else "No results to refine"

    def _extract_features_from_raw_results(self, retriever_outputs: List[Dict]) -> List[Dict[str, str]]:
        """
        원본 검색 결과에서 feature_id와 feature_name 직접 추출

        변환되지 않은 원본 데이터 사용 → 정보 손실 방지
        """
        features = []
        seen_ids: set = set()

        for output in retriever_outputs:
            result = output.get("result", {})
            results_list = result.get("results", []) if isinstance(result, dict) else (result if isinstance(result, list) else [])
            if not isinstance(results_list, list):
                continue
            for item in results_list:
                if not isinstance(item, str):
                    continue
                feature_id_matches = re.findall(r'FGR-[A-Z]{2}\d{4}', item)
                feature_name_matches = re.findall(r'"feature_name":\s*"([^"]+)"', item)
                for i, feature_id in enumerate(feature_id_matches):
                    if feature_id not in seen_ids:
                        feature_name = feature_name_matches[i].strip() if i < len(feature_name_matches) else ""
                        features.append({"feature_id": feature_id, "feature_name": feature_name})
                        seen_ids.add(feature_id)
                        logger.debug("원본 데이터에서 feature 추출: %s - %s", feature_id, feature_name)
        return features

    def _extract_features_from_results(self, results: List[Dict]) -> List[Dict[str, str]]:
        """변환된 결과 텍스트에서 feature_id와 feature_name 추출 (fallback)"""
        features = []
        seen_ids: set = set()

        for result in results:
            text = result.get("text", "")
            if not text:
                continue
            feature_id_matches = re.findall(r'FGR-[A-Z]{2}\d{4}', text)
            feature_name_matches = re.findall(r'"feature_name":\s*"([^"]+)"', text)
            feature_name_matches += re.findall(r'Feature\s*Name:\s*([^\n]+)', text, re.IGNORECASE)
            for i, feature_id in enumerate(feature_id_matches):
                if feature_id not in seen_ids:
                    feature_name = feature_name_matches[i].strip() if i < len(feature_name_matches) else ""
                    features.append({"feature_id": feature_id, "feature_name": feature_name})
                    seen_ids.add(feature_id)
        return features

    async def _llm_refine_with_verdict(
        self, results: List[Dict], state: AgentState, llm: BaseLanguageModel
    ) -> dict:
        """LLM을 사용한 결과 정제 및 verdict 결정"""
        latest_task_id = state.get("current_executing_subtask_id", 0)
        subtasks = state.get("subtasks", [])
        current_subtask = next((s for s in subtasks if s.get("id") == latest_task_id), None)

        # retriever_history에서 현재 subtask의 결과 추출
        retriever_history = state.get("retriever_history", [])
        retriever_outputs = [h for h in retriever_history if h.get("subtask_id") == latest_task_id]
        raw_extracted_features = self._extract_features_from_raw_results(retriever_outputs)
        logger.debug("원본 데이터에서 추출된 features: %s", raw_extracted_features)

        results_text = self._format_results_for_refiner(results)
        user_content = (
            f"사용자 질문: {state.get('user_query', '')}\n\n"
            f"현재 Subtask: {json.dumps(current_subtask, ensure_ascii=False, indent=2) if current_subtask else 'N/A'}\n\n"
            f"검색 결과:\n{results_text}\n\n"
            f"위 검색 결과를 분석하고, subtask의 목표를 달성했는지 판단하여 JSON 형식으로 반환해주세요.\n"
            f'reference_features 필드에 검색 결과에서 사용된 모든 feature의 feature_id와 feature_name을 반드시 포함하세요.\n'
            f'형식: {{"feature_id": "FGR-XXXX", "feature_name": "..."}}'
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            response = await llm.ainvoke(messages)
            # ainvoke는 str 반환, 혹은 AIMessage — 둘 다 처리
            content = response if isinstance(response, str) else getattr(response, "content", "{}")
            logger.debug("Refiner LLM 응답: %s", content[:500] if len(content) > 500 else content)

            try:
                result_dict = json.loads(content)
                refined_data = {
                    "refined_text": result_dict.get("refined_text", ""),
                    "subtask_answer": result_dict.get("subtask_answer", ""),
                    "reference_features": result_dict.get("reference_features", []),
                    "verdict": result_dict.get("verdict", True),
                    "retry_reason": result_dict.get("retry_reason", ""),
                }
                # reference_features 우선순위: 원본 추출 > LLM 추출 > 변환 결과 fallback
                if raw_extracted_features:
                    refined_data["reference_features"] = raw_extracted_features
                    logger.debug("원본 데이터에서 추출된 features 사용: %s", raw_extracted_features)
                elif not refined_data["reference_features"]:
                    extracted = self._extract_features_from_results(results)
                    if extracted:
                        refined_data["reference_features"] = extracted
                        logger.debug("Fallback으로 추출된 features: %s", extracted)
                return refined_data
            except json.JSONDecodeError:
                logger.debug("JSON 파싱 실패, 기본값 + 원본 추출 사용")
                features = raw_extracted_features or self._extract_features_from_results(results)
                return {
                    "refined_text": content,
                    "subtask_answer": content,
                    "reference_features": features,
                    "verdict": True,
                }
        except Exception as e:
            logger.error("LLM refine error: %s\n%s", e, traceback.format_exc())
            features = raw_extracted_features or self._extract_features_from_results(results)
            return {
                "refined_text": str(e),
                "subtask_answer": "",
                "reference_features": features,
                "verdict": False,
            }

    def _format_results_for_refiner(self, results: List[Dict]) -> str:
        """Refiner용 결과 포맷팅"""
        parts = []
        for i, result in enumerate(results, 1):
            text = result.get("text", "")
            if text:
                parts.append(f"[Document {i}/{len(results)}]")
                parts.append(text)
                parts.append("")
        return "\n".join(parts) if parts else "No results to refine"


refiner_node = RefinerNode()
