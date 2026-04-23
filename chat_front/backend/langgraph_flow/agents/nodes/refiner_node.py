# langgraph_agenticrag/src/agents/nodes/refiner_node.py

import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.prompts.refiner import REFINER_PROMPT_TEMPLATE


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
        if config and "llm" in config:
            llm = config["llm"]
        elif config and hasattr(config,
                                'configurable') and config.configurable and "llm" in config.configurable:
            llm = config.configurable["llm"]
        elif isinstance(config, dict) and "configurable" in config and "llm" in config[
            "configurable"]:
            llm = config["configurable"]["llm"]

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
        latest_task_id = state.get("current_executing_subtask_id")
        print(f"=== DEBUG: Refiner - Executor에서 전달받은 Subtask ID: {latest_task_id} ===")

        # 현재 subtask의 결과가 retriever_history에 있는지 확인
        has_result = any(
            h.get("subtask_id") == latest_task_id
            for h in retriever_history
        )

        if not has_result:
            print(f"=== DEBUG: Refiner - Subtask {latest_task_id}의 검색 결과 없음 - synthesizer로 이동 ===")
            return update_state(state, next="synthesizer")

        # retriever_history에서 현재 subtask의 결과만 추출
        retriever_outputs = [
            h for h in retriever_history
            if h.get("subtask_id") == latest_task_id
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
                if subtask_copy.get("id") == latest_task_id:
                    subtask_copy["verdict"] = verdict
                    subtask_copy["subtask_answer"] = subtask_answer
                    subtask_copy["refined_text"] = refined_text
                    subtask_copy["reference_features"] = reference_features_found
                    print(f"=== DEBUG: Subtask {latest_task_id} verdict 설정: {verdict} ===")
                    print(
                        f"=== DEBUG: Subtask {latest_task_id} reference_features: {reference_features_found} ===")
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

            # 재시도 로직 - 원본 agent_service.py와 동일한 로직
            if not verdict:
                # 재시도 횟수 증가
                current_retries = retry_counts.get(latest_task_id, 0)
                retry_counts[latest_task_id] = current_retries + 1
                print(f"=== DEBUG: Subtask {latest_task_id} 재시도 {current_retries + 1}/3 ===")

                # 최대 재시도 초과 시 exceeded로 설정 (원본과 동일)
                if current_retries + 1 >= 3:
                    for subtask in updated_subtasks:
                        if subtask.get("id") == latest_task_id:
                            subtask["verdict"] = "exceeded"
                            subtask["subtask_answer"] = "최대 재시도 횟수를 초과하여 요청을 중단합니다."
                            subtask["retry_reason"] = "최대 재시도 횟수(3회) 초과"
                            print(
                                f"=== DEBUG: Subtask {latest_task_id} 최대 재시도 초과 - exceeded 설정 ===")

                    # exceeded된 subtask도 subtask_results에 추가 (원본과 동일)
                    exceeded_subtask = None
                    for subtask in updated_subtasks:
                        if subtask.get("id") == latest_task_id:
                            exceeded_subtask = subtask.copy()
                            break
                    if exceeded_subtask:
                        update_kwargs["subtask_results"] = [exceeded_subtask]
                        print(f"=== DEBUG: exceeded subtask 결과 추가됨 ===")

            # 완료된 subtask가 있으면 subtask_results에 추가 (binding resolution을 위해)
            if verdict:
                completed_subtask = None
                for subtask in updated_subtasks:
                    if subtask.get("id") == latest_task_id and subtask.get("verdict"):
                        completed_subtask = subtask.copy()
                        break

                if completed_subtask:
                    # subtask_results에 완료된 subtask 추가 (binding resolution에서 사용)
                    update_kwargs["subtask_results"] = [completed_subtask]
                    print(f"=== DEBUG: subtask_results에 추가됨: Task {latest_task_id} ===")
                    print(
                        f"=== DEBUG: subtask_results 내용: {completed_subtask.get('reference_features', [])} ===")

                # reference_features 업데이트
                if reference_features_found:
                    update_kwargs[
                        "reference_features"] = reference_features + reference_features_found

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
        latest_task_id = state.get("current_executing_subtask_id", 0)
        subtasks = state.get("subtasks", [])
        current_subtask = None
        for subtask in subtasks:
            if subtask.get("id") == latest_task_id:
                current_subtask = subtask
                break

        # ⭐ retriever_history에서 현재 subtask의 결과 추출
        retriever_history = state.get("retriever_history", [])
        retriever_outputs = [
            h for h in retriever_history
            if h.get("subtask_id") == latest_task_id
        ]
        raw_extracted_features = self._extract_features_from_raw_results(retriever_outputs)
        print(f"=== DEBUG: 원본 데이터에서 추출된 features: {raw_extracted_features} ===")

        # 검색 결과 텍스트 포맷팅
        results_text = self._format_results_for_refiner(results)

        # subtask context 구성
        subtask_context = {
            "current_subtask": current_subtask,
            "user_query": state.get("user_query", "")
        }

        user_content = f"""사용자 질문: {state.get('user_query', '')}

현재 Subtask: {json.dumps(current_subtask, ensure_ascii=False, indent=2) if current_subtask else 'N/A'}

검색 결과:
{results_text}

위 검색 결과를 분석하고, subtask의 목표를 달성했는지 판단하여 JSON 형식으로 반환해주세요.
reference_features 필드에 검색 결과에서 사용된 모든 feature의 feature_id와 feature_name을 반드시 포함하세요.

형식: {{"feature_id": "FGR-XXXX", "feature_name": "..."}}"""

        # LLM 호출
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ]

        try:
            response = await llm.ainvoke(messages)
            content = response.content or "{}"

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
