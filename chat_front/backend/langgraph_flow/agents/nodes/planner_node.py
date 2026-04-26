# langgraph_agenticrag/src/agents/nodes/planner_node.py

import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.language_models import BaseLanguageModel

from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.prompts.planner import PLANNER_SYSTEM_PROMPT
from langgraph_flow.prompts.var_constructor import CONSTRUCTOR_SYSTEM_PROMPT

# 디버그를 위한 전역 카운터
planner_call_count = 0
binding_call_count = 0


class PlannerNode:
    """Planner 노드 클래스 - Agentic RAG 구조로 개선"""

    def __init__(self):
        self.name = "planner"
        self.system_prompt = PLANNER_SYSTEM_PROMPT

    async def invoke(self, state: AgentState,
                     config: Optional[RunnableConfig] = None) -> AgentState:
        """
        Planner 노드 실행
        
        Args:
            state: 현재 에이전트 상태
            config: 설정 정보 (선택적)
            
        Returns:
            업데이트된 상태
        """
        # LLM 추출 로직 개선 - RunnableConfig 또는 dict 모두 처리
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

        # 상태에 next 필드가 없으면 기본값 설정 - planner에서 executor로 이동
        if "next" not in state or state["next"] is None:
            state["next"] = "executor"

        return await self.plan_next_step(state, llm)

    # DUPLICATED CODE — var_constructor_node.construct_binding_context already extracts this
    # via the same CONSTRUCTOR_SYSTEM_PROMPT and stores it in state["binding_context"].
    # Method body kept intact; invocation in plan_next_step is disabled.
    async def _extract_binding_context_with_llm(self, state: AgentState,
                                                llm: BaseLanguageModel) -> dict:
        """
        LLM을 사용하여 binding context 추출
        """
        global binding_call_count
        binding_call_count += 1
        print(f"=== DEBUG: _extract_binding_context_with_llm 호출 #{binding_call_count} ===")
        print(f"User Query: {state.get('user_query', '')}")
        print(f"Current Step: {state.get('current_step', 0)}")

        binding_extraction_prompt = CONSTRUCTOR_SYSTEM_PROMPT

        messages = [
            SystemMessage(content=binding_extraction_prompt),
            HumanMessage(content=f"User Query: {state.get('user_query', '')}")
        ]

        try:
            response = await llm.bind(temperature=0.1).ainvoke(messages)
            content = response.content or "{}"
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.rsplit("```", 1)[0].strip()
            if not content:
                content = "{}"
            binding_context = json.loads(content)

            # 기본 구조 보장
            if "query_entities" not in binding_context:
                binding_context["query_entities"] = {"features": [], "keywords": []}
            if "previous_features" not in binding_context:
                binding_context["previous_features"] = []
            if "explicit_dependencies" not in binding_context:
                binding_context["explicit_dependencies"] = []

            print(f"Binding Context 결과: {binding_context}")
            return binding_context
        except Exception as e:
            print(f"Binding context extraction failed: {e}")
            return {
                "query_entities": {"features": [], "keywords": []},
                "previous_features": [],
                "explicit_dependencies": []
            }

    async def plan_next_step(self, state: AgentState,
                             llm: Optional[BaseLanguageModel] = None) -> AgentState:
        """
        다음 단계를 계획하는 함수 - 원본 플로우 기반으로 한 번만 subtasks 생성
        
        Args:
            state: 현재 에이전트 상태
            llm: 언어 모델 (선택적, 실제 구현에서는 주입됨)
            
        Returns:
            업데이트된 상태
        """
        global planner_call_count
        planner_call_count += 1

        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", 10)
        is_finished = state.get("is_finished", False)

        print(f"=== DEBUG: plan_next_step 호출 #{planner_call_count} ===")
        print(f"Current Step: {current_step}, Max Steps: {max_steps}")
        print(f"Is Finished: {is_finished}")
        print(f"User Query: {state.get('user_query', '')}")

        # 종료 조건 체크
        if is_finished or current_step >= max_steps:
            print(f"=== DEBUG: 종료 조건 충족 - synthesizer로 이동 ===")
            state_copy = state.copy()
            state_copy.update({
                "is_finished": True,
                "next": "synthesizer"
            })
            return state_copy

        # 이미 subtasks가 있으면 새로 생성하지 않고 executor로 이동
        # 원본 agentic rag 방식: subtasks 필드 확인
        existing_subtasks = state.get("subtasks", [])
        if existing_subtasks:
            print(f"=== DEBUG: 이미 Subtasks 존재 ({len(existing_subtasks)}개) - executor로 이동 ===")
            state_copy = state.copy()
            state_copy.update({
                "next": "executor"
            })
            return state_copy

        # steps 업데이트 로직 적용
        next_step = current_step + 1
        print(f"Next Step will be: {next_step}")

        # LLM이 제공되지 않은 경우 (테스트용) 기본 응답 반환
        if llm is None:
            print("=== DEBUG: LLM 없음 - 테스트 응답 반환 ===")
            # 테스트용 기본 응답 - 원본 agentic rag 방식: subtasks에 저장
            default_subtasks = [{
                "id": 0,  # 원본처럼 0부터 시작
                "goal": f"Query processing: {state.get('user_query', 'No query')}",
                "task_type": "THINK",
                "verdict": False,
                "dependencies": [],
                "bindings": {}
            }]

            state_copy = state.copy()
            state_copy.update({
                "subtasks": default_subtasks,  # 원본 방식: subtasks에 저장
                "current_step": state.get("current_step", 0) + 1,
                "next": "executor"  # executor로 이동
            })
            return state_copy

        try:
            print("=== DEBUG: Binding context 추출 시작 ===")
            # Reuse binding_context from var_constructor_node (already extracted upstream).
            binding_context = state.get("binding_context", {})
            # DUPLICATED: var_constructor_node already extracts this. Kept commented for reference.
            # binding_context = await self._extract_binding_context_with_llm(state, llm)
            print("=== DEBUG: Binding context 추출 완료 ===")

            # Enhanced system prompt with binding context
            enhanced_prompt = self.system_prompt + f"\n\n# Current Binding Context\n{json.dumps(binding_context, ensure_ascii=False, indent=2)}"

            # 사용자 쿼리와 컨텍스트 구성
            user_context = ""
            if state.get("history"):
                user_context += "\n이전 대화:\n"
                for msg in state["history"][-3:]:  # 최근 3개 대화만
                    user_context += f"{msg['role']}: {msg['content']}\n"

            user_query = f"""사용자 질문: {state.get('user_query', '')}
{user_context}
현재 단계: {state.get('current_step', 0) + 1}/{state.get('max_steps', 10)}"""

            print("=== DEBUG: Planner LLM 호출 시작 ===")
            # LLM 호출
            messages = [
                SystemMessage(content=enhanced_prompt),
                HumanMessage(content=user_query)
            ]

            response = await llm.bind(temperature=0.2).ainvoke(messages)
            print("=== DEBUG: Planner LLM 호출 완료 ===")

            # 응답 파싱 + 검증/정규화 (id 강제, dep cleaning, binding-derived deps, DAG 보장)
            subtasks = self._parse_planner_response(response.content)
            subtasks = self._validate_subtasks(subtasks)
            print(f"=== DEBUG: 생성된 Subtasks: {subtasks} ===")

            # 상태 업데이트 - 원본 agentic rag 방식: subtasks에 저장
            state_copy = state.copy()
            state_copy.update({
                "subtasks": subtasks,  # 원본 방식: subtasks에 저장 (불변)
                "current_step": state.get("current_step", 0) + 1,
                "next": "executor",  # executor로 이동
                "user_query": state.get("user_query", "")  # user_query 유지
            })

            print(f"=== DEBUG: Planner 완료 - 다음 단계: executor ===")
            print(f"=== DEBUG: user_query 유지 확인: '{state_copy.get('user_query', '')}' ===")
            return state_copy

        except Exception as e:
            # 에러 발생 시 synthesizer로 이동
            print(f"Planner error: {e}")
            state_copy = state.copy()
            state_copy.update({
                "next": "synthesizer"
            })
            return state_copy

    def _parse_planner_response(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Planner 응답 파싱 - Agentic RAG 구조에 맞게 개선
        
        Args:
            response_content: LLM 응답 내용
            
        Returns:
            파싱된 subtasks 리스트
        """
        print(f"=== DEBUG: _parse_planner_response 입력: {response_content[:200]}... ===")

        try:
            # JSON 파싱 시도
            content = response_content or "{}"
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```", 2)[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.rsplit("```", 1)[0].strip()
            if not content:
                content = "{}"
            response_dict = json.loads(content)
            if isinstance(response_dict, dict) and "subtasks" in response_dict:
                subtasks = response_dict["subtasks"]
                # subtasks 구조 검증 및 정규화
                normalized_subtasks = []
                for i, subtask in enumerate(subtasks):
                    normalized_subtask = {
                        "id": subtask.get("subtask_id", subtask.get("id", i)),
                        "goal": subtask.get("goal", subtask.get("description", str(subtask))),
                        "task_type": subtask.get("task_type", "THINK"),
                        "verdict": subtask.get("verdict", False),
                        "dependencies": subtask.get("dependencies", []),
                        "bindings": subtask.get("bindings", {})
                    }
                    normalized_subtasks.append(normalized_subtask)
                print(f"=== DEBUG: 파싱된 subtasks (dict): {len(normalized_subtasks)} ===")
                return normalized_subtasks
            elif isinstance(response_dict, list):
                # 리스트 형식인 경우 구조 정규화
                normalized_subtasks = []
                for i, subtask in enumerate(response_dict):
                    if isinstance(subtask, dict):
                        normalized_subtask = {
                            "id": subtask.get("subtask_id", subtask.get("id", i)),
                            "goal": subtask.get("goal", subtask.get("description", str(subtask))),
                            "task_type": subtask.get("task_type", "THINK"),
                            "verdict": subtask.get("verdict", False),
                            "dependencies": subtask.get("dependencies", []),
                            "bindings": subtask.get("bindings", {})
                        }
                    else:
                        normalized_subtask = {
                            "id": i,
                            "goal": str(subtask),
                            "task_type": "THINK",
                            "verdict": False,
                            "dependencies": [],
                            "bindings": {}
                        }
                    normalized_subtasks.append(normalized_subtask)
                print(f"=== DEBUG: 파싱된 subtasks (list): {len(normalized_subtasks)} ===")
                return normalized_subtasks
            else:
                # 기본 형식으로 변환
                result = [{
                    "id": 0,
                    "goal": str(response_dict),
                    "task_type": "THINK",
                    "verdict": False,
                    "dependencies": [],
                    "bindings": {}
                }]
                print(f"=== DEBUG: 파싱된 subtasks (기본): {len(result)} ===")
                return result
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트로 처리
            result = [{
                "id": 0,
                "goal": response_content,
                "task_type": "THINK",
                "verdict": False,
                "dependencies": [],
                "bindings": {}
            }]
            print(f"=== DEBUG: 파싱된 subtasks (텍스트): {len(result)} ===")
            return result

    def _validate_subtasks(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Subtask 정규화 + 검증 패스.
        - id를 위치(0-base 연속)로 강제 재할당, LLM 원본 id는 remap에 보존
        - dependencies: int 강제, unknown / self / forward (>= self.id) 엣지 제거
        - bindings의 $task_N / $subtask_N 참조에서 누락된 dep 자동 보강
        - "dep < self.id" 규칙으로 DAG 자동 보장 (순환 감지 불필요)
        """
        if not subtasks:
            return subtasks

        # 1. id를 위치로 강제. 원본 id → 새 id 매핑
        id_remap: Dict[Any, int] = {}
        for i, s in enumerate(subtasks):
            old_id = s.get("id")
            if old_id is not None and old_id not in id_remap:
                id_remap[old_id] = i
            s["id"] = i

        valid_ids = set(range(len(subtasks)))
        binding_pat = re.compile(r"\$(?:task|subtask)_(\d+)")

        for i, s in enumerate(subtasks):
            # 2. explicit dependencies cleaning
            deps_raw = s.get("dependencies", []) or []
            cleaned: List[int] = []
            for d in deps_raw:
                try:
                    d_int = int(d)
                except (TypeError, ValueError):
                    print(f"=== WARN: subtask {i} dep '{d}' not int — dropped ===")
                    continue
                d_new = id_remap.get(d_int, d_int)
                if d_new not in valid_ids:
                    print(f"=== WARN: subtask {i} dep {d_int} unknown — dropped ===")
                    continue
                if d_new == i:
                    print(f"=== WARN: subtask {i} self-ref dep — dropped ===")
                    continue
                if d_new > i:
                    print(f"=== WARN: subtask {i} forward dep {d_new} — dropped ===")
                    continue
                if d_new not in cleaned:
                    cleaned.append(d_new)

            # 3. bindings에서 dep 보강
            bindings = s.get("bindings", {}) or {}
            for ref in bindings.values():
                if not isinstance(ref, str):
                    continue
                for m in binding_pat.finditer(ref):
                    d_int = int(m.group(1))
                    d_new = id_remap.get(d_int, d_int)
                    if d_new in valid_ids and d_new < i and d_new not in cleaned:
                        cleaned.append(d_new)
                        print(f"=== INFO: subtask {i} dep {d_new} inferred from binding ===")

            s["dependencies"] = cleaned

        return subtasks


# 노드 인스턴스 생성
planner_node = PlannerNode()
