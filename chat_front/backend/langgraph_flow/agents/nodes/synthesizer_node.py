"""
Synthesizer node — terminal node. 수집된 정보를 종합해 최종 답변 생성.
항상 END로 이동.
"""
from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph_flow.agents.state import AgentState, update_state
from langgraph_flow.prompts.synthesizer import SYNTHESIZER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class SynthesizerNode:
    """Synthesizer 노드 클래스"""

    def __init__(self):
        self.name = "synthesizer"

    async def invoke(self, state: AgentState, config: Optional[RunnableConfig] = None) -> AgentState:
        # executor_node와 동일한 패턴으로 LLM 추출
        configurable = (config or {}).get("configurable", {})
        llm: Optional[BaseLanguageModel] = configurable.get("llm")
        return await self.synthesize_final_response(state, llm)

    async def synthesize_final_response(
        self, state: AgentState, llm: Optional[BaseLanguageModel] = None
    ) -> AgentState:
        """최종 응답을 생성하는 함수"""
        if llm is None:
            raise ValueError("LLM is required for final response generation")

        try:
            final_response = await self._generate_llm_response(state, llm)
            return update_state(state, final_response=final_response, is_finished=True, next="end")

        except Exception as e:
            logger.error("Synthesizer error: %s\n%s", e, traceback.format_exc())
            error_response = f"죄송합니다. 응답을 생성하는 중에 문제가 발생했습니다. 다시 시도해 주세요. (오류: {e})"
            return update_state(state, final_response=error_response, is_finished=True, next="end")

    def _generate_default_response(self, state: AgentState) -> str:
        """기본 최종 응답 생성 (LLM 없이 fallback용)"""
        retriever_outputs = state.get("retriever_outputs", [])

        if not retriever_outputs:
            return "죄송합니다. 요청하신 정보를 찾을 수 없습니다."

        response_parts = []

        # refiner 결과 우선 처리
        refiner_outputs = [o for o in retriever_outputs if isinstance(o, dict) and o.get("source") == "refiner"]
        if refiner_outputs:
            response_parts.append("=== 최종 정제된 결과 ===")
            for output in refiner_outputs:
                content = output.get("content", "")
                if content:
                    response_parts.append(str(content))

        # 일반 retriever 결과 처리 (상위 3개)
        regular_outputs = [o for o in retriever_outputs if not (isinstance(o, dict) and o.get("source") == "refiner")]
        if regular_outputs:
            response_parts.append("=== 검색된 정보 ===")
            for i, output in enumerate(regular_outputs[:3], 1):
                if isinstance(output, dict):
                    if "result" in output:
                        result = output["result"]
                        if isinstance(result, list) and result and isinstance(result[0], dict):
                            response_parts.append(f"[정보 {i}] {result[0].get('text', result[0].get('content', str(result[0])))}")
                        elif isinstance(result, dict):
                            response_parts.append(f"[정보 {i}] {result.get('text', result.get('content', str(result)))}")
                        else:
                            response_parts.append(f"[정보 {i}] {str(result)}")
                    elif "content" in output:
                        response_parts.append(f"[정보 {i}] {output['content']}")
                elif isinstance(output, str):
                    response_parts.append(f"[정보 {i}] {output}")

        return "\n\n".join(response_parts) if response_parts else "죄송합니다. 관련 정보를 처리할 수 없습니다."

    async def _generate_llm_response(self, state: AgentState, llm: BaseLanguageModel) -> str:
        """LLM을 사용한 최종 응답 생성 - Agentic RAG 구조"""
        # ── pre-synthesizer state snapshot ───────────────────────────────────
        subtasks = state.get("subtasks", [])
        subtask_results = state.get("subtask_results", [])
        retriever_history = state.get("retriever_history", [])
        resolved_bindings = state.get("resolved_bindings", {})
        retriever_outputs_raw = state.get("retriever_outputs", [])

        logger.info("=" * 60)
        logger.info("[SYNTHESIZER] pre-synthesis state snapshot")
        logger.info("  user_query     : %s", state.get("user_query", ""))
        logger.info("  subtasks       : %d total", len(subtasks))
        for i, t in enumerate(subtasks):
            logger.info("    [%d] id=%s action=%s", i, t.get("id"), t.get("action", t.get("tool", "?")))
        logger.info("  subtask_results: %d items", len(subtask_results))
        for i, r in enumerate(subtask_results):
            status = r.get("status", "?")
            sid = r.get("id", "?")
            content_preview = str(r.get("content", r.get("result", "")))[:120]
            logger.info("    [%d] subtask_id=%s status=%s content_preview=%s", i, sid, status, content_preview)
        logger.info("  retriever_outputs: %d items", len(retriever_outputs_raw))
        for i, o in enumerate(retriever_outputs_raw):
            if isinstance(o, dict):
                src = o.get("source", "?")
                sid = o.get("subtask_id", "?")
                preview = str(o.get("content", o.get("result", "")))[:120]
                logger.info("    [%d] source=%s subtask_id=%s preview=%s", i, src, sid, preview)
            else:
                logger.info("    [%d] %s", i, str(o)[:120])
        logger.info("  retriever_history: %d queries", len(retriever_history))
        for i, h in enumerate(retriever_history):
            logger.info("    [%d] subtask_id=%s query=%s", i, h.get("subtask_id"), str(h.get("query", ""))[:80])
        logger.info("  resolved_bindings: %s", list(resolved_bindings.keys()))
        verdict_ok = sum(1 for s in subtasks if s.get("verdict") is True)
        verdict_exceeded = sum(1 for s in subtasks if s.get("verdict") == "exceeded")
        verdict_pending = len(subtasks) - verdict_ok - verdict_exceeded
        logger.info(
            "  subtask verdict: %d total, %d ok, %d exceeded, %d pending",
            len(subtasks), verdict_ok, verdict_exceeded, verdict_pending,
        )
        logger.info("=" * 60)
        # ─────────────────────────────────────────────────────────────────────

        retriever_outputs_text = ""

        # 1순위: subtask_results의 refined 답변 (refiner가 저장하는 주 결과)
        # id별 verdict=True 중 attempt 최대값만 (재시도 성공 시 최신만)
        if subtask_results:
            latest_per_id: Dict[Any, Dict] = {}
            for r in subtask_results:
                if r.get("verdict") is not True:
                    continue
                sid = r.get("id")
                if sid is None:
                    continue
                if sid not in latest_per_id or r.get("attempt", 0) > latest_per_id[sid].get("attempt", 0):
                    latest_per_id[sid] = r
            if latest_per_id:
                retriever_outputs_text += "=== 정제된 최종 결과 ===\n"
                for sid in sorted(latest_per_id.keys(), key=lambda x: (x is None, x)):
                    result = latest_per_id[sid]
                    payload = result.get("result") or {}
                    answer = (
                        payload.get("subtask_answer")
                        or payload.get("refined_text")
                        or result.get("subtask_answer")  # backward compat
                        or result.get("refined_text", "")
                    )
                    if answer:
                        retriever_outputs_text += f"[Subtask {sid}]\n{answer}\n\n"

        # 2순위: subtasks에서 verdict=True인 항목의 subtask_answer (subtask_results에 없을 경우)
        if not retriever_outputs_text:
            completed_subtasks = [s for s in subtasks if s.get("verdict") is True]
            if completed_subtasks:
                retriever_outputs_text += "=== 정제된 최종 결과 ===\n"
                for s in completed_subtasks:
                    answer = s.get("subtask_answer") or s.get("refined_text", "")
                    if answer:
                        retriever_outputs_text += f"[Subtask {s.get('id', '?')}]\n{answer}\n\n"

        # 3순위: retriever_history 원본 (refiner를 거치지 않은 경우)
        if not retriever_outputs_text and retriever_history:
            retriever_outputs_text += "=== 검색된 문서 정보 ===\n"
            for h in retriever_history[:10]:
                result = h.get("result", {})
                if isinstance(result, dict):
                    items = result.get("results", [])
                    for item in (items if isinstance(items, list) else [])[:3]:
                        text = item.get("text", item.get("content", "")) if isinstance(item, dict) else str(item)
                        if text:
                            retriever_outputs_text += f"[Subtask {h.get('subtask_id', '?')}]\n{text}\n\n"
                            break

        # 대화 히스토리 (최근 3개)
        history_text = ""
        history = state.get("history", [])
        if history:
            history_text = "\n=== 대화 히스토리 ===\n"
            for msg in history[-3:]:
                history_text += f"{msg['role']}: {msg['content']}\n"

        # 실행 히스토리 요약 (verdict 기준 — retrieve 성공 ≠ 답 성공)
        execution_summary = ""
        if subtasks:
            ok_tasks = sum(1 for s in subtasks if s.get("verdict") is True)
            exceeded_tasks = sum(1 for s in subtasks if s.get("verdict") == "exceeded")
            execution_summary = (
                f"\n=== 실행 히스토리 요약 ===\n"
                f"총 작업 수: {len(subtasks)}, 성공: {ok_tasks}, 재시도 초과: {exceeded_tasks}\n"
            )

        user_query = (
            f"사용자 질문: {state.get('user_query', '')}\n\n"
            f"수집된 정보:\n{retriever_outputs_text}{history_text}{execution_summary}\n\n"
            f"위 정보들을 바탕으로 사용자 질문에 대한 최종 답변을 생성해주세요. "
            f"정제된 최종 결과를 중심으로 답변하고, 필요한 경우 검색된 문서 정보를 참고하여 상세한 답변을 제공해주세요."
        )

        messages = [
            SystemMessage(content=SYNTHESIZER_PROMPT_TEMPLATE),
            HumanMessage(content=user_query),
        ]

        response = await llm.bind(temperature=0.4).ainvoke(messages)
        # ainvoke는 str 반환, 혹은 AIMessage — 둘 다 처리
        return response if isinstance(response, str) else getattr(response, "content", "")


synthesizer_node = SynthesizerNode()
