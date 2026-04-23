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
        execution_history = state.get("execution_history", {})
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
            sid = r.get("subtask_id", r.get("id", "?"))
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
        total_exec = len(execution_history)
        ok_exec = sum(
            1 for execs in execution_history.values()
            for e in execs if e.get("status") == "success"
        )
        logger.info("  execution_history: %d tasks, %d successful", total_exec, ok_exec)
        logger.info("=" * 60)
        # ─────────────────────────────────────────────────────────────────────

        retriever_outputs = retriever_outputs_raw
        retriever_outputs_text = ""

        # refiner 결과 우선 처리
        refiner_outputs = [o for o in retriever_outputs if isinstance(o, dict) and o.get("source") == "refiner"]
        if refiner_outputs:
            retriever_outputs_text += "=== 정제된 최종 결과 ===\n"
            for output in refiner_outputs:
                content = output.get("content", "")
                if content:
                    retriever_outputs_text += f"{content}\n\n"

        # 일반 retriever 결과 처리 (최대 10개)
        regular_outputs = [o for o in retriever_outputs if not (isinstance(o, dict) and o.get("source") == "refiner")]
        if regular_outputs:
            retriever_outputs_text += "=== 검색된 문서 정보 ===\n"
            processed_count = 0
            for output in regular_outputs:
                if processed_count >= 10:
                    break
                try:
                    if isinstance(output, dict):
                        subtask_id = output.get("subtask_id", "unknown")
                        if "result" in output:
                            result = output["result"]
                            if isinstance(result, list) and result:
                                item = result[0]
                                content = item.get("text", item.get("content", str(item))) if isinstance(item, dict) else str(item)
                            elif isinstance(result, dict):
                                content = result.get("text", result.get("content", str(result)))
                            else:
                                content = str(result)
                        elif "content" in output:
                            content = output["content"]
                        else:
                            continue
                        retriever_outputs_text += f"[Subtask {subtask_id}]\n내용: {content}\n\n"
                        processed_count += 1
                    elif isinstance(output, str):
                        try:
                            parsed = json.loads(output)
                            subtask_id = parsed.get("subtask_id", "unknown") if isinstance(parsed, dict) else "unknown"
                            content = parsed.get("content", parsed.get("text", str(parsed))) if isinstance(parsed, dict) else output
                        except Exception:
                            subtask_id, content = "unknown", output
                        retriever_outputs_text += f"[Subtask {subtask_id}]\n내용: {content}\n\n"
                        processed_count += 1
                except Exception as e:
                    logger.debug("Error processing retriever output: %s", e)
                    continue

        # 대화 히스토리 (최근 3개)
        history_text = ""
        history = state.get("history", [])
        if history:
            history_text = "\n=== 대화 히스토리 ===\n"
            for msg in history[-3:]:
                history_text += f"{msg['role']}: {msg['content']}\n"

        # 실행 히스토리 요약
        execution_summary = ""
        execution_history = state.get("execution_history", {})
        if execution_history:
            total_tasks = len(execution_history)
            successful_tasks = sum(
                1 for executions in execution_history.values()
                for execution in executions
                if execution.get("status") == "success"
            )
            execution_summary = f"\n=== 실행 히스토리 요약 ===\n총 작업 수: {total_tasks}, 성공한 작업 수: {successful_tasks}\n"

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

        response = await llm.ainvoke(messages)
        # ainvoke는 str 반환, 혹은 AIMessage — 둘 다 처리
        return response if isinstance(response, str) else getattr(response, "content", "")


synthesizer_node = SynthesizerNode()
