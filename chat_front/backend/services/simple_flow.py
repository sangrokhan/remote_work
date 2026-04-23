"""
Simple (non-agentic) flow service.

Function-calling RAG loop: user input → LLM(+tools) → [retriever?] → final answer.
LLM decides whether to call retriever. Used when agentic_rag=false.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from llm.factory import get_llm
from tools.registry import RetrieverTool

if TYPE_CHECKING:
    from app.models import RunWorkflowRequest

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """당신은 5G 통신 장비 매뉴얼 전문가입니다.
사용자의 질문에 답변하기 위해 필요한 경우 retriever 도구를 사용하여 기술 문서를 검색하세요.

## 도구 사용 가이드라인
1. 기술적인 질문(파라미터, 기능 설명 등)에는 retriever를 사용하세요
2. 일반적인 인사나 대화에는 도구 없이 직접 답변하세요
3. 검색 결과를 바탕으로 정확하고 구체적인 답변을 제공하세요
4. 검색 결과에 없는 정보는 "제공된 문서에서 찾을 수 없습니다"라고 명시하세요

## 답변 형식
- 명확하고 구조화된 답변을 제공하세요
- 가능한 경우 구체적인 파라미터 값과 설명을 포함하세요
- 관련된 추가 정보도 함께 제공하세요"""

TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "retriever",
            "description": """Retrieves algorithmic details and parameter descriptions from the features' documents.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to perform in *English*. This should be semantically close to your target documents. Use an affirmative statement rather than a question."
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents from the highest ranking ones to be considered. If the user wants more detailed information, increase 'top_k' to 15 or 20; otherwise, decrease 'top_k' to 5.",
                        "default": 10
                    }
                },
                "required": ["query"],
            },
        },
    }
]


class SimpleService:
    def __init__(self) -> None:
        self._retriever = RetrieverTool()

    async def process(self, req: RunWorkflowRequest) -> AsyncGenerator[dict, None]:
        llm = get_llm(req.model)
        llm_with_tools = llm._llm.bind_tools(TOOLS_DEFINITION)

        messages: list = [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=req.input),
        ]

        logger.debug("SimpleService.process: model=%s input_len=%d", req.model, len(req.input))

        MAX_LOOPS = 5
        final_payload: dict = {}
        retrieved_docs: list[str] = []
        loop = 0

        while loop < MAX_LOOPS:
            loop += 1
            yield {
                "event": "node_started",
                "node": "llm",
                "name": "llm",
                "stage": "start",
                "message": f"LLM 호출 중 (루프 {loop})",
            }

            response: AIMessage = await llm_with_tools.ainvoke(messages)
            logger.info("loop:%d tool_calls:%s", loop, response.tool_calls)

            if not response.tool_calls or loop == MAX_LOOPS:
                final_payload = {"final_output": response.content}
                yield {
                    "event": "node_finished",
                    "node": "llm",
                    "name": "llm",
                    "stage": "end",
                    "message": response.content,
                    "payload": final_payload,
                }
                break

            messages.append(response)

            for tool_call in response.tool_calls:
                tool_name: str = tool_call["name"]
                tool_args: dict = tool_call["args"]
                tool_call_id: str = tool_call["id"]
                query: str = tool_args.get("query", req.input)

                yield {
                    "event": "node_started",
                    "node": "retriever",
                    "name": "retriever",
                    "stage": "start",
                    "message": f"검색 중: {query}",
                }

                if tool_name == "retriever":
                    result = await self._retriever.ainvoke(
                        {"query": query, "top_k": tool_args.get("top_k", 5)}
                    )
                    docs: list[str] = result.get("results", [])
                    retrieved_docs.extend(docs)
                    context_text = "\n\n".join(docs) if docs else "검색 결과가 없습니다."
                else:
                    context_text = "알 수 없는 툴입니다."

                yield {
                    "event": "node_finished",
                    "node": "retriever",
                    "name": "retriever",
                    "stage": "end",
                    "message": f"{len(retrieved_docs)}개 문서 검색됨",
                }

                messages.append(
                    ToolMessage(
                        content=f"[검색된 문서]\n{context_text}",
                        tool_call_id=tool_call_id,
                    )
                )

        logger.debug("SimpleService.process complete: final_payload keys=%s", list(final_payload.keys()))

        yield {
            "event": "workflow_complete",
            "message": "완료",
            "final_response": final_payload.get("final_output", ""),
            "steps": retrieved_docs,
            "reference_features": [],
        }
