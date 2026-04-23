# src/tools/registry.py

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """기본 툴 클래스"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    async def ainvoke(self, input_data: Dict[str, Any]) -> Any:
        """툴 실행"""
        pass


class RetrieverTool(BaseTool):
    """Retriever 툴 클래스 """

    def __init__(self):
        super().__init__(
            name="retriever",
            description="Retrieves algorithmic details and verifies your results from the features' documents."
        )
        self._original_retriever = None

    def _get_original_retriever(self):
        if self._original_retriever is None:
            try:
                from tools.retriever import RetrieverTool as OriginalRetrieverTool
                self._original_retriever = OriginalRetrieverTool
            except ImportError as e:
                logger.warning("RetrieverTool 로드 실패 (closed-system 모듈 없음): %s", e)
                self._original_retriever = False  # 재시도 방지
        return self._original_retriever if self._original_retriever is not False else None

    async def ainvoke(self, input_data: Dict[str, Any]) -> Any:
        """
        Retriever 툴 실행 -  Milvus + BGE3 + Reranker 검색 수행
        
        Args:
            input_data: 입력 데이터 (query, top_k, feature_ids 등)
            
        Returns:
            검색 결과 리스트
        """
        try:
            query = input_data.get("query", "")
            top_k = input_data.get("top_k", 5)

            logger.info(f"   [Retriever] Executing with query: {query[:99]}...")
            logger.info(f"   [Retriever] top_k: {top_k}")

            original_retriever = self._get_original_retriever()
            if original_retriever is None:
                return {"query": query, "results": [], "status": "unavailable"}

            # 동기 함수를 비동기로 실행
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: original_retriever(
                    query=query,
                    top_k=top_k
                )
            )

            # 결과가 리스트인 경우 첫 번째 요소 반환 (원본이 [[...]] 형태로 반환)
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], list):
                # 검색 결과 텍스트들을 하나의 리스트로 변환
                flattened = []
                for chunk_list in result:
                    if isinstance(chunk_list, list):
                        flattened.extend(chunk_list)
                    else:
                        flattened.append(chunk_list)

                logger.info(f"   [Retriever] Retrieved {len(flattened)} chunks from Milvus")
                return {
                    "query": query,
                    "results": flattened,
                    "status": "success"
                }

            # 결과가 dict 형태인 경우 그대로 반환
            if isinstance(result, dict):
                return result

            # 기타 경우
            return {
                "query": query,
                "results": result if isinstance(result, list) else [str(result)],
                "status": "success"
            }

        except Exception as e:
            error_msg = f"Error executing retriever tool: {str(e)}"
            logger.error("❌ %s", error_msg, exc_info=True)
            return {"error": error_msg, "results": [], "status": "failed"}


class ToolRegistry:
    """툴 레지스트리"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """기본 툴 등록"""
        self.register_tool(RetrieverTool())

    def register_tool(self, tool: BaseTool) -> None:
        """툴 등록"""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """툴 조회"""
        return self._tools.get(name)

    def get(self, name: str) -> Optional[BaseTool]:
        """툴 조회 (executor_node 호환 alias)"""
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, BaseTool]:
        """등록된 모든 툴 목록"""
        return self._tools.copy()
