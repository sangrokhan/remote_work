from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spar.pipeline.state import SparState
from spar.preprocessing.abbrev_mapper import (
    build_reverse_index,
    expand_query,
    load_acronyms,
)
from spar.reranker.client import CrossEncoderClient
from spar.router.hybrid_router import HybridRouter

_ACRONYMS_PATH = Path(__file__).parent.parent.parent.parent.parent / "dictionary" / "acronyms.json"


def _load_acronyms() -> tuple[dict, dict[str, str]]:
    if _ACRONYMS_PATH.exists():
        acronyms = load_acronyms(_ACRONYMS_PATH)
        return acronyms, build_reverse_index(acronyms)
    return {}, {}


def _append_trace(state: SparState, node: str) -> list[str]:
    return [*state.get("node_trace", []), node]  # type: ignore[arg-type]


@dataclass
class Nodes:
    router: HybridRouter
    reranker: CrossEncoderClient
    _acronyms: dict
    _reverse_index: dict[str, str]

    @classmethod
    def create(
        cls,
        router: HybridRouter,
        reranker: CrossEncoderClient,
        acronyms_path: Path | None = None,
    ) -> Nodes:
        path = acronyms_path or _ACRONYMS_PATH
        if path.exists():
            acronyms = load_acronyms(path)
            reverse_index = build_reverse_index(acronyms)
        else:
            acronyms, reverse_index = {}, {}
        return cls(router=router, reranker=reranker, _acronyms=acronyms, _reverse_index=reverse_index)

    async def preprocess(self, state: SparState) -> SparState:
        query = state["query"]
        expanded = expand_query(query, self._acronyms, self._reverse_index)
        return {**state, "expanded_query": expanded, "node_trace": _append_trace(state, "preprocess")}

    async def route(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        result = await self.router.route(query)
        return {**state, "route_result": result, "node_trace": _append_trace(state, "route")}

    async def rag_retrieve(self, state: SparState) -> SparState:
        # Stub — replace with MilvusClient.hybrid_search() when Task 1.4 wired to API
        query = state.get("expanded_query") or state["query"]
        chunks: list[dict[str, Any]] = [
            {"chunk_id": "stub-001", "score": 0.95, "text": f"[stub] chunk for: {query}"}
        ]
        return {**state, "raw_chunks": chunks, "node_trace": _append_trace(state, "rag_retrieve")}

    async def structured_retrieve(self, state: SparState) -> SparState:
        # Phase 3: KG/DB lookup — fallback to RAG until implemented
        result = await self.rag_retrieve(state)
        return {**result, "node_trace": _append_trace(result, "structured_retrieve")}

    async def multi_hop_retrieve(self, state: SparState) -> SparState:
        # Phase 5: iterative retrieval via LangGraph Send — fallback to RAG
        result = await self.rag_retrieve(state)
        return {**result, "node_trace": _append_trace(result, "multi_hop_retrieve")}

    async def rerank(self, state: SparState) -> SparState:
        chunks = state.get("raw_chunks", [])
        if not chunks:
            return {**state, "reranked_chunks": [], "node_trace": _append_trace(state, "rerank")}
        query = state.get("expanded_query") or state["query"]
        scores = await self.reranker.rerank(query, [c["text"] for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return {
            **state,
            "reranked_chunks": [c for c, _ in ranked],
            "node_trace": _append_trace(state, "rerank"),
        }

    async def generate(self, state: SparState) -> SparState:
        # Stub — replace with LLMClient.chat() when generation module ready
        chunks = state.get("reranked_chunks") or state.get("raw_chunks", [])
        answer = f"[stub] Answer for '{state['query']}' based on {len(chunks)} chunks."
        return {**state, "answer": answer, "node_trace": _append_trace(state, "generate")}
