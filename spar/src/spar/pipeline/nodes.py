from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spar.encoder.base import EncoderClient
from spar.pipeline.state import SparState
from spar.preprocessing.abbrev_mapper import (
    build_reverse_index,
    expand_query,
    load_acronyms,
)
from spar.reranker.client import CrossEncoderClient
from spar.retrieval.milvus_client import SparMilvusClient
from spar.retrieval.query_decomposer import QueryDecomposer
from spar.retrieval.query_rewriter import build_context
from spar.retrieval.routing import build_expr, doc_types_for_route
from spar.router.hybrid_router import HybridRouter

_ACRONYMS_PATH = Path(__file__).parent.parent.parent.parent.parent / "dictionary" / "acronyms.json"


def _append_trace(state: SparState, node: str) -> list[str]:
    return [*state.get("node_trace", []), node]  # type: ignore[arg-type]


@dataclass
class Nodes:
    router: HybridRouter
    reranker: CrossEncoderClient
    encoder: EncoderClient
    milvus: SparMilvusClient
    _acronyms: dict
    _reverse_index: dict[str, str]
    _decomposer: QueryDecomposer = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._decomposer is None:
            self._decomposer = QueryDecomposer()

    @classmethod
    def create(
        cls,
        router: HybridRouter,
        reranker: CrossEncoderClient,
        encoder: EncoderClient,
        milvus: SparMilvusClient,
        acronyms_path: Path | None = None,
    ) -> "Nodes":
        path = acronyms_path or _ACRONYMS_PATH
        if path.exists():
            acronyms = load_acronyms(path)
            reverse_index = build_reverse_index(acronyms)
        else:
            acronyms, reverse_index = {}, {}
        return cls(
            router=router,
            reranker=reranker,
            encoder=encoder,
            milvus=milvus,
            _acronyms=acronyms,
            _reverse_index=reverse_index,
        )

    async def preprocess(self, state: SparState) -> SparState:
        query = state["query"]
        expanded = expand_query(query, self._acronyms, self._reverse_index)
        return {**state, "expanded_query": expanded, "node_trace": _append_trace(state, "preprocess")}

    async def prepare_context(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        history = state.get("history", [])
        ctx = build_context(query, history, self._acronyms)
        return {**state, "history_context": ctx, "node_trace": _append_trace(state, "prepare_context")}

    async def route(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        result = await self.router.route(query)
        return {**state, "route_result": result, "node_trace": _append_trace(state, "route")}

    async def decompose(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        sub_questions = await self._decomposer.decompose(query)
        return {**state, "sub_questions": sub_questions, "node_trace": _append_trace(state, "decompose")}

    async def decomposed_retrieve(self, state: SparState) -> SparState:
        sub_questions = state.get("sub_questions") or [state.get("expanded_query") or state["query"]]
        route_result = state["route_result"]
        top_k = state.get("top_k", 10)
        doc_types = doc_types_for_route(route_result)
        expr = build_expr(route_result)

        seen: set[str] = set()
        merged: list[dict[str, Any]] = []

        async def _retrieve_one(sq: str) -> list[dict[str, Any]]:
            vec: list[float] = self.encoder.encode([sq])[0].tolist()
            return await self._hybrid_search_multi(doc_types, sq, vec, top_k, expr)

        per_question = await asyncio.gather(*[_retrieve_one(sq) for sq in sub_questions])
        for chunks in per_question:
            for chunk in chunks:
                key = chunk.get("id") or chunk.get("text", "")[:120]
                if key not in seen:
                    seen.add(key)
                    merged.append(chunk)

        merged.sort(key=lambda c: c["score"], reverse=True)
        return {**state, "raw_chunks": merged[:top_k * 2], "node_trace": _append_trace(state, "decomposed_retrieve")}

    async def _hybrid_search_multi(
        self,
        doc_types: list[str],
        query_text: str,
        query_vector: list[float],
        top_k: int,
        expr: str | None,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()

        async def _search_one(doc_type: str) -> list[dict[str, Any]]:
            return await loop.run_in_executor(
                None,
                lambda: self.milvus.hybrid_search(
                    doc_type=doc_type,
                    query_text=query_text,
                    query_vector=query_vector,
                    top_k=top_k,
                    expr=expr,
                ),
            )

        results = await asyncio.gather(*[_search_one(dt) for dt in doc_types])
        merged = [chunk for chunks in results for chunk in chunks]
        merged.sort(key=lambda c: c["score"], reverse=True)
        return merged[:top_k]

    async def rag_retrieve(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        route_result = state["route_result"]
        top_k = state.get("top_k", 10)

        query_vector: list[float] = self.encoder.encode([query])[0].tolist()
        doc_types = doc_types_for_route(route_result)
        expr = build_expr(route_result)

        chunks = await self._hybrid_search_multi(doc_types, query, query_vector, top_k, expr)
        return {**state, "raw_chunks": chunks, "node_trace": _append_trace(state, "rag_retrieve")}

    async def structured_retrieve(self, state: SparState) -> SparState:
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
        reranked = [{"score": s, **c} for c, s in ranked]
        return {**state, "reranked_chunks": reranked, "node_trace": _append_trace(state, "rerank")}

    async def generate(self, state: SparState) -> SparState:
        chunks = state.get("reranked_chunks") or state.get("raw_chunks", [])
        context = "\n\n".join(c["text"] for c in chunks[:5])
        query = state["query"]
        answer = f"[stub] context={len(chunks)} chunks\nquery={query}\n{context[:200]}"
        return {**state, "answer": answer, "node_trace": _append_trace(state, "generate")}
