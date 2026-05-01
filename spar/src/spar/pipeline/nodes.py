from __future__ import annotations

import asyncio
import time
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
from spar.retrieval.query_rewriter import build_context
from spar.retrieval.routing import build_expr, doc_types_for_route
from spar.router.hybrid_router import HybridRouter

_ACRONYMS_PATH = Path(__file__).parent.parent.parent.parent.parent / "dictionary" / "acronyms.json"


def _append_trace(state: SparState, node: str) -> list[str]:
    return [*state.get("node_trace", []), node]  # type: ignore[arg-type]


def _record_timing(state: SparState, node: str, elapsed_ms: float) -> dict[str, float]:
    timings = dict(state.get("node_timings") or {})
    timings[node] = elapsed_ms
    return timings


@dataclass
class Nodes:
    router: HybridRouter
    reranker: CrossEncoderClient
    encoder: EncoderClient
    milvus: SparMilvusClient
    _acronyms: dict
    _reverse_index: dict[str, str]

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
        t0 = time.monotonic()
        query = state["query"]
        expanded = expand_query(query, self._acronyms, self._reverse_index)
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "expanded_query": expanded,
            "node_trace": _append_trace(state, "preprocess"),
            "node_timings": _record_timing(state, "preprocess", elapsed),
        }

    async def prepare_context(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        query = state.get("expanded_query") or state["query"]
        history = state.get("history", [])
        ctx = build_context(query, history, self._acronyms)
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "history_context": ctx,
            "node_trace": _append_trace(state, "prepare_context"),
            "node_timings": _record_timing(state, "prepare_context", elapsed),
        }

    async def route(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        query = state.get("expanded_query") or state["query"]
        result = await self.router.route(query)
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "route_result": result,
            "node_trace": _append_trace(state, "route"),
            "node_timings": _record_timing(state, "route", elapsed),
        }

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
        t0 = time.monotonic()
        query = state.get("expanded_query") or state["query"]
        route_result = state["route_result"]
        top_k = state.get("top_k", 10)

        query_vector: list[float] = self.encoder.encode([query])[0].tolist()
        doc_types = doc_types_for_route(route_result)
        expr = build_expr(route_result)

        chunks = await self._hybrid_search_multi(doc_types, query, query_vector, top_k, expr)
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "raw_chunks": chunks,
            "node_trace": _append_trace(state, "rag_retrieve"),
            "node_timings": _record_timing(state, "rag_retrieve", elapsed),
        }

    async def structured_retrieve(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        result = await self.rag_retrieve(state)
        # strip inner rag_retrieve timing — this node's key replaces it
        timings = {k: v for k, v in (result.get("node_timings") or {}).items() if k != "rag_retrieve"}
        elapsed = (time.monotonic() - t0) * 1000
        timings["structured_retrieve"] = elapsed
        return {
            **result,
            "node_trace": _append_trace(result, "structured_retrieve"),
            "node_timings": timings,
        }

    async def multi_hop_retrieve(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        # Phase 5: iterative retrieval via LangGraph Send — fallback to RAG
        result = await self.rag_retrieve(state)
        # strip inner rag_retrieve timing — this node's key replaces it
        timings = {k: v for k, v in (result.get("node_timings") or {}).items() if k != "rag_retrieve"}
        elapsed = (time.monotonic() - t0) * 1000
        timings["multi_hop_retrieve"] = elapsed
        return {
            **result,
            "node_trace": _append_trace(result, "multi_hop_retrieve"),
            "node_timings": timings,
        }

    async def rerank(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        chunks = state.get("raw_chunks", [])
        if not chunks:
            elapsed = (time.monotonic() - t0) * 1000
            return {
                **state,
                "reranked_chunks": [],
                "node_trace": _append_trace(state, "rerank"),
                "node_timings": _record_timing(state, "rerank", elapsed),
            }
        query = state.get("expanded_query") or state["query"]
        scores = await self.reranker.rerank(query, [c["text"] for c in chunks])
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        reranked = [{"score": s, **c} for c, s in ranked]
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "reranked_chunks": reranked,
            "node_trace": _append_trace(state, "rerank"),
            "node_timings": _record_timing(state, "rerank", elapsed),
        }

    async def generate(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        chunks = state.get("reranked_chunks") or state.get("raw_chunks", [])
        context = "\n\n".join(c["text"] for c in chunks[:5])
        query = state["query"]
        answer = f"[stub] context={len(chunks)} chunks\nquery={query}\n{context[:200]}"
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "answer": answer,
            "node_trace": _append_trace(state, "generate"),
            "node_timings": _record_timing(state, "generate", elapsed),
        }
