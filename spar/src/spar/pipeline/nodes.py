from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spar.encoder.base import EncoderClient
from spar.llm.client import LLMClient
from spar.pipeline.state import SparState
from spar.prompts import load_prompt
from spar.preprocessing.abbrev_mapper import (
    build_reverse_index,
    expand_query,
    extract_terms,
    load_acronyms,
    load_keywords,
)
from spar.reranker.client import CrossEncoderClient
from spar.retrieval.milvus_client import SparMilvusClient
from spar.retrieval.query_decomposer import QueryDecomposer
from spar.retrieval.query_rewriter import build_context, rewrite_query as _rewrite_query
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
    _keywords: set[str]
    _decomposer: QueryDecomposer = None  # type: ignore[assignment]
    llm: LLMClient | None = None

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
        llm: LLMClient | None = None,
    ) -> "Nodes":
        path = acronyms_path or _ACRONYMS_PATH
        if path.exists():
            acronyms = load_acronyms(path)
            reverse_index = build_reverse_index(acronyms)
            keywords = load_keywords(acronyms)
        else:
            acronyms, reverse_index = {}, {}
            keywords = set()
        return cls(
            router=router,
            reranker=reranker,
            encoder=encoder,
            milvus=milvus,
            _acronyms=acronyms,
            _reverse_index=reverse_index,
            _keywords=keywords,
            llm=llm,
        )

    async def preprocess(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        query = state["query"]
        expanded = expand_query(query, self._acronyms, self._reverse_index)
        matched = extract_terms(query, self._keywords)
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "expanded_query": expanded,
            "matched_terms": matched,
            "node_trace": _append_trace(state, "preprocess"),
            "node_timings": _record_timing(state, "preprocess", elapsed),
        }

    async def rewrite_query(self, state: SparState) -> SparState:
        query = state.get("expanded_query") or state["query"]
        history = state.get("history", [])
        result = await _rewrite_query(query, history, self._acronyms)
        return {
            **state,
            "rewritten_query": result.rewritten,
            "query_complexity": result.complexity,
            "node_trace": _append_trace(state, f"rewrite_query:{result.complexity}"),
        }

    async def prepare_context(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]
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
        query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]
        result = await self.router.route(query)
        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "route_result": result,
            "node_trace": _append_trace(state, "route"),
            "node_timings": _record_timing(state, "route", elapsed),
        }

    async def decompose(self, state: SparState) -> SparState:
        query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]
        sub_questions = await self._decomposer.decompose(query)
        return {**state, "sub_questions": sub_questions, "node_trace": _append_trace(state, "decompose")}

    async def decomposed_retrieve(self, state: SparState) -> SparState:
        sub_questions = state.get("sub_questions") or [state.get("expanded_query") or state["query"]]
        route_result = state["route_result"]
        top_k = state.get("top_k", 10)
        doc_types = doc_types_for_route(route_result)
        matched_terms = state.get("matched_terms", [])
        expr = build_expr(route_result, matched_terms=matched_terms)

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
        t0 = time.monotonic()
        query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]
        route_result = state["route_result"]
        top_k = state.get("top_k", 10)

        query_vector: list[float] = self.encoder.encode([query])[0].tolist()
        doc_types = doc_types_for_route(route_result)
        matched_terms = state.get("matched_terms", [])
        expr = build_expr(route_result, matched_terms=matched_terms)

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
        # strip inner delegation artifacts — timing and trace
        timings = {k: v for k, v in (result.get("node_timings") or {}).items() if k != "rag_retrieve"}
        trace = [n for n in (result.get("node_trace") or []) if n != "rag_retrieve"]
        elapsed = (time.monotonic() - t0) * 1000
        timings["structured_retrieve"] = elapsed
        return {
            **result,
            "node_trace": [*trace, "structured_retrieve"],
            "node_timings": timings,
        }

    async def multi_hop_retrieve(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        # Phase 5: iterative retrieval via LangGraph Send — fallback to RAG
        result = await self.rag_retrieve(state)
        # strip inner delegation artifacts — timing and trace
        timings = {k: v for k, v in (result.get("node_timings") or {}).items() if k != "rag_retrieve"}
        trace = [n for n in (result.get("node_trace") or []) if n != "rag_retrieve"]
        elapsed = (time.monotonic() - t0) * 1000
        timings["multi_hop_retrieve"] = elapsed
        return {
            **result,
            "node_trace": [*trace, "multi_hop_retrieve"],
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
        query = state.get("rewritten_query") or state.get("expanded_query") or state["query"]
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
        query = state["query"]

        if self.llm is None:
            answer = f"[stub] context={len(chunks)} chunks\nquery={query}"
        else:
            context = "\n\n".join(c["text"] for c in chunks[:5])
            history_ctx = state.get("history_context", "")
            user_content = f"{history_ctx}\n\nContext:\n{context}\n\nQuestion: {query}".strip()
            messages = [
                {"role": "system", "content": "You are a Samsung RAN expert. Answer using only the provided context."},
                {"role": "user", "content": user_content},
            ]
            answer = await self.llm.chat(messages)

        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "answer": answer,
            "node_trace": _append_trace(state, "generate"),
            "node_timings": _record_timing(state, "generate", elapsed),
        }

    async def verify(self, state: SparState) -> SparState:
        t0 = time.monotonic()
        if self.llm is None:
            elapsed = (time.monotonic() - t0) * 1000
            return {
                **state,
                "verify_score": 5.0,
                "verify_reason": "no llm — skipping verify",
                "node_trace": _append_trace(state, "verify"),
                "node_timings": _record_timing(state, "verify", elapsed),
            }

        query = state.get("improved_query") or state.get("rewritten_query") or state.get("expanded_query") or state["query"]
        answer = state.get("answer", "")
        chunks = state.get("reranked_chunks") or state.get("raw_chunks", [])
        contexts_summary = "\n---\n".join(c["text"][:300] for c in chunks[:5])

        prompt = load_prompt("verify.txt").format(
            query=query,
            answer=answer,
            contexts_summary=contexts_summary or "(no context)",
        )
        raw = await self.llm.chat([{"role": "user", "content": prompt}], max_tokens=128)

        try:
            parsed = json.loads(raw.strip())
            score = float(parsed["score"])
            reason = str(parsed.get("reason", ""))
        except Exception:
            score = 5.0
            reason = "parse error — treating as sufficient"

        elapsed = (time.monotonic() - t0) * 1000
        return {
            **state,
            "verify_score": score,
            "verify_reason": reason,
            "node_trace": _append_trace(state, "verify"),
            "node_timings": _record_timing(state, "verify", elapsed),
        }
