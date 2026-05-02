from __future__ import annotations

from typing import Any, TypedDict

from spar.router.schemas import RouteResult


class SparState(TypedDict, total=False):
    # input
    query: str
    product: str | None
    release: str | None
    top_k: int
    request_id: str
    history: list[dict[str, str]]

    # preprocess
    expanded_query: str
    history_context: str
    matched_terms: list[str]

    # query rewriting
    rewritten_query: str
    query_complexity: str

    # routing
    route_result: RouteResult

    # decomposition
    sub_questions: list[str]

    # retrieval
    raw_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]

    # generation
    answer: str

    # observability
    error: str | None
    node_trace: list[str]
    node_timings: dict[str, float]   # node_name -> execution time ms

    # performance eval inputs (populated by eval_suite; ignored in production)
    gold_chunks: list[str] | None
    gold_answer: str | None
    eval_metrics: dict[str, Any]

    # verify loop
    retry_count: int
    tried_strategies: list[str]
    verify_score: float | None
    verify_reason: str | None
    improved_query: str | None
