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
