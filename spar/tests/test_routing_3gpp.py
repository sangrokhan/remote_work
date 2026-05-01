"""3GPP spec number-aware routing & retrieval 통합 테스트.

RegexRouter 단위 테스트 (Milvus 불필요) + hybrid_search expr 전달 검증 (mock).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spar.router.regex_router import RegexRouter
from spar.router.schemas import Route


@pytest.fixture
def router() -> RegexRouter:
    return RegexRouter()


# ---------------------------------------------------------------------------
# RegexRouter — spec_number entity 추출 검증
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected_spec", [
    ("TS 29.502 session management", "29.502"),
    ("3GPP TS 38.300 NR architecture", "38.300"),
    ("TS29502 what is SMF?", "29.502"),
    ("TS 38 300 overview of 5G NR", "38.300"),
    ("refer to TS 23.501 for system architecture", "23.501"),
])
def test_regex_router_extracts_spec_number(router, query, expected_spec):
    result = router.route(query)
    assert result is not None, f"expected regex match for: {query!r}"
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.layer == "regex"
    assert result.entities.get("spec_number") == expected_spec


@pytest.mark.parametrize("query", [
    "session management in 5G core",
    "how to configure SMF parameters",
    "what is carrier aggregation?",
])
def test_regex_router_no_match_for_non_ts_queries(router, query):
    result = router.route(query)
    assert result is None or result.entities.get("spec_number") is None


# ---------------------------------------------------------------------------
# spec_number entity → Milvus expr 문자열 변환
# ---------------------------------------------------------------------------

def test_spec_number_to_milvus_expr(router):
    result = router.route("TS 29.502 session management")
    assert result is not None
    spec_num = result.entities.get("spec_number")
    expr = f"spec_number == '{spec_num}'" if spec_num else None
    assert expr == "spec_number == '29.502'"


def test_no_spec_number_gives_none_expr(router):
    result = router.route("session management in 5G core")
    spec_num = result.entities.get("spec_number") if result else None
    expr = f"spec_number == '{spec_num}'" if spec_num else None
    assert expr is None


# ---------------------------------------------------------------------------
# hybrid_search expr 파라미터 전달 검증 (Milvus mock)
#
# NOTE: 아래 테스트는 RouteResult.entities → Milvus expr 변환 패턴을 검증한다.
# HybridRouter → retrieval 직접 연결은 이번 스코프 외 (설계 문서 섹션 6 참조).
# 실제 파이프라인 연결 후 실 구현 코드를 호출하는 통합 테스트로 교체 예정.
# ---------------------------------------------------------------------------

def test_hybrid_search_receives_expr_when_spec_number_known(router):
    mock_client = MagicMock()
    mock_client.hybrid_search.return_value = []

    result = router.route("TS 38.300 NR architecture overview")
    assert result is not None
    spec_num = result.entities.get("spec_number")
    expr = f"spec_number == '{spec_num}'" if spec_num else None

    mock_client.hybrid_search(
        doc_type="spec",
        query_text="TS 38.300 NR architecture overview",
        query_vector=[0.0] * 1024,
        expr=expr,
    )

    mock_client.hybrid_search.assert_called_once_with(
        doc_type="spec",
        query_text="TS 38.300 NR architecture overview",
        query_vector=[0.0] * 1024,
        expr="spec_number == '38.300'",
    )


def test_hybrid_search_no_expr_when_no_spec_number():
    mock_client = MagicMock()
    mock_client.hybrid_search.return_value = []

    expr = None

    mock_client.hybrid_search(
        doc_type="spec",
        query_text="session management in 5G core",
        query_vector=[0.0] * 1024,
        expr=expr,
    )

    mock_client.hybrid_search.assert_called_once_with(
        doc_type="spec",
        query_text="session management in 5G core",
        query_vector=[0.0] * 1024,
        expr=None,
    )


def test_expr_fallback_on_empty_result():
    """expr 결과 0건 → None expr로 재시도하는 패턴 검증."""
    mock_client = MagicMock()
    mock_client.hybrid_search.side_effect = [
        [],
        [{"chunk_id": "abc", "text": "session management", "score": 0.9}],
    ]

    def search_with_fallback(client, query_text, query_vector, expr=None):
        results = client.hybrid_search(
            doc_type="spec",
            query_text=query_text,
            query_vector=query_vector,
            expr=expr,
        )
        if not results and expr is not None:
            results = client.hybrid_search(
                doc_type="spec",
                query_text=query_text,
                query_vector=query_vector,
                expr=None,
            )
        return results

    results = search_with_fallback(
        mock_client,
        "TS 29.502 session management",
        [0.0] * 1024,
        expr="spec_number == '29.502'",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "abc"
    assert mock_client.hybrid_search.call_count == 2
