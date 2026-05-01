import pytest
from spar.router.schemas import Route, RouteResult
from spar.retrieval.routing import doc_types_for_route, build_expr


def _result(route, entities=None, product=None, release=None):
    return RouteResult(
        route=route, confidence=1.0, layer="test",
        entities=entities or {}, product=product, release=release,
    )


def test_structured_lookup_alarm():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP, entities={"alarm_code": "ALM-123"}))
    assert types == ["alarm_ref"]


def test_structured_lookup_param():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP, entities={"param_name": "maxTxPower"}))
    assert types == ["parameter_ref"]


def test_structured_lookup_mo():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP, entities={"mo_name": "NRCellDU"}))
    assert types == ["parameter_ref"]


def test_structured_lookup_no_entities_fallback():
    types = doc_types_for_route(_result(Route.STRUCTURED_LOOKUP))
    assert set(types) == {"parameter_ref", "counter_ref", "alarm_ref"}


def test_definition_explain():
    types = doc_types_for_route(_result(Route.DEFINITION_EXPLAIN))
    assert set(types) == {"feature_desc", "spec"}


def test_procedural():
    types = doc_types_for_route(_result(Route.PROCEDURAL))
    assert set(types) == {"mop", "install_guide"}


def test_diagnostic():
    types = doc_types_for_route(_result(Route.DIAGNOSTIC))
    assert set(types) == {"alarm_ref", "feature_desc"}


def test_comparative():
    types = doc_types_for_route(_result(Route.COMPARATIVE))
    assert set(types) == {"release_notes", "feature_desc"}


def test_default_rag():
    types = doc_types_for_route(_result(Route.DEFAULT_RAG))
    assert "feature_desc" in types


def test_build_expr_alarm():
    expr = build_expr(_result(Route.STRUCTURED_LOOKUP, entities={"alarm_code": "ALM-123"}))
    assert expr == 'mo_name == "ALM-123"'


def test_build_expr_param():
    expr = build_expr(_result(Route.STRUCTURED_LOOKUP, entities={"param_name": "maxTxPower"}))
    assert expr is None


def test_build_expr_product_filter():
    expr = build_expr(_result(Route.DEFAULT_RAG, product="NR"))
    assert expr == 'product == "NR"'


def test_build_expr_product_and_release():
    expr = build_expr(_result(Route.DEFAULT_RAG, product="LTE", release="v6.0"))
    assert 'product == "LTE"' in expr
    assert 'release == "v6.0"' in expr


def test_build_expr_none_when_no_filters():
    expr = build_expr(_result(Route.DEFAULT_RAG))
    assert expr is None
