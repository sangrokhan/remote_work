from spar.router.schemas import Route, RouteResult


def test_route_values():
    assert Route.STRUCTURED_LOOKUP.value == "structured_lookup"
    assert Route.DEFAULT_RAG.value == "default_rag"


def test_route_result_defaults():
    r = RouteResult(route=Route.DEFAULT_RAG, confidence=0.5, layer="regex")
    assert r.entities == {}
    assert r.product is None
    assert r.release is None
