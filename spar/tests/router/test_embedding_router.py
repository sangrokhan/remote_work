import pytest
from spar.router.embedding_router import EmbeddingRouter
from spar.router.schemas import Route


@pytest.fixture(scope="module")
def router():
    return EmbeddingRouter(threshold=0.5)


def test_procedural_query(router):
    result = router.route("How do I configure RACH parameters in NR?")
    assert result is not None
    assert result.route == Route.PROCEDURAL
    assert result.layer == "embedding"


def test_definition_query(router):
    result = router.route("What is Carrier Aggregation?")
    assert result is not None
    assert result.route == Route.DEFINITION_EXPLAIN


def test_low_confidence_returns_none(router):
    strict = EmbeddingRouter(threshold=0.99)
    result = strict.route("hello world something random xyz")
    assert result is None
