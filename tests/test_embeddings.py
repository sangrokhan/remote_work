import pytest
from causality_graph.store.embeddings import EmbeddingStore


@pytest.fixture
def store(tmp_path):
    return EmbeddingStore(persist_dir=str(tmp_path / "chroma"))


def test_upsert_and_search(store):
    store.upsert_node(
        node_id="feature:CA",
        text="Carrier Aggregation combines multiple frequency bands to increase throughput",
        metadata={"node_type": "feature", "gen": "both"}
    )
    store.upsert_node(
        node_id="kpi:dl_throughput",
        text="Downlink throughput measures data rate in the downlink direction in Mbps",
        metadata={"node_type": "kpi"}
    )
    results = store.search("how to increase data rate", top_k=2)
    ids = [r["id"] for r in results]
    assert len(results) == 2
    assert all("id" in r and "score" in r for r in results)


def test_upsert_is_idempotent(store):
    store.upsert_node("feature:CA", "Carrier Aggregation", {"node_type": "feature"})
    store.upsert_node("feature:CA", "Carrier Aggregation updated", {"node_type": "feature"})
    results = store.search("carrier aggregation", top_k=5)
    ids = [r["id"] for r in results]
    assert ids.count("feature:CA") == 1


def test_search_returns_empty_on_no_data(store):
    results = store.search("anything", top_k=3)
    assert results == []
