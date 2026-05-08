import pytest
from pathlib import Path
from causality_graph.extraction.review_queue import ReviewQueue, ReviewStatus
from causality_graph.extraction.committer import commit_approved
from causality_graph.store.graph import CausalityGraph
from causality_graph.store.db import MetadataDB
from causality_graph.store.embeddings import EmbeddingStore


@pytest.fixture
def stores(tmp_path):
    graph_path = tmp_path / "graph.json"
    graph = CausalityGraph()
    db = MetadataDB(tmp_path / "meta.db")
    embeddings = EmbeddingStore(persist_dir=str(tmp_path / "chroma"))
    return graph, db, embeddings, graph_path


@pytest.fixture
def queue_with_approved(tmp_path):
    q = ReviewQueue(tmp_path / "queue.jsonl")
    triple = {
        "from": "feature:CA",
        "to": "kpi:dl_throughput",
        "relation": "AFFECTS",
        "direction": "+",
        "magnitude": "high",
        "condition": "",
        "confidence": 0.92,
    }
    item_id = q.enqueue(triple)
    q.approve(item_id)
    return q


def test_commit_adds_edge_to_graph(stores, queue_with_approved):
    graph, db, embeddings, graph_path = stores
    committed = commit_approved(queue_with_approved, graph, db, embeddings, graph_path)
    assert committed == 1
    edges = graph.get_edges_from("feature:CA")
    assert len(edges) > 0


def test_commit_adds_edge_to_db(stores, queue_with_approved):
    graph, db, embeddings, graph_path = stores
    commit_approved(queue_with_approved, graph, db, embeddings, graph_path)
    edges = db.filter_edges(validated=True)
    assert len(edges) == 1


def test_commit_saves_graph(stores, queue_with_approved, tmp_path):
    graph, db, embeddings, graph_path = stores
    commit_approved(queue_with_approved, graph, db, embeddings, graph_path)
    assert graph_path.exists()
    g2 = CausalityGraph.load(graph_path)
    assert g2.edge_count() == 1


def test_commit_skips_pending(stores, tmp_path):
    graph, db, embeddings, graph_path = stores
    q = ReviewQueue(tmp_path / "queue2.jsonl")
    q.enqueue({"from": "feature:CA", "to": "kpi:dl_throughput", "relation": "AFFECTS",
                "direction": "+", "magnitude": "high", "condition": "", "confidence": 0.8})
    committed = commit_approved(q, graph, db, embeddings, graph_path)
    assert committed == 0
