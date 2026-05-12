from pathlib import Path
from causality_graph.fixtures import make_sample_graph
from causality_graph.store.graph import CausalityGraph


def test_add_and_get_node():
    g = make_sample_graph()
    assert g.get_node("kpi:dl_throughput") is not None
    assert g.get_node("feature:CA") is not None


def test_get_neighbors():
    g = make_sample_graph()
    neighbors = g.get_neighbors("feature:CA")
    ids = [n["id"] for n in neighbors]
    assert "kpi:dl_throughput" in ids
    assert "param:maxCaBands" in ids


def test_get_edges_from():
    g = make_sample_graph()
    edges = g.get_edges_from("feature:CA")
    assert len(edges) == 2
    relations = {e["relation"] for e in edges}
    assert "AFFECTS" in relations
    assert "CONTROLLED_BY" in relations


def test_unknown_node_returns_none():
    g = CausalityGraph()
    assert g.get_node("nonexistent:x") is None


def test_serialize_deserialize(tmp_path):
    g = make_sample_graph()
    path = tmp_path / "graph.json"
    g.save(path)
    g2 = CausalityGraph.load(path)
    assert g2.get_node("feature:CA") is not None
    edges = g2.get_edges_from("feature:CA")
    assert len(edges) == 2


def test_node_count():
    g = make_sample_graph()
    assert g.node_count() == 12  # 3 KPIs + 4 features + 5 params


def test_edge_count():
    g = make_sample_graph()
    assert g.edge_count() == 10
