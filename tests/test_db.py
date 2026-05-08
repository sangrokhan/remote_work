import pytest
from causality_graph.schema import (
    KPINode, FeatureNode, ParameterNode, Edge,
    Direction, Generation, EdgeType, Magnitude, NodeType
)
from causality_graph.store.db import MetadataDB


@pytest.fixture
def db(tmp_path):
    return MetadataDB(tmp_path / "meta.db")


def test_upsert_and_get_node(db):
    node = FeatureNode(id="feature:CA", name="Carrier Aggregation",
                       gen=Generation.BOTH, category="rrm")
    db.upsert_node(node, NodeType.FEATURE)
    row = db.get_node("feature:CA")
    assert row["name"] == "Carrier Aggregation"
    assert row["gen"] == "both"


def test_filter_nodes_by_type(db):
    db.upsert_node(KPINode(id="kpi:dl_throughput", name="DL Throughput",
                            unit="Mbps", good_direction=Direction.POSITIVE), NodeType.KPI)
    db.upsert_node(FeatureNode(id="feature:CA", name="CA",
                                gen=Generation.BOTH, category="rrm"), NodeType.FEATURE)
    kpis = db.filter_nodes(node_type=NodeType.KPI)
    assert len(kpis) == 1
    assert kpis[0]["id"] == "kpi:dl_throughput"


def test_filter_features_by_gen(db):
    db.upsert_node(FeatureNode(id="feature:CA", name="CA",
                                gen=Generation.BOTH, category="rrm"), NodeType.FEATURE)
    db.upsert_node(FeatureNode(id="feature:MIMO5G", name="MIMO 5G",
                                gen=Generation.G5, category="rrm"), NodeType.FEATURE)
    results = db.filter_nodes(node_type=NodeType.FEATURE, gen=Generation.G5)
    ids = [r["id"] for r in results]
    assert "feature:MIMO5G" in ids
    assert "feature:CA" not in ids


def test_upsert_and_filter_edges(db):
    edge = Edge(from_id="feature:CA", to_id="kpi:dl_throughput",
                relation=EdgeType.AFFECTS, direction=Direction.POSITIVE,
                magnitude=Magnitude.HIGH, confidence=0.92, validated=True)
    db.upsert_edge(edge)
    edges = db.filter_edges(validated=True)
    assert len(edges) == 1
    edges_low_conf = db.filter_edges(min_confidence=0.95)
    assert len(edges_low_conf) == 0


def test_upsert_is_idempotent(db):
    node = FeatureNode(id="feature:CA", name="CA", gen=Generation.BOTH, category="rrm")
    db.upsert_node(node, NodeType.FEATURE)
    db.upsert_node(node, NodeType.FEATURE)
    rows = db.filter_nodes(node_type=NodeType.FEATURE)
    assert len(rows) == 1
