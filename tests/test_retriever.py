import pytest
from causality_graph.schema import (
    KPINode, FeatureNode, ParameterNode, Edge,
    Direction, Generation, EdgeType, Magnitude
)
from causality_graph.store.graph import CausalityGraph
from causality_graph.store.embeddings import EmbeddingStore
from causality_graph.agent.retriever import Retriever


@pytest.fixture
def populated_stores(tmp_path):
    graph = CausalityGraph()
    embeddings = EmbeddingStore(persist_dir=str(tmp_path / "chroma"))

    kpi = KPINode(id="kpi:dl_throughput", name="DL Throughput", unit="Mbps",
                  good_direction=Direction.POSITIVE,
                  description="Downlink data rate in Mbps")
    feature = FeatureNode(id="feature:CA", name="Carrier Aggregation",
                          gen=Generation.BOTH, category="rrm",
                          description="Combines carriers to increase downlink throughput")
    param = ParameterNode(id="param:maxCaBands", name="maxCaBands",
                          data_type="int", range_min=1, range_max=4,
                          default_value="1",
                          description="Max number of component carriers to aggregate")

    graph.add_kpi(kpi)
    graph.add_feature(feature)
    graph.add_parameter(param)
    graph.add_edge(Edge(from_id="feature:CA", to_id="kpi:dl_throughput",
                        relation=EdgeType.AFFECTS, direction=Direction.POSITIVE,
                        magnitude=Magnitude.HIGH, validated=True))
    graph.add_edge(Edge(from_id="feature:CA", to_id="param:maxCaBands",
                        relation=EdgeType.CONTROLLED_BY, validated=True))

    embeddings.upsert_node("kpi:dl_throughput", kpi.name + " " + kpi.description,
                           {"node_type": "kpi"})
    embeddings.upsert_node("feature:CA", feature.name + " " + feature.description,
                           {"node_type": "feature"})
    embeddings.upsert_node("param:maxCaBands", param.name + " " + param.description,
                           {"node_type": "parameter"})

    return graph, embeddings


def test_retriever_returns_subgraph(populated_stores):
    graph, embeddings = populated_stores
    retriever = Retriever(graph=graph, embeddings=embeddings)
    subgraph = retriever.retrieve("how to improve downlink throughput", top_k=2, hops=1)
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert "kpi:dl_throughput" in node_ids or "feature:CA" in node_ids


def test_subgraph_includes_edges(populated_stores):
    graph, embeddings = populated_stores
    retriever = Retriever(graph=graph, embeddings=embeddings)
    subgraph = retriever.retrieve("downlink throughput carrier aggregation", top_k=3, hops=1)
    assert len(subgraph["edges"]) > 0


def test_subgraph_structure(populated_stores):
    graph, embeddings = populated_stores
    retriever = Retriever(graph=graph, embeddings=embeddings)
    subgraph = retriever.retrieve("maxCaBands parameter", top_k=2, hops=1)
    assert "nodes" in subgraph
    assert "edges" in subgraph
    assert all("id" in n for n in subgraph["nodes"])


def test_gen_filter(populated_stores):
    graph, embeddings = populated_stores
    retriever = Retriever(graph=graph, embeddings=embeddings)
    subgraph = retriever.retrieve("throughput", top_k=3, hops=1, gen_filter="5G")
    # feature:CA has gen=both, should still appear
    node_ids = {n["id"] for n in subgraph["nodes"]}
    assert "feature:CA" in node_ids
