from causality_graph.schema import (
    NodeType, EdgeType, Direction, Generation, Magnitude,
    KPINode, FeatureNode, ParameterNode, Edge
)

def test_kpi_node_fields():
    node = KPINode(id="kpi:dl_throughput", name="DL Throughput",
                   unit="Mbps", good_direction=Direction.POSITIVE)
    assert node.id == "kpi:dl_throughput"
    assert node.good_direction == Direction.POSITIVE

def test_feature_node_fields():
    node = FeatureNode(id="feature:CA", name="Carrier Aggregation",
                       gen=Generation.BOTH, category="radio_resource_management")
    assert node.gen == Generation.BOTH

def test_parameter_node_fields():
    node = ParameterNode(id="param:maxCaBands", name="maxCaBands",
                         data_type="int", range_min=1.0, range_max=4.0,
                         default_value="1", unit="")
    assert node.range_max == 4.0

def test_edge_fields():
    edge = Edge(from_id="feature:CA", to_id="kpi:dl_throughput",
                relation=EdgeType.AFFECTS, direction=Direction.POSITIVE,
                magnitude=Magnitude.HIGH, confidence=0.92, validated=False)
    assert edge.confidence == 0.92

def test_edge_defaults():
    edge = Edge(from_id="feature:CA", to_id="param:maxCaBands",
                relation=EdgeType.CONTROLLED_BY)
    assert edge.direction is None
    assert edge.validated is False
    assert edge.confidence == 1.0
