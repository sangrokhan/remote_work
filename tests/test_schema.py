from causality_graph.schema import (
    NodeType, EdgeType, Direction, Magnitude, Layer,
    ControlGroupNode, FunctionNode, FeatureNode, LayerNode,
    KPINode, ParameterNode, Edge,
)


def test_node_types():
    assert NodeType.CONTROL_GROUP.value == "control_group"
    assert NodeType.FUNCTION.value == "function"
    assert NodeType.FEATURE.value == "feature"
    assert NodeType.LAYER.value == "layer"
    assert NodeType.KPI.value == "kpi"
    assert NodeType.PARAMETER.value == "parameter"


def test_edge_types():
    assert EdgeType.INCLUDES.value == "INCLUDES"
    assert EdgeType.REALIZED_BY.value == "REALIZED_BY"
    assert EdgeType.MEASURED_BY.value == "MEASURED_BY"
    assert EdgeType.IMPLEMENTED_IN.value == "IMPLEMENTED_IN"
    assert EdgeType.TUNED_BY.value == "TUNED_BY"
    assert EdgeType.DEPENDS_ON.value == "DEPENDS_ON"
    assert EdgeType.AFFECTS.value == "AFFECTS"
    assert EdgeType.CONFLICTS_WITH.value == "CONFLICTS_WITH"


def test_control_group_node():
    node = ControlGroupNode(id="cg:mobility", name="Mobility")
    assert node.id == "cg:mobility"
    assert node.description == ""


def test_function_node():
    node = FunctionNode(id="fn:handover", name="Handover")
    assert node.id == "fn:handover"


def test_feature_node_fields():
    node = FeatureNode(id="feat:intra_freq_ho", name="Intra-frequency HO",
                       layer=Layer.RRC)
    assert node.layer == Layer.RRC
    assert node.description == ""


def test_layer_node():
    node = LayerNode(id="layer:mac", name=Layer.MAC)
    assert node.name == Layer.MAC


def test_kpi_node_fields():
    node = KPINode(id="kpi:dl_throughput", name="DL Throughput",
                   unit="Mbps", good_direction=Direction.POSITIVE,
                   category="integrity", layer="MAC",
                   spec_ref="TS 28.552 §5.5.1.1")
    assert node.id == "kpi:dl_throughput"
    assert node.good_direction == Direction.POSITIVE
    assert node.spec_ref == "TS 28.552 §5.5.1.1"


def test_parameter_node_fields():
    node = ParameterNode(id="param:A3Offset", name="A3Offset",
                         data_type="float", range_min=-15.0, range_max=15.0,
                         default_value="3", unit="dB",
                         spec_ref="TS 38.331 §6.3.2")
    assert node.range_max == 15.0
    assert node.spec_ref == "TS 38.331 §6.3.2"


def test_edge_affects():
    edge = Edge(from_id="param:A3Offset", to_id="kpi:ho_intra_freq_success_rate",
                relation=EdgeType.AFFECTS, direction=Direction.POSITIVE,
                magnitude=Magnitude.HIGH, confidence=0.92)
    assert edge.confidence == 0.92
    assert edge.magnitude == Magnitude.HIGH


def test_edge_defaults():
    edge = Edge(from_id="feat:intra_freq_ho", to_id="param:A3Offset",
                relation=EdgeType.TUNED_BY)
    assert edge.direction is None
    assert edge.validated is False
    assert edge.confidence == 1.0
