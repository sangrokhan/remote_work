from causality_graph.schema import (
    ControlGroupNode, FunctionNode, FeatureNode, LayerNode,
    KPINode, ParameterNode, Edge,
    Direction, EdgeType, Layer, Magnitude,
)
from causality_graph.store.graph import CausalityGraph


def make_sample_graph() -> CausalityGraph:
    g = CausalityGraph()

    # ── Layers ────────────────────────────────────────────────────────────────
    g.add_layer(LayerNode(id="layer:phy",  name=Layer.PHY))
    g.add_layer(LayerNode(id="layer:mac",  name=Layer.MAC))
    g.add_layer(LayerNode(id="layer:rrc",  name=Layer.RRC))

    # ── KPIs ──────────────────────────────────────────────────────────────────
    g.add_kpi(KPINode(id="kpi:dl_cell_throughput",
                      name="DL Cell Throughput", unit="Mbps",
                      good_direction=Direction.POSITIVE,
                      category="integrity", layer="PDCP",
                      spec_ref="TS 28.552 §5.5.1.1"))
    g.add_kpi(KPINode(id="kpi:avg_active_scell_count",
                      name="Average Active SCell Count", unit="count",
                      good_direction=Direction.POSITIVE,
                      category="ca", layer="RRC",
                      spec_ref="TS 28.552 §5.10.2.1"))
    g.add_kpi(KPINode(id="kpi:scell_add_success_rate",
                      name="SCell Addition Success Rate", unit="%",
                      good_direction=Direction.POSITIVE,
                      category="ca", layer="RRC",
                      spec_ref="TS 28.552 §5.10.1.1"))
    g.add_kpi(KPINode(id="kpi:dl_mimo_avg_layers",
                      name="DL Average MIMO Layers", unit="count",
                      good_direction=Direction.POSITIVE,
                      category="beam", layer="PHY",
                      spec_ref="TS 28.552 §5.9.2.1"))
    g.add_kpi(KPINode(id="kpi:ho_intra_freq_success_rate",
                      name="Intra-frequency HO Success Rate", unit="%",
                      good_direction=Direction.POSITIVE,
                      category="mobility", layer="RRC",
                      spec_ref="TS 28.552 §5.3.1.1"))
    g.add_kpi(KPINode(id="kpi:ho_ping_pong_rate",
                      name="Ping-Pong HO Rate", unit="%",
                      good_direction=Direction.NEGATIVE,
                      category="mobility", layer="RRC",
                      spec_ref="TS 28.552 §5.3.4.1"))
    g.add_kpi(KPINode(id="kpi:ho_interruption_time",
                      name="HO Interruption Time", unit="ms",
                      good_direction=Direction.NEGATIVE,
                      category="mobility", layer="RRC",
                      spec_ref="TS 28.552 §5.3.1.5"))

    # ── Parameters ────────────────────────────────────────────────────────────
    g.add_parameter(ParameterNode(id="param:maxCaBands", name="maxCaBands",
                                  data_type="int", range_min=1, range_max=4,
                                  default_value="1",
                                  spec_ref="TS 38.331 §6.3.2"))
    g.add_parameter(ParameterNode(id="param:mimoLayers", name="mimoLayers",
                                  data_type="int", range_min=1, range_max=8,
                                  default_value="2",
                                  spec_ref="TS 38.331 §6.3.2"))
    g.add_parameter(ParameterNode(id="param:A3Offset", name="A3Offset",
                                  data_type="float", range_min=-15.0, range_max=15.0,
                                  default_value="3", unit="dB",
                                  spec_ref="TS 38.331 §6.3.2"))
    g.add_parameter(ParameterNode(id="param:hysteresis", name="hysteresis",
                                  data_type="float", range_min=0.0, range_max=15.0,
                                  default_value="3", unit="dB",
                                  spec_ref="TS 38.331 §6.3.2"))
    g.add_parameter(ParameterNode(id="param:timeToTrigger", name="timeToTrigger",
                                  data_type="int", range_min=0, range_max=5120,
                                  default_value="320", unit="ms",
                                  spec_ref="TS 38.331 §6.3.2"))

    # ── ControlGroups ─────────────────────────────────────────────────────────
    g.add_control_group(ControlGroupNode(id="cg:media_access", name="MediaAccess",
                                         description="MAC/PHY radio access control"))
    g.add_control_group(ControlGroupNode(id="cg:mobility", name="Mobility",
                                         description="UE movement and HO control"))

    # ── Functions ─────────────────────────────────────────────────────────────
    g.add_function(FunctionNode(id="fn:carrier_aggregation", name="Carrier Aggregation"))
    g.add_function(FunctionNode(id="fn:mimo",               name="MIMO"))
    g.add_function(FunctionNode(id="fn:handover",           name="Handover"))

    # ── Features ──────────────────────────────────────────────────────────────
    g.add_feature(FeatureNode(id="feat:scell_mgmt",    name="SCell Management",    layer=Layer.RRC))
    g.add_feature(FeatureNode(id="feat:su_mimo",       name="SU-MIMO",             layer=Layer.PHY))
    g.add_feature(FeatureNode(id="feat:intra_freq_ho", name="Intra-frequency HO",  layer=Layer.RRC))
    g.add_feature(FeatureNode(id="feat:cho",           name="Conditional HO (CHO)", layer=Layer.RRC))

    # ── Edges: ControlGroup → Function (INCLUDES) ─────────────────────────────
    g.add_edge(Edge(from_id="cg:media_access", to_id="fn:carrier_aggregation", relation=EdgeType.INCLUDES))
    g.add_edge(Edge(from_id="cg:media_access", to_id="fn:mimo",                relation=EdgeType.INCLUDES))
    g.add_edge(Edge(from_id="cg:mobility",     to_id="fn:handover",            relation=EdgeType.INCLUDES))

    # ── Edges: Function → Feature (REALIZED_BY) ───────────────────────────────
    g.add_edge(Edge(from_id="fn:carrier_aggregation", to_id="feat:scell_mgmt",    relation=EdgeType.REALIZED_BY))
    g.add_edge(Edge(from_id="fn:mimo",                to_id="feat:su_mimo",       relation=EdgeType.REALIZED_BY))
    g.add_edge(Edge(from_id="fn:handover",            to_id="feat:intra_freq_ho", relation=EdgeType.REALIZED_BY))
    g.add_edge(Edge(from_id="fn:handover",            to_id="feat:cho",           relation=EdgeType.REALIZED_BY))

    # ── Edges: Function → KPI (MEASURED_BY) ──────────────────────────────────
    g.add_edge(Edge(from_id="fn:carrier_aggregation", to_id="kpi:dl_cell_throughput",     relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="fn:mimo",                to_id="kpi:dl_cell_throughput",     relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="fn:handover",            to_id="kpi:ho_intra_freq_success_rate", relation=EdgeType.MEASURED_BY))

    # ── Edges: Feature → Layer (IMPLEMENTED_IN) ───────────────────────────────
    g.add_edge(Edge(from_id="feat:scell_mgmt",    to_id="layer:rrc", relation=EdgeType.IMPLEMENTED_IN))
    g.add_edge(Edge(from_id="feat:su_mimo",       to_id="layer:phy", relation=EdgeType.IMPLEMENTED_IN))
    g.add_edge(Edge(from_id="feat:intra_freq_ho", to_id="layer:rrc", relation=EdgeType.IMPLEMENTED_IN))
    g.add_edge(Edge(from_id="feat:cho",           to_id="layer:rrc", relation=EdgeType.IMPLEMENTED_IN))

    # ── Edges: Feature → KPI (MEASURED_BY) ───────────────────────────────────
    g.add_edge(Edge(from_id="feat:scell_mgmt",    to_id="kpi:avg_active_scell_count",       relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="feat:scell_mgmt",    to_id="kpi:scell_add_success_rate",       relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="feat:scell_mgmt",    to_id="kpi:dl_cell_throughput",           relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="feat:su_mimo",       to_id="kpi:dl_mimo_avg_layers",           relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="feat:su_mimo",       to_id="kpi:dl_cell_throughput",           relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="feat:intra_freq_ho", to_id="kpi:ho_intra_freq_success_rate",   relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="feat:intra_freq_ho", to_id="kpi:ho_ping_pong_rate",            relation=EdgeType.MEASURED_BY))
    g.add_edge(Edge(from_id="feat:cho",           to_id="kpi:ho_interruption_time",         relation=EdgeType.MEASURED_BY))

    # ── Edges: Feature → Parameter (TUNED_BY) ────────────────────────────────
    g.add_edge(Edge(from_id="feat:scell_mgmt",    to_id="param:maxCaBands",   relation=EdgeType.TUNED_BY))
    g.add_edge(Edge(from_id="feat:su_mimo",       to_id="param:mimoLayers",   relation=EdgeType.TUNED_BY))
    g.add_edge(Edge(from_id="feat:intra_freq_ho", to_id="param:A3Offset",     relation=EdgeType.TUNED_BY))
    g.add_edge(Edge(from_id="feat:intra_freq_ho", to_id="param:hysteresis",   relation=EdgeType.TUNED_BY))
    g.add_edge(Edge(from_id="feat:intra_freq_ho", to_id="param:timeToTrigger", relation=EdgeType.TUNED_BY))

    # ── Edges: Feature → Feature (DEPENDS_ON) ────────────────────────────────
    g.add_edge(Edge(from_id="feat:cho", to_id="feat:intra_freq_ho", relation=EdgeType.DEPENDS_ON))

    # ── Edges: Parameter → KPI (AFFECTS) ─────────────────────────────────────
    g.add_edge(Edge(from_id="param:maxCaBands", to_id="kpi:dl_cell_throughput",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.HIGH, confidence=0.92))
    g.add_edge(Edge(from_id="param:maxCaBands", to_id="kpi:avg_active_scell_count",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.HIGH, confidence=0.95))
    g.add_edge(Edge(from_id="param:mimoLayers", to_id="kpi:dl_cell_throughput",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.HIGH, confidence=0.88))
    g.add_edge(Edge(from_id="param:mimoLayers", to_id="kpi:dl_mimo_avg_layers",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.HIGH, confidence=0.97))
    g.add_edge(Edge(from_id="param:A3Offset", to_id="kpi:ho_intra_freq_success_rate",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.HIGH, confidence=0.85))
    g.add_edge(Edge(from_id="param:A3Offset", to_id="kpi:ho_ping_pong_rate",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.NEGATIVE, magnitude=Magnitude.MEDIUM, confidence=0.80))
    g.add_edge(Edge(from_id="param:hysteresis", to_id="kpi:ho_intra_freq_success_rate",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.MEDIUM, confidence=0.75))
    g.add_edge(Edge(from_id="param:hysteresis", to_id="kpi:ho_ping_pong_rate",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.NEGATIVE, magnitude=Magnitude.MEDIUM, confidence=0.78))
    g.add_edge(Edge(from_id="param:timeToTrigger", to_id="kpi:ho_intra_freq_success_rate",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.LOW, confidence=0.70))
    g.add_edge(Edge(from_id="param:timeToTrigger", to_id="kpi:ho_interruption_time",
                    relation=EdgeType.AFFECTS,
                    direction=Direction.POSITIVE, magnitude=Magnitude.MEDIUM, confidence=0.72))

    # ── Edges: Parameter → Parameter (CONFLICTS_WITH) ─────────────────────────
    g.add_edge(Edge(from_id="param:A3Offset", to_id="param:hysteresis",
                    relation=EdgeType.CONFLICTS_WITH,
                    notes="Both tune HO trigger threshold; increasing both over-delays HO"))

    return g
