from pathlib import Path
import io
from causality_graph.extraction.excel_parser import parse_param_csv
from causality_graph.schema import ParameterNode

FIXTURE = Path("tests/fixtures/sample_params.csv")


def test_returns_parameter_nodes():
    nodes = parse_param_csv(FIXTURE)
    assert len(nodes) == 3
    assert all(isinstance(n, ParameterNode) for n in nodes)


def test_node_fields():
    nodes = parse_param_csv(FIXTURE)
    ca = next(n for n in nodes if n.id == "param:maxCaBands")
    assert ca.name == "maxCaBands"
    assert ca.data_type == "int"
    assert ca.range_min == 1.0
    assert ca.range_max == 4.0
    assert ca.default_value == "1"


def test_linked_feature_id_in_metadata():
    nodes, metadata = parse_param_csv(FIXTURE, return_metadata=True)
    ca_meta = next(m for m in metadata if m["id"] == "param:maxCaBands")
    assert ca_meta["linked_feature_id"] == "feature:CA"


def test_missing_range_is_none():
    csv_text = "id,name,data_type,range_min,range_max,default_value,unit,description,linked_feature_id\nparam:X,X,str,,,default,,desc,feature:Y\n"
    nodes = parse_param_csv(io.StringIO(csv_text))
    assert nodes[0].range_min is None
    assert nodes[0].range_max is None
