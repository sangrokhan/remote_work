from pathlib import Path
from causality_graph.extraction.md_parser import parse_feature_doc, ParsedFeature

FIXTURE = Path("tests/fixtures/sample_feature.md")


def test_parses_feature_id_and_name():
    result = parse_feature_doc(FIXTURE.read_text())
    assert result.feature_id == "feature:CA"
    assert result.name == "Carrier Aggregation (CA)"


def test_parses_generation():
    result = parse_feature_doc(FIXTURE.read_text())
    assert result.gen == "both"


def test_parses_kpi_impacts():
    result = parse_feature_doc(FIXTURE.read_text())
    assert len(result.kpi_impacts) == 2
    dl = next(k for k in result.kpi_impacts if k["kpi_id"] == "kpi:dl_throughput")
    assert dl["direction"] == "+"
    assert dl["magnitude"] == "high"
    assert "multi-band" in dl["condition"]


def test_parses_controlling_params():
    result = parse_feature_doc(FIXTURE.read_text())
    assert len(result.controlling_params) == 2
    ids = [p["param_id"] for p in result.controlling_params]
    assert "param:maxCaBands" in ids


def test_parses_feature_dependencies():
    result = parse_feature_doc(FIXTURE.read_text())
    assert len(result.dependencies) == 1
    assert result.dependencies[0]["feature_id"] == "feature:MIMO"
    assert result.dependencies[0]["dep_type"] == "enables"


def test_missing_section_returns_empty_list():
    minimal = """# Feature: Test\n**Feature ID**: feature:TEST\n**Generation**: 5G\n**Category**: test\n## Description\nTest feature."""
    result = parse_feature_doc(minimal)
    assert result.kpi_impacts == []
    assert result.controlling_params == []
    assert result.dependencies == []
