from __future__ import annotations

import json
from pathlib import Path

import pytest

PARAM_SAMPLE = Path("data/samples/parameter_ref_sample.xlsx")
COUNTER_SAMPLE = Path("data/samples/counter_ref_sample.xlsx")
ALARM_SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")


@pytest.mark.skipif(not PARAM_SAMPLE.exists(), reason="parameter_ref_sample.xlsx not found")
def test_build_entity_glossary_produces_valid_json(tmp_path: Path) -> None:
    from build_entity_glossary import build_and_write

    output = tmp_path / "samsung_entities.json"
    build_and_write(
        param_paths=[PARAM_SAMPLE],
        counter_paths=[COUNTER_SAMPLE] if COUNTER_SAMPLE.exists() else [],
        alarm_paths=[ALARM_SAMPLE] if ALARM_SAMPLE.exists() else [],
        output_path=output,
    )
    assert output.exists()
    data = json.loads(output.read_text())
    assert "parameters" in data
    assert "counters" in data
    assert "alarms" in data
    assert isinstance(data["parameters"], list)
    assert len(data["parameters"]) > 0


@pytest.mark.skipif(not PARAM_SAMPLE.exists(), reason="parameter_ref_sample.xlsx not found")
def test_build_entity_glossary_no_noise_tokens(tmp_path: Path) -> None:
    from build_entity_glossary import build_and_write

    output = tmp_path / "samsung_entities.json"
    build_and_write(
        param_paths=[PARAM_SAMPLE],
        counter_paths=[],
        alarm_paths=[],
        output_path=output,
    )
    data = json.loads(output.read_text())
    for item in data.get("parameters", []):
        name = item["name"] if isinstance(item, dict) else item
        assert not name.isdigit(), f"noise: digits-only token: {name}"
        assert len(name) > 2, f"noise: too-short token: {name}"


@pytest.mark.skipif(not COUNTER_SAMPLE.exists(), reason="counter_ref_sample.xlsx not found")
def test_scan_counter_refs_extracts_groups(tmp_path: Path) -> None:
    from build_entity_glossary import build_and_write

    output = tmp_path / "samsung_entities.json"
    build_and_write(
        param_paths=[],
        counter_paths=[COUNTER_SAMPLE],
        alarm_paths=[],
        output_path=output,
    )
    data = json.loads(output.read_text())
    assert "counters" in data
    counters = data["counters"]
    assert len(counters) > 0
    assert all("name" in c and "group_name" in c and "large_group" in c for c in counters)


def test_build_entity_glossary_empty_inputs(tmp_path: Path) -> None:
    from build_entity_glossary import build_and_write

    output = tmp_path / "samsung_entities.json"
    result = build_and_write(param_paths=[], counter_paths=[], alarm_paths=[], output_path=output)
    assert output.exists()
    data = json.loads(output.read_text())
    for v in data.values():
        assert v == []
