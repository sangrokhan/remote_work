from __future__ import annotations

import json
from pathlib import Path

from spar.preprocessing.abbrev_mapper import get_all_keywords, load_entity_glossary


def test_load_entity_glossary_returns_dict(tmp_path: Path) -> None:
    entities = {
        "parameter_names": ["nrDlCellMaxTxPower", "rachRootSequenceIndex"],
        "counter_names": ["pmRrcConnEstabAtt"],
        "counter_groups": ["RRC"],
        "alarm_codes": ["4050"],
        "alarm_names": ["Cell Unavailable"],
        "yang_paths": ["NRCellDU/nrDlCellMaxTxPower"],
        "feature_names": ["MIMO"],
    }
    p = tmp_path / "samsung_entities.json"
    p.write_text(json.dumps(entities))
    result = load_entity_glossary(p)
    assert result["parameter_names"] == ["nrDlCellMaxTxPower", "rachRootSequenceIndex"]


def test_load_entity_glossary_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_entity_glossary(tmp_path / "nonexistent.json")
    assert result == {}


def test_get_all_keywords_combines_acronyms_and_entities() -> None:
    acronyms = {
        "global": {
            "HO": {"expansion": "Handover", "variants": []},
            "CA": {"expansion": "Carrier Aggregation", "variants": []},
        }
    }
    entities = {
        "parameter_names": ["nrDlCellMaxTxPower"],
        "counter_names": ["pmRrcConnEstabAtt"],
        "counter_groups": ["RRC"],
        "alarm_codes": ["4050"],
        "alarm_names": [],
        "yang_paths": [],
        "feature_names": [],
    }
    keywords = get_all_keywords(acronyms, entities)
    assert "HO" in keywords
    assert "CA" in keywords
    assert "nrDlCellMaxTxPower" in keywords
    assert "pmRrcConnEstabAtt" in keywords
    assert "RRC" in keywords
    assert "4050" not in keywords


def test_get_all_keywords_empty_entities() -> None:
    acronyms = {"global": {"HO": {"expansion": "Handover", "variants": []}}}
    keywords = get_all_keywords(acronyms, {})
    assert "HO" in keywords
