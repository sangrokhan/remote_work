from __future__ import annotations

from spar.preprocessing.abbrev_mapper import extract_terms, get_all_keywords


def test_extract_terms_finds_parameter_name() -> None:
    entities = {"parameter_names": ["nrDlCellMaxTxPower"], "counter_names": [], "counter_groups": [], "alarm_ids": [], "alarm_names": [], "yang_paths": [], "feature_names": []}
    acronyms = {"global": {}}
    keywords = get_all_keywords(acronyms, entities)
    found = extract_terms("The parameter nrDlCellMaxTxPower controls downlink power", keywords)
    assert "nrDlCellMaxTxPower" in found


def test_extract_terms_finds_counter_group() -> None:
    entities = {"parameter_names": [], "counter_names": [], "counter_groups": ["RRC", "Mobility"], "alarm_ids": [], "alarm_names": [], "yang_paths": [], "feature_names": []}
    acronyms = {"global": {}}
    keywords = get_all_keywords(acronyms, entities)
    found = extract_terms("RRC connection counters show high values", keywords)
    assert "RRC" in found


def test_extract_terms_3gpp_acronym_still_works() -> None:
    acronyms = {"global": {"HO": {"expansion": "Handover", "variants": []}}}
    keywords = get_all_keywords(acronyms, {})
    found = extract_terms("HO failure rate increased after upgrade", keywords)
    assert "HO" in found


def test_extract_terms_alarm_id_not_matched() -> None:
    entities = {"parameter_names": [], "counter_names": [], "counter_groups": [], "alarm_ids": ["4050"], "alarm_names": [], "yang_paths": [], "feature_names": []}
    acronyms = {"global": {}}
    keywords = get_all_keywords(acronyms, entities)
    # alarm_ids are digits-only → filtered by _NOISE_PATTERN → not in keywords
    assert "4050" not in keywords
