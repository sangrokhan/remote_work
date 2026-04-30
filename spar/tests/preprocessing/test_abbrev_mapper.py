from __future__ import annotations

import json
from pathlib import Path

import pytest

from spar.preprocessing.abbrev_mapper import (
    build_reverse_index,
    load_acronyms,
    map_abbreviations,
)

SAMPLE_ACRONYMS = {
    "global": {
        "HO": {"expansion": "Handover", "variants": ["Hand-Over"]},
        "TTT": {"expansion": "Time-To-Trigger", "variants": []},
        "RACH": {"expansion": "Random Access Channel", "variants": []},
    },
    "conflicts": {
        "CA": {"candidates": ["Carrier Aggregation", "Cell Activation"], "variants": []},
    },
}


class TestLoadAcronyms:
    def test_loads_json_from_path(self, tmp_path: Path) -> None:
        f = tmp_path / "acronyms.json"
        f.write_text(json.dumps(SAMPLE_ACRONYMS), encoding="utf-8")
        result = load_acronyms(f)
        assert result["global"]["HO"]["expansion"] == "Handover"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_acronyms(tmp_path / "nonexistent.json")


class TestApplyGlobal:
    def test_expands_abbreviation_with_annotation(self) -> None:
        text = "HO is triggered when TTT expires."
        result = map_abbreviations(text, SAMPLE_ACRONYMS)
        assert "HO(Handover)" in result
        assert "TTT(Time-To-Trigger)" in result

    def test_expands_variant(self) -> None:
        text = "Hand-Over failure occurred."
        result = map_abbreviations(text, SAMPLE_ACRONYMS)
        assert "Hand-Over(Handover)" in result

    def test_word_boundary_not_partial_match(self) -> None:
        text = "RACHAEL sent RACH preamble."
        result = map_abbreviations(text, SAMPLE_ACRONYMS)
        assert "RACHAEL" in result
        assert "RACHAEL(Random Access Channel)" not in result
        assert "RACH(Random Access Channel)" in result

    def test_no_llm_client_conflict_annotates_all_candidates(self) -> None:
        text = "CA is configured between PCell and SCell."
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=None)
        assert "CA(Carrier Aggregation|Cell Activation)" in result


class TestBuildReverseIndex:
    def test_expansion_maps_to_abbreviation(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        assert reverse["Handover"] == "HO"
        assert reverse["handover"] == "HO"

    def test_variant_maps_to_abbreviation(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        assert reverse["Hand-Over"] == "HO"
        assert reverse["hand-over"] == "HO"

    def test_hyphen_stripped_form_maps(self) -> None:
        reverse = build_reverse_index(SAMPLE_ACRONYMS)
        assert reverse["handover"] == "HO"
