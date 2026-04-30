from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

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

    def test_no_double_expansion_on_reentry(self) -> None:
        text = "HO is triggered."
        once = map_abbreviations(text, SAMPLE_ACRONYMS)
        twice = map_abbreviations(once, SAMPLE_ACRONYMS)
        assert twice.count("HO(Handover)") == once.count("HO(Handover)")


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


class TestConflictLlmResolution:
    def _make_llm_client(self, json_response: str) -> MagicMock:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = json_response
        client.chat.completions.create.return_value = MagicMock(choices=[choice])
        return client

    def test_llm_high_confidence_single_expansion(self) -> None:
        text = "CA is configured between PCell and SCell for throughput."
        client = self._make_llm_client('{"CA": "Carrier Aggregation"}')
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert "CA(Carrier Aggregation)" in result
        assert "|" not in result

    def test_llm_uncertain_annotates_all_candidates(self) -> None:
        text = "The CA procedure was initiated."
        client = self._make_llm_client('{"CA": "uncertain"}')
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert "CA(Carrier Aggregation|Cell Activation)" in result

    def test_llm_json_parse_failure_falls_back_to_all_candidates(self) -> None:
        text = "CA enabled for this UE."
        client = self._make_llm_client("not valid json")
        result = map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert "CA(Carrier Aggregation|Cell Activation)" in result

    def test_no_conflict_in_text_skips_llm_call(self) -> None:
        text = "HO triggered after TTT expires."
        client = self._make_llm_client("{}")
        map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        client.chat.completions.create.assert_not_called()

    def test_single_llm_call_for_multiple_conflict_occurrences(self) -> None:
        text = "CA is used here. CA is also used there."
        client = self._make_llm_client('{"CA": "Carrier Aggregation"}')
        map_abbreviations(text, SAMPLE_ACRONYMS, llm_client=client)
        assert client.chat.completions.create.call_count == 1
