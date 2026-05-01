from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pytest

from spar.ingest.excel_loader import load_excel_terms, merge_into_acronyms


def _make_excel(tmp_path: Path, rows: list[dict]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    if rows:
        ws.append(list(rows[0].keys()))
        for row in rows:
            ws.append(list(row.values()))
    path = tmp_path / "test.xlsx"
    wb.save(path)
    return path


class TestLoadExcelTerms:
    def test_extracts_single_column(self, tmp_path: Path) -> None:
        path = _make_excel(tmp_path, [
            {"Parameter Name": "maxRetransmissions", "Description": "Max retrans"},
            {"Parameter Name": "t301", "Description": "Timer T301"},
        ])
        result = load_excel_terms(path, columns=["Parameter Name"])
        assert "maxRetransmissions" in result
        assert "t301" in result
        assert result["maxRetransmissions"] == {"type": "keyword"}

    def test_extracts_multiple_columns(self, tmp_path: Path) -> None:
        path = _make_excel(tmp_path, [
            {"Param": "maxRetrans", "Alarm": "ALM-001"},
        ])
        result = load_excel_terms(path, columns=["Param", "Alarm"])
        assert "maxRetrans" in result
        assert "ALM-001" in result

    def test_skips_empty_cells(self, tmp_path: Path) -> None:
        path = _make_excel(tmp_path, [
            {"Parameter Name": ""},
            {"Parameter Name": None},
            {"Parameter Name": "validParam"},
        ])
        result = load_excel_terms(path, columns=["Parameter Name"])
        assert list(result.keys()) == ["validParam"]

    def test_ignores_unknown_column(self, tmp_path: Path) -> None:
        path = _make_excel(tmp_path, [{"Param": "p1"}])
        result = load_excel_terms(path, columns=["NonExistent"])
        assert result == {}

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        path = _make_excel(tmp_path, [{"Param": "  NRCellDU  "}])
        result = load_excel_terms(path, columns=["Param"])
        assert "NRCellDU" in result


class TestMergeIntoAcronyms:
    def test_adds_keywords_section(self, tmp_path: Path) -> None:
        acronyms_path = tmp_path / "acronyms.json"
        acronyms_path.write_text(json.dumps({"global": {}, "conflicts": {}}))
        merge_into_acronyms({"NRCellDU": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        assert data["keywords"]["NRCellDU"] == {"type": "keyword"}

    def test_does_not_remove_global_entries(self, tmp_path: Path) -> None:
        acronyms_path = tmp_path / "acronyms.json"
        existing = {"global": {"HO": {"expansion": "Handover"}}, "conflicts": {}}
        acronyms_path.write_text(json.dumps(existing))
        merge_into_acronyms({"p1": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        assert data["global"]["HO"]["expansion"] == "Handover"

    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        acronyms_path = tmp_path / "new_acronyms.json"
        merge_into_acronyms({"p1": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        assert "p1" in data["keywords"]

    def test_merges_without_removing_existing_keywords(self, tmp_path: Path) -> None:
        acronyms_path = tmp_path / "acronyms.json"
        acronyms_path.write_text(json.dumps({
            "global": {}, "conflicts": {}, "keywords": {"existing": {}}
        }))
        merge_into_acronyms({"new": {"type": "keyword"}}, acronyms_path)
        data = json.loads(acronyms_path.read_text())
        assert "existing" in data["keywords"]
        assert "new" in data["keywords"]
