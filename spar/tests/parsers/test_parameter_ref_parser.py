"""ParameterRefParser 단위 테스트."""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from spar.parsers.parameter_ref_parser import (
    ParameterRecord,
    parse_parameter_ref_excel,
)

SAMPLE_PATH = (
    Path(__file__).parent.parent.parent
    / "data" / "samples" / "parameter_ref_sample.xlsx"
)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

def _make_excel(tmp_path: Path, rows: list[list], headers: list[str] | None = None) -> Path:
    """테스트용 임시 Excel 파일 생성."""
    if headers is None:
        headers = ["Feature Name", "YANG Path", "Parameter Name",
                   "Type", "Default", "Min", "Max", "Description"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(headers)
    for row in rows:
        ws.append(row)
    p = tmp_path / "test.xlsx"
    wb.save(p)
    return p


# ---------------------------------------------------------------------------
# 기본 파싱
# ---------------------------------------------------------------------------

class TestBasicParsing:
    def test_sample_file_loads(self):
        result = parse_parameter_ref_excel(SAMPLE_PATH)
        assert len(result.records) == 10
        assert result.skipped_rows == 0

    def test_record_fields_populated(self):
        result = parse_parameter_ref_excel(SAMPLE_PATH)
        rec = result.records[0]
        assert rec.param_name == "hoPrepTimerHO"
        assert rec.yang_path == "ManagedElement/GNBCUCPFunction/NRCellCU/HandoverConfig"
        assert rec.feature_name == "HO"
        assert rec.type == "INTEGER"
        assert rec.default == "100"
        assert rec.min == "0"
        assert rec.max == "1023"
        assert "Handover preparation" in rec.description

    def test_enumeration_no_min_max(self):
        result = parse_parameter_ref_excel(SAMPLE_PATH)
        rach_enum = next(r for r in result.records if r.param_name == "powerRampingStep")
        assert rach_enum.type == "ENUMERATION"
        assert rach_enum.min == ""
        assert rach_enum.max == ""

    def test_all_features_present(self):
        result = parse_parameter_ref_excel(SAMPLE_PATH)
        features = {r.feature_name for r in result.records}
        assert features == {"HO", "CA", "RACH", "BWP", "TTT"}


# ---------------------------------------------------------------------------
# YANG Path 헬퍼
# ---------------------------------------------------------------------------

class TestYangPathHelpers:
    def test_mo_path_splits_correctly(self):
        rec = ParameterRecord(
            param_name="x",
            yang_path="ManagedElement/GNBCUCPFunction/NRCellCU/HandoverConfig",
        )
        assert rec.mo_path == ["ManagedElement", "GNBCUCPFunction", "NRCellCU", "HandoverConfig"]

    def test_leaf_mo(self):
        rec = ParameterRecord(
            param_name="x",
            yang_path="ManagedElement/GNBDUFunction/NRCellDU/RachConfig",
        )
        assert rec.leaf_mo == "RachConfig"

    def test_empty_yang_path(self):
        rec = ParameterRecord(param_name="x", yang_path="")
        assert rec.mo_path == []
        assert rec.leaf_mo == ""


# ---------------------------------------------------------------------------
# to_chunk_text
# ---------------------------------------------------------------------------

class TestToChunkText:
    def test_chunk_text_contains_key_fields(self):
        rec = ParameterRecord(
            param_name="hoPrepTimerHO",
            yang_path="ManagedElement/GNBCUCPFunction/NRCellCU/HandoverConfig",
            feature_name="HO",
            type="INTEGER",
            default="100",
            min="0",
            max="1023",
            description="Handover prep timer in ms.",
        )
        text = rec.to_chunk_text()
        assert "hoPrepTimerHO" in text
        assert "ManagedElement/GNBCUCPFunction" in text
        assert "HO" in text
        assert "default=100" in text
        assert "min=0" in text
        assert "max=1023" in text
        assert "Handover prep timer" in text

    def test_chunk_text_omits_empty_range(self):
        rec = ParameterRecord(param_name="x", yang_path="A/B")
        text = rec.to_chunk_text()
        assert "Value:" not in text


# ---------------------------------------------------------------------------
# 헤더 유연성
# ---------------------------------------------------------------------------

class TestHeaderFlexibility:
    def test_alternate_column_names(self, tmp_path):
        p = _make_excel(tmp_path, [
            ["HO", "ManagedElement/GNBDUFunction/NRCellDU", "testParam",
             "INTEGER", "10", "0", "100", "Test desc"],
        ], headers=["Feature", "YANG", "Param Name", "Type", "Default Value", "Minimum", "Maximum", "Desc"])
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 1
        rec = result.records[0]
        assert rec.param_name == "testParam"
        assert rec.feature_name == "HO"
        assert rec.min == "0"

    def test_header_not_on_first_row(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Some metadata line"])
        ws.append(["Feature Name", "YANG Path", "Parameter Name",
                   "Type", "Default", "Min", "Max", "Description"])
        ws.append(["BWP", "ManagedElement/GNBDUFunction/NRCellDU/BWPConfig",
                   "bwpTimer", "INTEGER", "100", "0", "2560", "BWP timer"])
        p = tmp_path / "offset.xlsx"
        wb.save(p)
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 1
        assert result.records[0].param_name == "bwpTimer"


# ---------------------------------------------------------------------------
# 엣지 케이스 / 오류 처리
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_skips_row_missing_param_name(self, tmp_path):
        p = _make_excel(tmp_path, [
            ["HO", "ManagedElement/A/B", "", "INTEGER", "0", "0", "10", "desc"],
            ["HO", "ManagedElement/A/B", "validParam", "INTEGER", "0", "0", "10", "desc"],
        ])
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 1
        assert result.skipped_rows == 1
        assert len(result.warnings) == 1

    def test_skips_row_missing_yang_path(self, tmp_path):
        p = _make_excel(tmp_path, [
            ["HO", "", "paramX", "INTEGER", "0", "0", "10", "desc"],
        ])
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 0
        assert result.skipped_rows == 1

    def test_skips_entirely_empty_rows(self, tmp_path):
        p = _make_excel(tmp_path, [
            ["HO", "ManagedElement/A/B", "p1", "INTEGER", "0", "0", "10", "d"],
            [None, None, None, None, None, None, None, None],
            ["CA", "ManagedElement/A/C", "p2", "INTEGER", "1", "0", "10", "d"],
        ])
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 2

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_parameter_ref_excel("/nonexistent/file.xlsx")

    def test_missing_required_headers_raises(self, tmp_path):
        p = _make_excel(tmp_path, [["HO", "path", "p"]], headers=["Feature", "Path", "Name"])
        with pytest.raises(ValueError, match="헤더 행 탐색 실패"):
            parse_parameter_ref_excel(p)

    def test_to_dict_keys(self):
        rec = ParameterRecord(param_name="p", yang_path="A/B")
        d = rec.to_dict()
        assert set(d.keys()) == {
            "param_name", "yang_path", "feature_name",
            "type", "default", "min", "max", "description"
        }
