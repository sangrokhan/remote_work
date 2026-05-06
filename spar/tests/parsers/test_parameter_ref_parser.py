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
        headers = ["Parameter Name", "YANG Path", "Type",
                   "Default", "Min", "Max", "Description", "Feature Name"]
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
        assert len(result.records) == 11
        assert result.skipped_rows == 0

    def test_record_fields_populated(self):
        result = parse_parameter_ref_excel(SAMPLE_PATH)
        rec = result.records[0]
        assert rec.param_name == "handover-preparation-timer"
        assert rec.yang_path == "ManagedElement/GNBCUCPFunction/NRCellCU/HandoverConfig"
        assert rec.related_feature_id == "FGR-HO0101"
        assert rec.type == "INTEGER"
        assert rec.default == "100"
        assert rec.range == "0..1023"
        assert "Handover preparation" in rec.description

    def test_enumeration_no_range(self):
        result = parse_parameter_ref_excel(SAMPLE_PATH)
        rach_enum = next(r for r in result.records if r.param_name == "rach-power-ramping-step")
        assert rach_enum.type == "ENUMERATION"
        assert rach_enum.range == ""

    def test_all_features_present(self):
        result = parse_parameter_ref_excel(SAMPLE_PATH)
        features = {r.related_feature_id for r in result.records}
        assert features == {"FGR-HO0101", "FGR-HO0102", "FGR-CA0201", "FGR-RS0101", "FGR-BW0301"}


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
            param_name="handover-preparation-timer",
            yang_path="ManagedElement/GNBCUCPFunction/NRCellCU/HandoverConfig",
            feature_name="FGR-HO0101",
            type="INTEGER",
            default="100",
            min="0",
            max="1023",
            description="Handover prep timer in ms.",
        )
        text = rec.to_chunk_text()
        assert "handover-preparation-timer" in text
        assert "ManagedElement/GNBCUCPFunction" in text
        assert "FGR-HO0101" in text
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
            ["test-param", "ManagedElement/GNBDUFunction/NRCellDU", "INTEGER",
             "10", "0", "100", "Test desc", "FGR-TS0101"],
        ], headers=["Param Name", "YANG", "Type", "Default Value", "Minimum", "Maximum", "Desc", "Feature"])
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 1
        rec = result.records[0]
        assert rec.param_name == "test-param"
        assert rec.feature_name == "FGR-TS0101"
        assert rec.min == "0"

    def test_header_not_on_first_row(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Some metadata line"])
        ws.append(["Parameter Name", "YANG Path", "Type",
                   "Default", "Min", "Max", "Description", "Feature Name"])
        ws.append(["bwp-inactivity-timer", "ManagedElement/GNBDUFunction/NRCellDU/BWPConfig",
                   "INTEGER", "100", "0", "2560", "BWP timer", "FGR-BW0301"])
        p = tmp_path / "offset.xlsx"
        wb.save(p)
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 1
        assert result.records[0].param_name == "bwp-inactivity-timer"


# ---------------------------------------------------------------------------
# 엣지 케이스 / 오류 처리
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_skips_row_missing_param_name(self, tmp_path):
        p = _make_excel(tmp_path, [
            ["", "ManagedElement/A/B", "INTEGER", "0", "0", "10", "desc", "FGR-TS0101"],
            ["valid-param", "ManagedElement/A/B", "INTEGER", "0", "0", "10", "desc", "FGR-TS0101"],
        ])
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 1
        assert result.skipped_rows == 1
        assert len(result.warnings) == 1

    def test_skips_row_missing_yang_path(self, tmp_path):
        p = _make_excel(tmp_path, [
            ["param-x", "", "INTEGER", "0", "0", "10", "desc", "FGR-TS0101"],
        ])
        result = parse_parameter_ref_excel(p)
        assert len(result.records) == 0
        assert result.skipped_rows == 1

    def test_skips_entirely_empty_rows(self, tmp_path):
        p = _make_excel(tmp_path, [
            ["param-one", "ManagedElement/A/B", "INTEGER", "0", "0", "10", "d", "FGR-TS0101"],
            [None, None, None, None, None, None, None, None],
            ["param-two", "ManagedElement/A/C", "INTEGER", "1", "0", "10", "d", "FGR-TS0102"],
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
            "param_name", "yang_path", "description",
            "leaf_status", "units", "type", "pattern", "range", "default",
            "bandwidth_dependency", "config_value", "level", "restriction",
            "service_impact", "realtime_change", "reference", "mandatory",
            "param_family", "related_feature_id", "user_level",
            "feature_name", "min", "max",
        }
