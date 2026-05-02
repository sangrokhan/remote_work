from pathlib import Path

import openpyxl
import pytest

from spar.parsers.alarm_ref_parser import (
    AlarmRecord,
    parse_alarm_ref_excel,
)

SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")


def _write_xlsx(tmp_path: Path, rows: list[tuple]) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    p = tmp_path / "t.xlsx"
    wb.save(p)
    return p


def test_parse_sample_round_trip():
    res = parse_alarm_ref_excel(SAMPLE)
    assert len(res.records) == 12
    ids = [r.alarm_id for r in res.records]
    assert ids[0] == "ALM-1001"
    assert ids[-1] == "ALM-1012"
    rec = res.records[2]
    assert rec.alarm_name == "Cell Down"
    assert rec.severity == "Critical"
    assert rec.category == "Radio"
    assert rec.module == "gNB-DU"


def test_header_alias_resolution(tmp_path):
    p = _write_xlsx(tmp_path, [
        ("Alarm Code", "Name", "Level", "Group", "Node"),
        ("alm-2001", "Test Alarm", "Major", "Radio", "gNB-DU"),
    ])
    res = parse_alarm_ref_excel(p)
    assert len(res.records) == 1
    assert res.records[0].alarm_id == "ALM-2001"


def test_required_field_missing_skipped(tmp_path):
    p = _write_xlsx(tmp_path, [
        ("Alarm ID", "Alarm Name", "Severity"),
        ("", "Orphan", "Critical"),
        ("ALM-3001", "", "Critical"),
        ("ALM-3002", "Good", "Major"),
    ])
    res = parse_alarm_ref_excel(p)
    assert len(res.records) == 1
    assert res.records[0].alarm_id == "ALM-3002"
    assert res.skipped_rows == 2


def test_to_keywords_excludes_blanks():
    rec = AlarmRecord(alarm_id="ALM-9", alarm_name="X")
    assert rec.to_keywords() == ["ALM-9", "X"]
    rec2 = AlarmRecord(alarm_id="ALM-9", alarm_name="X", severity="Major", module="RU")
    assert rec2.to_keywords() == ["ALM-9", "X", "Major", "RU"]


def test_to_chunk_text_format():
    rec = AlarmRecord(alarm_id="ALM-1", alarm_name="Cell Down",
                      severity="Critical", category="Radio", module="gNB-DU")
    txt = rec.to_chunk_text()
    assert "Alarm: ALM-1 — Cell Down" in txt
    assert "Severity: Critical" in txt
    assert "Module: gNB-DU" in txt
