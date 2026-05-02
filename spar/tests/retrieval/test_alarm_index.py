from pathlib import Path

import pytest

from spar.parsers.alarm_ref_parser import AlarmRecord
from spar.retrieval import alarm_index as ai_mod
from spar.retrieval.alarm_index import AlarmIndex, get_alarm_index

SAMPLE = Path("data/samples/alarm_excel_ref_sample.xlsx")


@pytest.fixture(autouse=True)
def _reset_singleton():
    ai_mod._SINGLETON = None
    yield
    ai_mod._SINGLETON = None


def test_lookup_hit():
    idx = AlarmIndex([
        AlarmRecord("ALM-1", "A"),
        AlarmRecord("ALM-2", "B"),
    ])
    rec = idx.lookup("ALM-2")
    assert rec is not None
    assert rec.alarm_name == "B"


def test_lookup_miss():
    idx = AlarmIndex([AlarmRecord("ALM-1", "A")])
    assert idx.lookup("ALM-999") is None


def test_lookup_case_insensitive():
    idx = AlarmIndex([AlarmRecord("ALM-1", "A")])
    assert idx.lookup("alm-1") is not None
    assert idx.lookup("Alm-1") is not None


def test_search_by_name_partial():
    idx = AlarmIndex([
        AlarmRecord("ALM-1", "Cell Down"),
        AlarmRecord("ALM-2", "Link Down"),
        AlarmRecord("ALM-3", "Fan Failure"),
    ])
    hits = idx.search_by_name("down")
    assert len(hits) == 2
    assert {r.alarm_id for r in hits} == {"ALM-1", "ALM-2"}


def test_singleton_loads_default_sample():
    idx = get_alarm_index()
    assert len(idx) == 12
    assert idx.lookup("ALM-1003").alarm_name == "Cell Down"


def test_singleton_caches():
    idx1 = get_alarm_index()
    idx2 = get_alarm_index()
    assert idx1 is idx2


def test_env_override(monkeypatch):
    monkeypatch.setenv("SPAR_ALARM_REF_PATH", str(SAMPLE))
    idx = get_alarm_index()
    assert len(idx) == 12
