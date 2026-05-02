from __future__ import annotations

import pytest

from spar.ingest.chunkers import dispatch_records
from spar.parsers.alarm_ref_parser import AlarmRecord
from spar.parsers.counter_ref_parser import CounterRecord
from spar.parsers.parameter_ref_parser import ParameterRecord


def test_dispatch_records_parameter_ref():
    records = [ParameterRecord(param_name="nrDlCellMaxTxPower", yang_path="NRCellDU/nrDlCellMaxTxPower")]
    chunks = dispatch_records(records, "param.xlsx", doc_type="parameter_ref")
    assert len(chunks) == 1
    assert chunks[0]["doc_type"] == "parameter_ref"
    assert "nrDlCellMaxTxPower" in chunks[0]["text"]


def test_dispatch_records_counter_ref():
    records = [CounterRecord(large_group="RRC", mid_group="Connection", mid_group_id="RRC.01", counter_name="pmRrcConnEstabAtt")]
    chunks = dispatch_records(records, "counter.xlsx", doc_type="counter_ref")
    assert len(chunks) == 1
    assert chunks[0]["doc_type"] == "counter_ref"


def test_dispatch_records_alarm_ref():
    records = [AlarmRecord(alarm_id="4050", alarm_name="Cell Unavailable")]
    chunks = dispatch_records(records, "alarm.xlsx", doc_type="alarm_ref")
    assert len(chunks) == 1
    assert chunks[0]["doc_type"] == "alarm_ref"


def test_dispatch_records_unknown_type_raises():
    with pytest.raises(ValueError, match="unsupported doc_type"):
        dispatch_records([], "x.xlsx", doc_type="unknown_type")
