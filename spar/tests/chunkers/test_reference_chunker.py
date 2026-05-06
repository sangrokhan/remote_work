from __future__ import annotations

from spar.chunkers.reference_chunker import chunk_alarm_ref, chunk_counter_ref, chunk_parameter_ref
from spar.parsers.alarm_ref_parser import AlarmRecord
from spar.parsers.counter_ref_parser import CounterRecord
from spar.parsers.parameter_ref_parser import ParameterRecord


def _make_param() -> ParameterRecord:
    return ParameterRecord(
        param_name="nrDlCellMaxTxPower",
        yang_path="NRCellDU/nrDlCellMaxTxPower",
        feature_name="MIMO",
        type="int32",
        default="23",
        min="0",
        max="40",
        description="Maximum downlink transmission power",
    )


def _make_counter() -> CounterRecord:
    return CounterRecord(
        large_group="RRC",
        mid_group="Connection",
        mid_group_id="RRC.01",
        counter_name="pmRrcConnEstabAtt",
        description="RRC connection establishment attempts",
        period="15min",
        unit="count",
        min_val="0",
        max_val="",
    )


def _make_alarm() -> AlarmRecord:
    return AlarmRecord(
        alarm_id="A0010001R",
        alarm_name="Cell Unavailable",
        severity="Critical",
        group="Hardware",
    )


def test_chunk_parameter_ref_one_record_one_chunk():
    chunks = chunk_parameter_ref([_make_param()], source_doc="param_ref_v6.xlsx")
    assert len(chunks) == 1
    c = chunks[0]
    assert "nrDlCellMaxTxPower" in c["text"]
    assert c["doc_type"] == "parameter_ref"
    assert c["source_doc"] == "param_ref_v6.xlsx"
    assert c["mo_name"] == "nrDlCellMaxTxPower"  # leaf_mo returns parts[-1] of yang_path
    assert c["yang_path"] == "NRCellDU/nrDlCellMaxTxPower"
    assert c["keywords"] == []


def test_chunk_parameter_ref_empty_returns_empty():
    assert chunk_parameter_ref([], source_doc="x.xlsx") == []


def test_chunk_parameter_ref_multiple_records():
    r1 = _make_param()
    r2 = ParameterRecord(param_name="rachRootSequenceIndex", yang_path="NRCellDU/rachRootSequenceIndex")
    chunks = chunk_parameter_ref([r1, r2], source_doc="param.xlsx")
    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]


def test_chunk_counter_ref_one_record_one_chunk():
    chunks = chunk_counter_ref([_make_counter()], source_doc="counter_ref_v6.xlsx")
    assert len(chunks) == 1
    c = chunks[0]
    assert "pmRrcConnEstabAtt" in c["text"]
    assert c["doc_type"] == "counter_ref"
    assert c["mo_name"] == "Connection"
    assert c["section"] == "RRC"
    assert c["keywords"] == []


def test_chunk_counter_ref_empty_returns_empty():
    assert chunk_counter_ref([], source_doc="x.xlsx") == []


def test_chunk_alarm_ref_one_record_one_chunk():
    chunks = chunk_alarm_ref([_make_alarm()], source_doc="alarm_ref_v6.xlsx")
    assert len(chunks) == 1
    c = chunks[0]
    assert "4050" in c["text"] or "Cell Unavailable" in c["text"]
    assert c["doc_type"] == "alarm_ref"
    assert c["source_doc"] == "alarm_ref_v6.xlsx"
    assert c["keywords"] == []


def test_chunk_alarm_ref_empty_returns_empty():
    assert chunk_alarm_ref([], source_doc="x.xlsx") == []


def test_all_chunks_have_required_fields():
    param_chunks = chunk_parameter_ref([_make_param()], "p.xlsx")
    counter_chunks = chunk_counter_ref([_make_counter()], "c.xlsx")
    alarm_chunks = chunk_alarm_ref([_make_alarm()], "a.xlsx")

    for chunks in (param_chunks, counter_chunks, alarm_chunks):
        for c in chunks:
            for field in ("chunk_id", "text", "doc_type", "source_doc", "keywords", "parent_sections"):
                assert field in c, f"missing field: {field}"
