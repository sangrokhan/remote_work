"""aipt.export.packets -- packets.csv from a pcap (DESIGN.md 4.6 B6).

No real capture is used: aipt.export.packets.write_pcap() builds a minimal
classic-pcap file by hand (module-level helper, kept in packets.py itself so
both the stdlib and dpkt read paths can be exercised without a real
tcpdump run -- see its docstring), and these tests round-trip through it.

Runs the stdlib fallback reader always; if dpkt is installed
(pip install aipt[export]) it also cross-checks that both readers agree on
the same fixture file, so neither path can silently drift from the other.
"""

from __future__ import annotations

import csv
import io

import pytest

from aipt.export import packets as packets_mod
from aipt.export.packets import (
    PACKET_COLUMNS,
    PcapFormatError,
    gap_confidence_summary,
    iter_packets,
    packets_csv,
    write_pcap,
)


def _fixture_pcap(tmp_path):
    path = tmp_path / "sample.pcap"
    write_pcap(path, [
        (1_700_000_000.000000, b"A" * 60),
        (1_700_000_000.002000, b"B" * 1400),
        (1_700_000_000.002500, b"C" * 40),
        (1_700_000_002.500000, b"D" * 1400),  # idle gap before this one
    ])
    return path


def test_write_pcap_then_iter_packets_roundtrip(tmp_path):
    path = _fixture_pcap(tmp_path)
    pkts = list(iter_packets(path))
    assert len(pkts) == 4
    assert pkts[0]["caplen"] == 60
    assert pkts[0]["wire_len"] == 60
    assert pkts[1]["caplen"] == 1400
    # timestamps monotonic and close to what was written
    assert pkts[0]["ts"] == pytest.approx(1_700_000_000.0, abs=1e-3)
    assert pkts[3]["ts"] == pytest.approx(1_700_000_002.5, abs=1e-3)


def test_write_pcap_truncates_to_snaplen(tmp_path):
    path = tmp_path / "trunc.pcap"
    write_pcap(path, [(1.0, b"X" * 200)], snaplen=64)
    pkts = list(iter_packets(path))
    assert len(pkts) == 1
    assert pkts[0]["caplen"] == 64
    assert pkts[0]["wire_len"] == 200


def test_packets_csv_header_matches_columns(tmp_path):
    text = packets_csv(tmp_path / "does_not_exist.pcap")
    header = text.splitlines()[0].split(",")
    assert header == PACKET_COLUMNS
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows == []


def test_packets_csv_missing_file_is_empty_not_error(tmp_path):
    # A run without capture (unavailable, or a mock run with no real
    # traffic) must not raise -- same "absent, not failure" contract as
    # aipt.core.capture / aipt.core.cwnd.
    text = packets_csv(tmp_path / "nope.pcap")
    assert text.splitlines()[0].split(",") == PACKET_COLUMNS


def test_packets_csv_first_row_has_no_gap(tmp_path):
    path = _fixture_pcap(tmp_path)
    text = packets_csv(path)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert len(rows) == 4
    assert rows[0]["gap_ms"] == ""  # no previous packet to gap from
    assert rows[0]["index"] == "0"


def test_packets_csv_inter_arrival_gap_ms(tmp_path):
    path = _fixture_pcap(tmp_path)
    text = packets_csv(path)
    rows = list(csv.DictReader(io.StringIO(text)))
    # packet 1 arrived 2ms after packet 0
    assert float(rows[1]["gap_ms"]) == pytest.approx(2.0, abs=0.05)
    # packet 3 arrived ~2.4975s after packet 2 -- the idle gap
    assert float(rows[3]["gap_ms"]) > 2000


def test_packets_csv_marks_truncated_packets(tmp_path):
    path = tmp_path / "trunc.pcap"
    write_pcap(path, [(1.0, b"X" * 200), (1.001, b"Y" * 30)], snaplen=64)
    text = packets_csv(path)
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["truncated"] == "True"
    assert rows[0]["wire_len"] == "200"
    assert rows[0]["caplen"] == "64"
    assert rows[1]["truncated"] == "False"


def test_packets_csv_sizes_present(tmp_path):
    path = _fixture_pcap(tmp_path)
    text = packets_csv(path)
    rows = list(csv.DictReader(io.StringIO(text)))
    sizes = [int(r["caplen"]) for r in rows]
    assert sizes == [60, 1400, 40, 1400]


def test_stdlib_reader_rejects_garbage_file(tmp_path):
    path = tmp_path / "notapcap.bin"
    path.write_bytes(b"not a pcap file at all, just bytes")
    with pytest.raises(PcapFormatError):
        list(packets_mod._iter_packets_stdlib(path))


@pytest.mark.skipif(packets_mod.dpkt is None, reason="dpkt not installed")
def test_dpkt_and_stdlib_readers_agree(tmp_path):
    path = _fixture_pcap(tmp_path)
    stdlib_pkts = list(packets_mod._iter_packets_stdlib(path))
    dpkt_pkts = list(packets_mod._iter_packets_dpkt(path))
    assert len(stdlib_pkts) == len(dpkt_pkts) == 4
    for a, b in zip(stdlib_pkts, dpkt_pkts):
        assert a[0] == pytest.approx(b[0], abs=1e-6)  # ts
        assert a[1] == b[1]  # caplen
        assert a[2] == b[2]  # wire_len


@pytest.mark.skipif(packets_mod.dpkt is None, reason="dpkt not installed")
def test_dpkt_reader_reports_truncated_wire_len(tmp_path):
    path = tmp_path / "trunc.pcap"
    write_pcap(path, [(1.0, b"X" * 200)], snaplen=64)
    pkts = list(packets_mod._iter_packets_dpkt(path))
    assert pkts[0][1] == 64   # caplen
    assert pkts[0][2] == 200  # wire_len -- must differ, not silently equal caplen


# --- B13: gap_confidence_summary (packets.csv schema stays untouched) ------

def _sub_ms_gap_pcap(tmp_path):
    # gaps of 0.2ms and 0.3ms -- well under the 1ms short-gap threshold.
    path = tmp_path / "tight.pcap"
    write_pcap(path, [
        (1_700_000_000.0000, b"A" * 60),
        (1_700_000_000.0002, b"B" * 60),
        (1_700_000_000.0005, b"C" * 60),
    ])
    return path


def test_packets_csv_schema_is_unchanged_by_b13():
    # The header this asserts is the contract other tooling reads; B13 must
    # not add columns to it.
    assert PACKET_COLUMNS == [
        "index", "ts", "ts_ms", "gap_ms", "caplen", "wire_len", "truncated",
    ]


def test_gap_confidence_summary_warns_on_short_gaps_with_software_timestamps(tmp_path):
    path = _sub_ms_gap_pcap(tmp_path)
    ts_source = {"iface": "eth0", "available": True, "hardware_timestamping": False}
    summary = gap_confidence_summary(path, ts_source)
    assert summary["median_gap_ms"] == pytest.approx(0.25, abs=0.01)
    assert summary["hardware_timestamping"] is False
    assert "software-timestamped" in summary["timestamp_precision_reason"]


def test_gap_confidence_summary_is_silent_with_hardware_timestamps(tmp_path):
    path = _sub_ms_gap_pcap(tmp_path)
    ts_source = {"iface": "eth0", "available": True, "hardware_timestamping": True}
    summary = gap_confidence_summary(path, ts_source)
    assert summary["hardware_timestamping"] is True
    assert summary["timestamp_precision_reason"] == ""


def test_gap_confidence_summary_is_silent_when_gaps_are_not_short(tmp_path):
    path = _fixture_pcap(tmp_path)  # has a >2s idle gap, not sub-ms throughout
    ts_source = {"iface": "eth0", "available": True, "hardware_timestamping": False}
    summary = gap_confidence_summary(path, ts_source)
    # median of [2.0, 0.5, 2497.5] ms is 2.0ms -- above the 1ms threshold.
    assert summary["median_gap_ms"] > 1.0
    assert summary["timestamp_precision_reason"] == ""


def test_gap_confidence_summary_flags_unknown_timestamp_source(tmp_path):
    path = _sub_ms_gap_pcap(tmp_path)
    summary = gap_confidence_summary(path, None)
    assert summary["hardware_timestamping"] is None
    assert "unknown" in summary["timestamp_precision_reason"]


def test_gap_confidence_summary_handles_missing_or_tiny_pcap(tmp_path):
    summary = gap_confidence_summary(tmp_path / "nope.pcap", None)
    assert summary["median_gap_ms"] is None
    assert summary["timestamp_precision_reason"] == ""

    one_packet = tmp_path / "one.pcap"
    write_pcap(one_packet, [(1.0, b"A" * 10)])
    summary_one = gap_confidence_summary(one_packet, None)
    assert summary_one["median_gap_ms"] is None
