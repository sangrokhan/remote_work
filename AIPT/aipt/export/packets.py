"""packets.py -- ``packets.csv``: layer 3 of DESIGN.md 4.6's 3-layer export
set (B6, "완전 신규" -- nothing before this parsed the pcaps AIPT already
writes; ``aipt.core.capture`` only recorded that a file exists).

A byte count taken inside the process is a claim about what the code
believes it sent; the pcap on disk is the thing the claim is about (see
``aipt/core/capture.py``'s module docstring). This module reads that pcap
back and turns it into the one thing a spreadsheet can plot: packet
arrival time, inter-arrival gap, and packet size -- the same picture
``connection.py``'s ``snd_cwnd``-vs-``t_ms`` series is, at the packet level
instead of the tick level.

Parser: `dpkt` when importable (``pip install dpkt``, an optional
dependency -- see ``pyproject.toml``'s ``[project.optional-dependencies]
export`` group), falling back to a small pure-stdlib reader for the classic
(non-pcapng) ``.pcap`` format that ``tcpdump -w`` and this project's own
``core.capture`` always produce. The fallback exists so a checkout that
skipped the optional dependency (or is offline) still gets packets.csv --
DESIGN.md 4.6 says pcap parsing is "완전 신규", not "requires a new hard
dependency to run any test in this package".

Both readers only need three things per packet -- the capture timestamp and
the two length fields every pcap record header carries (``caplen``: bytes
actually captured, possibly truncated by ``snaplen``; ``len``: the packet's
original length on the wire) -- so the fallback need not understand
Ethernet/IP/TCP at all. It reads record headers and skips the payload,
which is also why it never needs a link-layer type table: sizes and timing
are exactly what ``docs/outputs.md``'s pcap section says the file is for,
and both survive an unparsed payload.
"""

from __future__ import annotations

import csv
import io
import struct
from pathlib import Path

try:
    import dpkt  # type: ignore
except ImportError:  # pragma: no cover - exercised only without the optional dep
    dpkt = None  # type: ignore

PACKET_COLUMNS = [
    "index", "ts", "ts_ms", "gap_ms", "caplen", "wire_len", "truncated",
]

# Classic pcap global-header magic numbers: microsecond and nanosecond
# resolution, each in both byte orders (a capture written on a big-endian
# box is a possibility, however remote, and getting the header wrong turns
# every timestamp and length in the file into noise instead of an error).
_MAGIC_LE_US = 0xA1B2C3D4
_MAGIC_BE_US = 0xD4C3B2A1
_MAGIC_LE_NS = 0xA1B23C4D
_MAGIC_BE_NS = 0x4D3CB2A1


class PcapFormatError(ValueError):
    """The file is not a classic (non-pcapng) pcap this reader understands.

    Raised rather than silently returning zero packets: a caller asking for
    packets.csv on a pcapng file (dpkt not installed, so the pcapng-aware
    path in ``dpkt`` isn't available either) needs to know its file is the
    wrong format, not that the capture was empty.
    """


def _iter_packets_stdlib(path: Path):
    """Yield (ts_seconds: float, caplen: int, wire_len: int) for a classic
    pcap file, using only ``struct`` -- no third-party parser at all.

    Deliberately minimal: this reads exactly the two headers a pcap file
    has (one global, one per record) and nothing past the record length,
    because inter-arrival gap and packet size are all layer 3 needs. It is
    not a general pcap library and does not try to be one.
    """
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if len(magic) < 4:
            raise PcapFormatError(f"{path}: file too short to be a pcap")
        magic_int = struct.unpack("<I", magic)[0]
        if magic_int in (_MAGIC_LE_US, _MAGIC_LE_NS):
            endian = "<"
        elif magic_int in (_MAGIC_BE_US, _MAGIC_BE_NS):
            endian = ">"
            magic_int = struct.unpack(">I", magic)[0]
        else:
            raise PcapFormatError(
                f"{path}: unrecognized magic {magic!r} -- not a classic "
                f"pcap (pcapng needs dpkt; 'pip install dpkt' or "
                f"'pip install aipt[export]')")
        nanosecond = magic_int in (_MAGIC_LE_NS, _MAGIC_BE_NS)

        header = fh.read(20)  # version_major/minor, thiszone, sigfigs, snaplen, network
        if len(header) < 20:
            raise PcapFormatError(f"{path}: truncated global header")

        divisor = 1_000_000_000.0 if nanosecond else 1_000_000.0
        index = 0
        while True:
            rec_header = fh.read(16)
            if not rec_header:
                break
            if len(rec_header) < 16:
                # A record header cut off mid-write (a killed capture).
                # Reported by the caller as a truncated file, not raised --
                # the packets already read are still real packets.
                break
            ts_sec, ts_frac, caplen, wire_len = struct.unpack(f"{endian}IIII", rec_header)
            ts = ts_sec + (ts_frac / divisor)
            payload = fh.read(caplen)
            if len(payload) < caplen:
                break
            yield ts, caplen, wire_len
            index += 1


def _iter_packets_dpkt(path: Path):
    """Same yield shape as :func:`_iter_packets_stdlib`, via ``dpkt``.

    Used when available because it is a maintained, well-tested parser
    (and the natural place to grow pcapng support later); the stdlib path
    above exists so this module works without it.

    ``dpkt.pcap.Reader.__iter__`` only yields ``(ts, buf)`` where ``buf``
    is the *captured* bytes -- it reads ``hdr.len`` (the original on-wire
    length) off the record header but never hands it back to the caller,
    so a truncated packet would otherwise look identical to one that was
    never bigger than ``caplen`` in the first place. This reads the record
    header directly (via dpkt's own ``PktHdr``/``LEPktHdr``/``BEPktHdr``
    classes, so the byte layout stays in one place) to recover ``hdr.len``
    alongside what ``Reader`` already gives us.
    """
    with open(path, "rb") as fh:
        file_header = dpkt.pcap.FileHdr(fh.read(dpkt.pcap.FileHdr.__hdr_len__))  # type: ignore[union-attr]
        magic = file_header.magic
        if magic not in dpkt.pcap.MAGIC_TO_PKT_HDR:  # type: ignore[union-attr]
            raise PcapFormatError(f"{path}: dpkt does not recognize magic {magic!r}")
        pkt_hdr_cls = dpkt.pcap.MAGIC_TO_PKT_HDR[magic]  # type: ignore[union-attr]
        while True:
            raw_header = fh.read(pkt_hdr_cls.__hdr_len__)
            if not raw_header:
                break
            if len(raw_header) < pkt_hdr_cls.__hdr_len__:
                break
            header = pkt_hdr_cls(raw_header)
            payload = fh.read(header.caplen)
            if len(payload) < header.caplen:
                break
            yield float(header.tv_sec) + (header.tv_usec / 1_000_000.0), header.caplen, header.len


def iter_packets(pcap_path: str | Path):
    """Yield one dict per packet: ``ts`` (epoch seconds, float), ``caplen``
    (bytes actually captured), ``wire_len`` (the packet's original length --
    equal to ``caplen`` unless a snaplen truncated it in flight).

    Packets come back in file order, which is capture order: a pcap is
    written as each packet arrives, so this is also arrival order.
    """
    path = Path(pcap_path)
    source = _iter_packets_dpkt(path) if dpkt is not None else _iter_packets_stdlib(path)
    for ts, caplen, wire_len in source:
        yield {"ts": ts, "caplen": caplen, "wire_len": wire_len}


def packets_csv(pcap_path: str | Path) -> str:
    """The packets in one pcap, one row per packet: arrival time,
    inter-arrival gap from the previous packet in the file, and size.

    ``gap_ms`` is ``"" `` (not ``0``) on the first packet -- there is no
    previous packet to gap from, and a ``0`` would read as "arrived
    instantly after the previous one" instead of "there was no previous
    one", the same "absent must not read as zero" rule
    ``aipt/backends/record.py``'s ``store_tail_ms`` follows.

    ``truncated`` is ``true`` when ``caplen < wire_len`` -- the capture's
    ``snaplen`` cut the packet short (see ``aipt/core/capture.py``'s
    ``PCAP_SNAPLEN``) -- so a reader can tell "this packet was smaller"
    from "this packet was recorded smaller than it was".

    A pcap that does not exist, or exists with zero packets (a mock run
    that produced no real traffic -- see ``docs/outputs.md``'s note on mock
    captures), yields a header with no rows.

    This function's output columns are a stable schema other tooling reads
    (see tests/export/test_packets.py's header assertions) -- DESIGN.md
    4.9's B13 timestamp-precision signal is deliberately *not* a column
    here; see :func:`gap_confidence_summary` for that instead, so nothing
    that already parses this CSV's columns/order breaks.
    """
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=PACKET_COLUMNS, extrasaction="ignore")
    w.writeheader()

    path = Path(pcap_path)
    if not path.exists():
        return buf.getvalue()

    prev_ts: float | None = None
    for index, pkt in enumerate(iter_packets(path)):
        ts = pkt["ts"]
        gap_ms = "" if prev_ts is None else round((ts - prev_ts) * 1000.0, 3)
        w.writerow({
            "index": index,
            "ts": round(ts, 6),
            "ts_ms": round(ts * 1000.0, 3),
            "gap_ms": gap_ms,
            "caplen": pkt["caplen"],
            "wire_len": pkt["wire_len"],
            "truncated": pkt["caplen"] < pkt["wire_len"],
        })
        prev_ts = ts
    return buf.getvalue()


# Below this, a gap under this many ms is short enough that software
# timestamp jitter (typically low tens of microseconds, but not bounded)
# is no longer obviously negligible next to the thing being measured --
# see DESIGN.md 4.9's problem statement: MockBackend/LocalLLMBackend RTTs
# can fall to sub-millisecond, where the old "RTT is always multi-ms
# internet" assumption stops holding.
_SHORT_GAP_MS_THRESHOLD = 1.0


def gap_confidence_summary(pcap_path: str | Path,
                            timestamp_source: dict | None = None) -> dict:
    """DESIGN.md 4.9's B13: is packets.csv's ``gap_ms`` trustworthy here?

    A separate summary dict, not new packets.csv columns -- packets.csv's
    schema is read by other tooling (see test_packets.py's
    ``test_packets_csv_header_matches_columns``), and B13's own note says
    this is a "짧은 RTT 경로에서 ... 신뢰도 판단 근거", a per-capture
    verdict, not a per-packet one. ``timestamp_source`` is the dict
    :func:`aipt.core.capture.timestamp_source` returns (or ``None`` if the
    caller has none, e.g. no capture ran) -- kept as a parameter rather than
    importing ``aipt.core.capture`` here, so this module (which parses
    pcaps) stays independent of that one (which runs tcpdump).

    Returns ``median_gap_ms`` (``None`` if fewer than 2 packets),
    ``hardware_timestamping`` (``None`` if unknown), and
    ``timestamp_precision_reason``: empty when there is nothing to warn
    about (hardware timestamps, or gaps not short enough to matter, or not
    enough packets to judge), else a sentence a reader can act on.
    """
    gaps: list[float] = []
    prev_ts: float | None = None
    path = Path(pcap_path)
    if path.exists():
        for pkt in iter_packets(path):
            ts = pkt["ts"]
            if prev_ts is not None:
                gaps.append((ts - prev_ts) * 1000.0)
            prev_ts = ts

    if len(gaps) < 1:
        median_gap_ms = None
    else:
        s = sorted(gaps)
        n = len(s)
        median_gap_ms = (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0)

    hardware = None
    if timestamp_source is not None and timestamp_source.get("available"):
        hardware = bool(timestamp_source.get("hardware_timestamping"))

    reason = ""
    if median_gap_ms is not None and median_gap_ms < _SHORT_GAP_MS_THRESHOLD:
        if hardware is False:
            iface_name = (timestamp_source or {}).get("iface", "?")
            reason = (
                f"median inter-arrival gap is {median_gap_ms:.3f}ms on a "
                f"software-timestamped interface "
                f"({iface_name}) -- kernel/GIL "
                f"timestamp jitter is not necessarily smaller than this gap, "
                f"so gap_ms values this small should not be read as precise"
            )
        elif hardware is None:
            reason = (
                f"median inter-arrival gap is {median_gap_ms:.3f}ms and the "
                f"interface's timestamp source is unknown (ethtool "
                f"unavailable) -- cannot rule out software-timestamp jitter "
                f"at this gap size"
            )
        # hardware is True: no warning, the NIC clock does not have this problem.

    return {
        "median_gap_ms": median_gap_ms,
        "hardware_timestamping": hardware,
        "timestamp_precision_reason": reason,
    }


# -- test/fixture support ---------------------------------------------------
#
# A minimal classic-pcap *writer*, kept here (not under tests/) so both the
# stdlib and dpkt read paths can be exercised without a real capture -- the
# task's "no real pcap file" fixture requirement. Writing valid pcap bytes
# by hand also means the round-trip test proves this module's own
# understanding of the format, not just that dpkt agrees with itself.

def write_pcap(path: str | Path, packets: list[tuple[float, bytes]],
                snaplen: int = 65535) -> None:
    """Write a minimal classic pcap (microsecond resolution, Ethernet
    linktype) containing ``packets`` as ``(timestamp_seconds, payload)``
    pairs. Payloads longer than ``snaplen`` are truncated on write, exactly
    as a real capture would, so a round-trip test can also cover
    ``truncated``.
    """
    with open(path, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", _MAGIC_LE_US, 2, 4, 0, 0, snaplen, 1))
        for ts, payload in packets:
            wire_len = len(payload)
            data = payload[:snaplen]
            ts_sec = int(ts)
            ts_frac = int(round((ts - ts_sec) * 1_000_000))
            fh.write(struct.pack("<IIII", ts_sec, ts_frac, len(data), wire_len))
            fh.write(data)
