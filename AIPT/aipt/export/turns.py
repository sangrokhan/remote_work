"""turns.py -- ``turns.csv``: layer 2 of DESIGN.md 4.6's 3-layer export set.

Rows come from ``aipt.backends.record.turn_record()`` (see
``aipt/backends/record.py``) -- the one place all three backends
(``public_ai``, ``mock``, ``local_llm``) build a turn row, so a chart that
puts a Gemini turn next to a local-llama turn or a mock replay is only
honest if the columns mean the same thing regardless of which backend
produced them.

The column set merges the two ancestors' schemas rather than picking one:

  * ``token_traffic/core/export.py``'s ``RECORD_COLUMNS`` (per-record:
    provider/arm/phase/turn/measure, wire/payload bytes, the five latency
    marks + ``store_tail_ms``, the token breakdown, ``error``) --
    provider is renamed ``backend`` per ``aipt/backends/record.py``.
  * ``tcp_congestion/tcp_congestion/export.py``'s ``turns_csv`` (per-turn:
    ``prompt_bytes``, ``request_ms``, ``idle_ms``, and a probe RTT summary)
    -- these only make sense for a mock/synthetic conversation with idle
    probing, so they ride as optional columns that are blank/0 for a
    backend that never populates them (``turn_record()``'s ``extra`` dict is
    how a backend supplies them; see ``send_turn`` callers).

New in this merge, not in either ancestor:

  ``goodput_bps``  DESIGN.md 4.6 B7. ``wire_recv`` (falling back to
                   ``resp_payload_bytes`` when wire-level accounting isn't
                   available -- a mock/local backend that never wraps a real
                   socket) divided by the ``req_sent_ms``..``turn_end_ms``
                   window, in bits per second. ``turn_record()`` leaves this
                   at ``0.0`` because computing it needs this module's byte
                   and window conventions, not just one exchange in
                   isolation (see its docstring) -- ``turns_csv`` below is
                   the only place that fills it in for real.
  ``transport``    DESIGN.md 4.5 B5's extension slot (``"http1"``/``"http3"``),
                   already on every ``turn_record()`` row.
  ``cache_bytes_saved``  local_llm-only (docs/engine_gateway_caching_seed.md):
                   bytes this turn's request payload was smaller than an
                   uncached-equivalent request would have been, from the
                   engine Gateway leaf-hash dedup protocol. 0 for every
                   other backend and for local_llm runs with caching off.
"""

from __future__ import annotations

import csv
import io

# Columns every backend fills in via aipt.backends.record.turn_record().
_CORE_COLUMNS = [
    "schema_version", "backend", "arm", "phase", "turn", "measure", "transport",
    "wire_sent", "wire_recv", "req_payload_bytes", "resp_payload_bytes",
    "req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms", "store_tail_ms",
    "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens",
    "goodput_bps",
    "cache_bytes_saved",
    "error",
]

# Columns that only ever came from the tcp_congestion side (synthetic_mock
# conversations with an idle-gap RTT probe). Optional: a backend/record that
# never populates them via turn_record()'s `extra` dict gets "" / 0, not a
# missing column -- the merged schema is one CSV shape for all three
# backends, not "whichever fields this row happens to have".
_OPTIONAL_COLUMNS = [
    "prompt_bytes", "request_ms", "idle_ms",
    "probe_count", "probe_rtt_mean_ms", "probe_rtt_min_ms", "probe_rtt_max_ms",
]

TURN_COLUMNS = [*_CORE_COLUMNS, *_OPTIONAL_COLUMNS]

# request_raw/response_raw/question/response_text are deliberately excluded
# (extrasaction="ignore" below just drops them): they are the evidence and
# belong in the run's JSON, not a spreadsheet cell -- same rule
# token_traffic/core/export.py states in its module docstring.


def goodput_bps(record: dict) -> float:
    """(bytes actually on the wire) / (turn_end_ms - req_sent_ms), in bits/s.

    DESIGN.md 4.6 B7: "기존 wire_sent/recv + 마크(req_sent_ms~turn_end_ms)로
    턴별 goodput 산출". Byte source is ``wire_recv`` when present and
    positive -- the socket-level count, headers and framing included, is the
    truer "goodput" measurement -- falling back to ``resp_payload_bytes``
    (the decoded body) for a backend that has no wire-level counter at all
    (a mock or local-llm backend not wrapping a real socket).

    Division-by-zero is guarded two ways, both returning ``0.0`` rather than
    raising: a non-positive window (``turn_end_ms <= req_sent_ms``, which
    happens on a record with no marks taken, e.g. a bytes-only measure pass)
    means the window itself is not measured, and a record with an error may
    have both marks pinned to the same instant. A ``0.0`` here reads as "not
    measured", matching how the underlying marks already read on a
    ``measure=bytes`` row (see ``docs/outputs.md``).
    """
    window_ms = int(record.get("turn_end_ms") or 0) - int(record.get("req_sent_ms") or 0)
    if window_ms <= 0:
        return 0.0
    wire_recv = int(record.get("wire_recv") or 0)
    payload_bytes = wire_recv if wire_recv > 0 else int(record.get("resp_payload_bytes") or 0)
    if payload_bytes <= 0:
        return 0.0
    return round((payload_bytes * 8) / (window_ms / 1000.0), 3)


def turns_csv(records: list[dict]) -> str:
    """One row per turn record (prep rows included, phased not dropped --
    same rule as ``token_traffic/core/export.py``'s ``records_csv``: a
    reader who wants only the steady turns can filter, but a reader never
    shown the prep call cannot discover what the arm paid before its first
    question).

    ``records`` is a list of ``aipt.backends.record.turn_record()`` dicts
    (or plain dicts with the same keys, e.g. read back from a saved run's
    JSON). ``goodput_bps`` is recomputed here from each record's own bytes
    and marks -- rather than trusting whatever ``turn_record()`` happened to
    leave at construction time (always ``0.0``, see its docstring) -- so the
    CSV and any programmatic reader of the same records agree.
    """
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TURN_COLUMNS, extrasaction="ignore")
    w.writeheader()
    for rec in records or []:
        row = {c: rec.get(c, "") for c in TURN_COLUMNS}
        row["goodput_bps"] = goodput_bps(rec)
        w.writerow(row)
    return buf.getvalue()
