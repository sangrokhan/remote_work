#!/usr/bin/env python3
"""scripts/measure_perf_cache_savings.py -- multi-turn wire-byte savings
measurement for the engine Gateway request-body leaf-hash cache
(docs/engine_gateway_caching_seed.md), driven by records/perf.json.

Runs the SAME 20-turn perf.json conversation twice against the real
local-llm engine Gateway, over the real docker-compose network topology
(web -> Network Gateway L3/L4 netem -> engine Gateway L7 -> llama-server):

  1. baseline run: cache_enabled=False (X-AIPT-Cache header not sent,
     engine Gateway behaves as a transparent proxy -- this is exactly
     today's pre-caching wire cost).
  2. cached run: cache_enabled=True, same conversation, fresh TCP
     connection (fresh session cache both sides).

For each turn records req_payload_bytes and wire_sent (actual TCP bytes,
aipt.core.wire's socket-level counter) for both runs, computes the
per-turn delta, and writes one CSV row per turn plus a final summary row.

Intended to be run from INSIDE the `web` container (docker compose exec
web python3 scripts/measure_perf_cache_savings.py) so LOCAL_LLM_ENGINE_URL
resolves to the real net-backend engine Gateway per docker-compose.yml,
and so this exercises the real Network Gateway hop (verified separately
via `ip route`/`tc qdisc` in the prior session).
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, ".")

from aipt.backends.local_llm.engine_adapter import EngineAdapter
from aipt.backends.local_llm.gateway import Gateway
from aipt.backends.mock.records import load_scenario_record
from aipt.core import wire

RECORD_PATH = os.environ.get("PERF_RECORD_PATH", "records/perf.json")
OUT_CSV = os.environ.get("OUT_CSV", "/app/data/runs/cache_savings_multiturn.csv")
# Keep responses short -- this measures REQUEST-side wire bytes (what the
# caching protocol targets), not generation latency/length, and a real
# llama-server call still costs real prefill time per turn even with a
# tiny max_tokens.
MAX_TOKENS = int(os.environ.get("PERF_MAX_TOKENS", "8"))
CACHE_THRESHOLD_BYTES = int(os.environ.get("PERF_CACHE_THRESHOLD_BYTES", "200"))


def run_conversation(record, *, cache_enabled: bool) -> list[dict]:
    """Replays every turn of `record` as one growing multi-turn
    conversation against the real engine Gateway, returning one dict per
    turn with the measured wire numbers."""
    adapter = EngineAdapter(model="local-model")
    gw = Gateway(adapter, cache_enabled=cache_enabled,
                 cache_threshold_bytes=CACHE_THRESHOLD_BYTES)

    messages = []
    if record.system_prompt:
        messages.append({"role": "system", "content": record.system_prompt})

    wire.reset_session()  # fresh TCP connection -> fresh session on both sides

    rows = []
    for i, turn in enumerate(record.turns):
        messages.append({"role": "user", "content": turn.question})
        result = gw.send(messages, max_tokens=MAX_TOKENS)
        if result.error:
            raise RuntimeError(f"turn {i} failed ({'cached' if cache_enabled else 'baseline'}): {result.error}")
        if result.text:
            messages.append({"role": "assistant", "content": result.text})
        rows.append({
            "turn": i,
            "req_payload_bytes": result.req_payload_bytes,
            "wire_sent": result.wire_sent,
            "wire_recv": result.wire_recv,
            "status": result.status,
        })
    return rows


def main() -> None:
    record = load_scenario_record(RECORD_PATH)
    print(f"Loaded record {record.name!r}: {len(record.turns)} turns, "
          f"system_prompt={len(record.system_prompt.encode())} bytes")

    print("\n=== Running BASELINE (cache disabled) ===")
    baseline_rows = run_conversation(record, cache_enabled=False)

    print("=== Running CACHED (cache enabled) ===")
    cached_rows = run_conversation(record, cache_enabled=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = [
        "turn", "baseline_req_payload_bytes", "cached_req_payload_bytes",
        "req_payload_bytes_saved", "req_payload_bytes_saved_pct",
        "baseline_wire_sent", "cached_wire_sent",
        "wire_sent_saved", "wire_sent_saved_pct",
    ]
    total_baseline_payload = 0
    total_cached_payload = 0
    total_baseline_wire = 0
    total_cached_wire = 0

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b, c in zip(baseline_rows, cached_rows):
            payload_saved = b["req_payload_bytes"] - c["req_payload_bytes"]
            payload_pct = (payload_saved / b["req_payload_bytes"] * 100) if b["req_payload_bytes"] else 0.0
            wire_saved = b["wire_sent"] - c["wire_sent"]
            wire_pct = (wire_saved / b["wire_sent"] * 100) if b["wire_sent"] else 0.0
            writer.writerow({
                "turn": b["turn"],
                "baseline_req_payload_bytes": b["req_payload_bytes"],
                "cached_req_payload_bytes": c["req_payload_bytes"],
                "req_payload_bytes_saved": payload_saved,
                "req_payload_bytes_saved_pct": round(payload_pct, 1),
                "baseline_wire_sent": b["wire_sent"],
                "cached_wire_sent": c["wire_sent"],
                "wire_sent_saved": wire_saved,
                "wire_sent_saved_pct": round(wire_pct, 1),
            })
            total_baseline_payload += b["req_payload_bytes"]
            total_cached_payload += c["req_payload_bytes"]
            total_baseline_wire += b["wire_sent"]
            total_cached_wire += c["wire_sent"]

        writer.writerow({
            "turn": "TOTAL",
            "baseline_req_payload_bytes": total_baseline_payload,
            "cached_req_payload_bytes": total_cached_payload,
            "req_payload_bytes_saved": total_baseline_payload - total_cached_payload,
            "req_payload_bytes_saved_pct": round(
                (total_baseline_payload - total_cached_payload) / total_baseline_payload * 100, 1
            ) if total_baseline_payload else 0.0,
            "baseline_wire_sent": total_baseline_wire,
            "cached_wire_sent": total_cached_wire,
            "wire_sent_saved": total_baseline_wire - total_cached_wire,
            "wire_sent_saved_pct": round(
                (total_baseline_wire - total_cached_wire) / total_baseline_wire * 100, 1
            ) if total_baseline_wire else 0.0,
        })

    print(f"\nCSV written to {OUT_CSV}")
    print(f"\nTotal req payload bytes: baseline={total_baseline_payload} cached={total_cached_payload} "
          f"saved={total_baseline_payload - total_cached_payload} "
          f"({(total_baseline_payload - total_cached_payload) / total_baseline_payload * 100:.1f}%)")
    print(f"Total wire_sent bytes:  baseline={total_baseline_wire} cached={total_cached_wire} "
          f"saved={total_baseline_wire - total_cached_wire} "
          f"({(total_baseline_wire - total_cached_wire) / total_baseline_wire * 100:.1f}%)")


if __name__ == "__main__":
    main()
