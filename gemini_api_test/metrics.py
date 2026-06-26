"""Pure metric math: cumulative series, totals, ratio, cost estimate.

No network. Fully unit-testable.
"""

from __future__ import annotations

from gemini_client import PRICE_PER_TOKEN


def _cumulative(values: list[float]) -> list[float]:
    out, acc = [], 0.0
    for v in values:
        acc += v
        out.append(acc)
    return out


def _mode_series(records: list[dict], mode: str) -> dict:
    rows = [r for r in records if r["mode"] == mode]
    rows.sort(key=lambda r: r["turn"])
    tokens = [r["total_tokens"] for r in rows]
    prompt = [r["prompt_tokens"] for r in rows]
    wire = [r["wire_sent"] + r["wire_recv"] for r in rows]
    payload = [r["req_payload_bytes"] + r["resp_payload_bytes"] for r in rows]
    return {
        "turns": [r["turn"] for r in rows],
        "per_turn_tokens": tokens,
        "per_turn_prompt_tokens": prompt,
        "per_turn_wire_bytes": wire,
        "cum_tokens": _cumulative(tokens),
        "cum_prompt_tokens": _cumulative(prompt),
        "cum_wire_bytes": _cumulative(wire),
        "cum_payload_bytes": _cumulative(payload),
        "errors": [r["error"] for r in rows if r["error"]],
    }


def summarize(experiment: dict) -> dict:
    records = experiment["records"]
    stateless = _mode_series(records, "stateless")
    delta = _mode_series(records, "delta")

    def total(series, key):
        return series[key][-1] if series[key] else 0

    sl_tokens = total(stateless, "cum_tokens")
    dl_tokens = total(delta, "cum_tokens")
    sl_wire = total(stateless, "cum_wire_bytes")
    dl_wire = total(delta, "cum_wire_bytes")

    def ratio(a, b):
        return round(a / b, 2) if b else None

    summary = {
        "stateless": stateless,
        "delta": delta,
        "totals": {
            "stateless_tokens": sl_tokens,
            "delta_tokens": dl_tokens,
            "stateless_wire_bytes": sl_wire,
            "delta_wire_bytes": dl_wire,
            "token_ratio": ratio(sl_tokens, dl_tokens),
            "wire_ratio": ratio(sl_wire, dl_wire),
            "stateless_cost_usd": round(sl_tokens * PRICE_PER_TOKEN, 6),
            "delta_cost_usd": round(dl_tokens * PRICE_PER_TOKEN, 6),
            "price_per_token": PRICE_PER_TOKEN,
        },
    }
    return summary
