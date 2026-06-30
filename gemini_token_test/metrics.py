"""Pure metric math for a single-mode run: cumulative series, totals, cost.

No network. Fully unit-testable. Cross-mode comparison is done by loading two
executions from history and overlaying their series (not here).
"""

from __future__ import annotations

from gemini_client import PRICE_PER_TOKEN


def _cumulative(values: list[float]) -> list[float]:
    out, acc = [], 0.0
    for v in values:
        acc += v
        out.append(acc)
    return out


def _series(records: list[dict]) -> dict:
    rows = sorted(records, key=lambda r: r["turn"])
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


def summarize_three_stage(experiment: dict) -> dict:
    """Stateless vs stateful series (cumulative wire + content) for the 3-stage
    caching run, plus traffic-ratio totals."""
    sl = _series(experiment["stateless_records"])
    sf = _series(experiment["stateful_records"])
    last2 = lambda s, k: s[k][-1] if s[k] else 0
    sl_wire, sf_wire = last2(sl, "cum_wire_bytes"), last2(sf, "cum_wire_bytes")
    sl_cont, sf_cont = last2(sl, "cum_payload_bytes"), last2(sf, "cum_payload_bytes")
    cached = sum(r.get("cached_tokens", 0) for r in experiment["stateful_records"])
    used = sum(1 for r in experiment["stateful_records"] if r.get("used_cache"))
    ratio = lambda a, b: round(a / b, 2) if b else None
    return {
        "mode": "caching-3stage",
        "stateless_series": sl,
        "stateful_series": sf,
        "totals": {
            "mode": "caching-3stage",
            "stateless_wire": sl_wire, "stateful_wire": sf_wire,
            "wire_ratio": ratio(sl_wire, sf_wire),
            "stateless_content": sl_cont, "stateful_content": sf_cont,
            "content_ratio": ratio(sl_cont, sf_cont),
            "cached_tokens": cached, "caches_used": used,
        },
    }


def summarize(experiment: dict) -> dict:
    mode = experiment["params"].get("mode", "stateless")
    series = _series(experiment["records"])
    last = lambda k: series[k][-1] if series[k] else 0
    tokens = last("cum_tokens")
    wire = last("cum_wire_bytes")
    return {
        "mode": mode,
        "series": series,
        "totals": {
            "mode": mode,
            "tokens": tokens,
            "wire_bytes": wire,
            "cost_usd": round(tokens * PRICE_PER_TOKEN, 6),
            "price_per_token": PRICE_PER_TOKEN,
        },
    }
