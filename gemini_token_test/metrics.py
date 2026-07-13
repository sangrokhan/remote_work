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


def _latency_stats(values: list[int]) -> dict:
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"mean": round(sum(s) / n, 1), "median": median,
            "min": s[0], "max": s[-1]}


def summarize_comparison(comparison: dict) -> dict:
    """Per-arm series and totals for a run_comparison result.

    Keeps wire bytes and input tokens on separate axes, and keeps the cache-build
    setup cost visible: an arm's cumulative series is offset by its setup cost, so
    the chart shows the true bytes/tokens spent to have run k turns that way, and a
    front-loaded cache does not look free.
    """
    records = comparison["records"]
    arms = comparison["params"].get("arms") or []
    if not arms:
        arms = list(dict.fromkeys(r["arm"] for r in records))
    wall = comparison.get("wall_ms") or {}

    series: dict = {}
    totals: dict = {}
    for arm in arms:
        # cachegen is preparation, not the thing being measured: the cached arm's
        # caches are built off the stateless transcript before any measured turn
        # runs. Counting those uploads would drown every other number -- each build
        # re-sends the whole system prompt, so n turns cost O(n^2) -- and would
        # compare an arm's setup against another arm's traffic. Reported separately
        # below, never folded into the totals.
        gen = [r for r in records if r["arm"] == arm and r["phase"] == "cachegen"]
        steady = sorted((r for r in records if r["arm"] == arm and r["phase"] == "steady"),
                        key=lambda r: r["turn"])

        per_wire = [r["wire_sent"] + r["wire_recv"] for r in steady]
        per_in = [r["input_tokens"] for r in steady]

        series[arm] = {
            "turns": [r["turn"] for r in steady],
            "per_turn_wire": per_wire,
            "per_turn_input_tokens": per_in,
            "cum_wire": _cumulative(per_wire),
            "cum_input_tokens": _cumulative(per_in),
        }
        totals[arm] = {
            "steady_wire": sum(per_wire),
            "steady_input_tokens": sum(per_in),
            "total_wire": sum(per_wire),
            "total_input_tokens": sum(per_in),
            "cached_tokens": sum(r["cached_tokens"] for r in steady),
            "output_tokens": sum(r["output_tokens"] for r in steady),
            "thought_tokens": sum(r["thought_tokens"] for r in steady),
            "latency": _latency_stats([r["elapsed_ms"] for r in steady]),
            # Excluded from the comparison, but not hidden: the build is real money.
            "cachegen_wire": sum(r["wire_sent"] + r["wire_recv"] for r in gen),
            "cachegen_tokens": sum(r["input_tokens"] for r in gen),
            "cachegen_ms": sum(r["elapsed_ms"] for r in gen),
            # Two clocks: call_ms is time spent inside measured calls, while wall_ms
            # is the arm's start-to-finish time, so it also covers the cache builds
            # and deletes that happen between them.
            "call_ms": sum(r["elapsed_ms"] for r in steady),
            "wall_ms": wall.get(arm, 0),
            "errors": sum(1 for r in gen + steady if r["error"]),
        }

    # A run with a broken arm still returns numbers, and numbers from a failed call
    # look like numbers from a good one. Name the failing cases so they can't hide.
    failures = [{"arm": r["arm"], "phase": r["phase"], "turn": r["turn"],
                 "error": r["error"]}
                for r in records if r.get("error")]

    return {"mode": "comparison", "arms": arms, "series": series,
            "totals": totals, "failures": failures}


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
