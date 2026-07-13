"""Aggregate an experiment into per-arm series, ratios, and cost.

The headline is a deliberate contrast between two ratios:

  wire_ratio  = uplink bytes,   stateless / stateful   -> large, grows with N
  token_ratio = input tokens,   stateless / stateful   -> ~1.0

Bytes collapse when the server holds the state. Billing does not. That is the
gap this experiment exists to quantify.
"""

from __future__ import annotations

import os

# gpt-5.4-nano list price, USD per million tokens. Override for another model.
PRICE_INPUT = float(os.environ.get("PRICE_INPUT", "0.20"))
PRICE_CACHED_INPUT = float(os.environ.get("PRICE_CACHED_INPUT", "0.02"))
PRICE_OUTPUT = float(os.environ.get("PRICE_OUTPUT", "1.25"))

_FIELDS = ("req_payload_bytes", "resp_payload_bytes", "wire_sent", "wire_recv",
           "input_tokens", "cached_tokens", "billed_uncached_tokens",
           "output_tokens", "reasoning_tokens", "latency_ms")


def _cumulative(values: list[float]) -> list[float]:
    out, total = [], 0.0
    for v in values:
        total += v
        out.append(total)
    return out


def cost_usd(input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * PRICE_INPUT
            + cached_tokens * PRICE_CACHED_INPUT
            + output_tokens * PRICE_OUTPUT) / 1_000_000


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def arm_series(experiment: dict, arm: str) -> dict:
    """Per-turn values for one arm, averaged over repeats."""
    runs = [r for r in experiment["runs"] if r["arm"] == arm]
    if not runs:
        return {}
    n_turns = len(runs[0]["turns"])

    per_turn = {f: [] for f in _FIELDS}
    for k in range(n_turns):
        for f in _FIELDS:
            per_turn[f].append(_mean([r["turns"][k][f] for r in runs]))

    # the stateful arm paid an upfront conversation-create call. It is a real
    # upload of the system prompt and is counted, not hidden.
    setup_sent = _mean([(r["setup"] or {}).get("req_payload_bytes", 0) for r in runs])
    setup_wire = _mean([(r["setup"] or {}).get("wire_sent", 0) for r in runs])

    series = {
        "arm": arm,
        "turns": n_turns,
        "setup_req_bytes": setup_sent,
        "setup_wire_sent": setup_wire,
        "per_turn": per_turn,
        "cum_req_bytes": _cumulative(per_turn["req_payload_bytes"]),
        "cum_wire_sent": _cumulative(per_turn["wire_sent"]),
        "cum_input_tokens": _cumulative(per_turn["input_tokens"]),
        "cum_cached_tokens": _cumulative(per_turn["cached_tokens"]),
        "cum_billed_uncached": _cumulative(per_turn["billed_uncached_tokens"]),
        "cum_output_tokens": _cumulative(per_turn["output_tokens"]),
    }

    totals = {f: sum(per_turn[f]) for f in _FIELDS}
    # setup bytes belong to the arm's total upload cost
    totals["req_payload_bytes"] += setup_sent
    totals["wire_sent"] += setup_wire
    totals["cost_usd"] = cost_usd(int(totals["input_tokens"]),
                                  int(totals["cached_tokens"]),
                                  int(totals["output_tokens"]))
    totals["mean_latency_ms"] = _mean(per_turn["latency_ms"])
    series["totals"] = totals
    return series


def summarize(experiment: dict) -> dict:
    arms = experiment["config"]["arms"]
    series = {a: arm_series(experiment, a) for a in arms}
    series = {a: s for a, s in series.items() if s}

    out = {"config": experiment["config"], "arms": series, "ratios": {}}

    stateful = series.get("responses_stateful")
    if stateful:
        for a, s in series.items():
            if a == "responses_stateful":
                continue
            sf = stateful["totals"]
            out["ratios"][f"{a}_vs_stateful"] = {
                "upload_bytes": _ratio(s["totals"]["req_payload_bytes"],
                                       sf["req_payload_bytes"]),
                "wire_sent": _ratio(s["totals"]["wire_sent"], sf["wire_sent"]),
                "input_tokens": _ratio(s["totals"]["input_tokens"],
                                       sf["input_tokens"]),
                "billed_uncached_tokens": _ratio(s["totals"]["billed_uncached_tokens"],
                                                 sf["billed_uncached_tokens"]),
                "cost_usd": _ratio(s["totals"]["cost_usd"], sf["cost_usd"]),
            }
    return out


def _ratio(a: float, b: float) -> float:
    return round(a / b, 3) if b else 0.0


def print_summary(summary: dict) -> None:
    cfg = summary["config"]
    print(f"\n=== {cfg['model']} · fixture={cfg['fixture']} · {cfg['turns']} turns "
          f"· {cfg['repeats']} repeats ===\n")

    hdr = (f"{'arm':<20} {'upload B':>10} {'wire sent':>10} {'in tok':>8} "
           f"{'cached':>8} {'billed':>8} {'out tok':>8} {'cost $':>9} {'ms':>7}")
    print(hdr)
    print("-" * len(hdr))
    for arm, s in summary["arms"].items():
        t = s["totals"]
        print(f"{arm:<20} {t['req_payload_bytes']:>10,.0f} {t['wire_sent']:>10,.0f} "
              f"{t['input_tokens']:>8,.0f} {t['cached_tokens']:>8,.0f} "
              f"{t['billed_uncached_tokens']:>8,.0f} {t['output_tokens']:>8,.0f} "
              f"{t['cost_usd']:>9.5f} {t['mean_latency_ms']:>7,.0f}")

    if summary["ratios"]:
        print("\nratios (stateless / stateful):")
        for name, r in summary["ratios"].items():
            print(f"  {name}")
            print(f"    upload bytes  : {r['upload_bytes']:.2f}x")
            print(f"    input tokens  : {r['input_tokens']:.2f}x")
            print(f"    billed tokens : {r['billed_uncached_tokens']:.2f}x")
            print(f"    cost          : {r['cost_usd']:.2f}x")

    print("\nRead it this way: upload bytes fall off a cliff when the server holds "
          "the history,\nbut input tokens barely move — every prior token is still "
          "billed on every turn.")
