"""Turn a run's records into per-(provider, arm) series and totals.

Pure math over the records a run produced: no network, no provider knowledge, no
prices. Everything a chart or a table needs comes from here, so that the two
provider adapters cannot each invent their own idea of what "total uplink" means
and quietly become incomparable.

Three things this module refuses to do, each because doing them once produced a
number that looked like evidence and was not:

* It never folds a prep phase into a total. A cache build re-sends the whole
  system prompt, so counting it would drown every measured turn.
* It never averages across `measure` modes. Bytes off a streamed pass carry the
  stream's framing (and, on OpenAI, obfuscation padding); bytes off a blocking
  pass do not. They are different measurements wearing the same column name.
* It never prices anything. A dollar figure built on a per-token rate nobody
  verified is a guess dressed up as a result.
"""

from __future__ import annotations

# The five marks that bracket a turn, plus the tail:
#   req_sent    the client's history is finally all on the wire
#   ttfb        the server starts answering (network + queue, no tokens yet)
#   ttft        the answer starts
#   ttlt        the answer ends            <- what a streaming user waits for
#   turn_end    the server lets go         <- what a blocking client waits for
# store_tail_ms is turn_end - ttlt: on a stored interaction the write lands after
# the last token, and that gap is a wait no streaming user ever does.
MARKS = ("req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms",
         "store_tail_ms")

_TOKENS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")


def key_of(provider: str, arm: str) -> str:
    """The identity of a series. An arm name alone is not unique across providers,
    and two providers can and do use the same word for different machinery."""
    return f"{provider}:{arm}"


def _cumulative(values: list[int]) -> list[int]:
    # acc starts at int 0 so an all-int series stays int: these are byte and token
    # counts, and a tooltip reading "1400.0 bytes" lies about the precision of the
    # measurement.
    out, acc = [], 0
    for v in values:
        acc += v
        out.append(acc)
    return out


def _stats(values: list[int]) -> dict:
    """mean/median/min/max over every call, not over per-turn averages.

    Averaging first erases the one turn that took four seconds, and that turn is
    the only one the user actually felt.
    """
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "n": 0}
    s = sorted(values)
    n = len(s)
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    return {"mean": round(sum(s) / n, 1), "median": median,
            "min": s[0], "max": s[-1], "n": n}


def _elapsed(rec: dict) -> int:
    """How long the call took. `turn_end_ms` is the record's own field for it; the
    fallback exists because a record that carried `elapsed_ms` instead would
    otherwise contribute a silent zero to the clock."""
    if rec.get("turn_end_ms"):
        return int(rec["turn_end_ms"])
    return int(rec.get("elapsed_ms") or 0)


def _mark(rec: dict, name: str) -> int:
    if name == "store_tail_ms":
        # Derived rather than read, so a record written before the field existed
        # still yields the tail instead of a silent zero.
        if "store_tail_ms" in rec:
            return int(rec["store_tail_ms"] or 0)
        return int(rec.get("turn_end_ms") or 0) - int(rec.get("ttlt_ms") or 0)
    return int(rec.get(name) or 0)


def summarize(run: dict) -> dict:
    """Per-(provider, arm) series, totals, prep cost and failures for one run.

    Keyed by "provider:arm" so nothing collides, with `provider` and `arm` carried
    as fields inside every entry so a UI can group along either axis without
    parsing the key back apart.
    """
    records = run.get("records") or []
    params = run.get("params") or {}
    wall = run.get("wall_ms") or {}

    # Bytes from a streamed pass and bytes from a blocking pass are not the same
    # measurement. The mode rides at the top of the summary and again on every
    # series, so no chart can put two of them on one axis without saying so.
    measure = params.get("measure") or ""

    keys: list[str] = []
    for r in records:
        k = key_of(r.get("provider", ""), r.get("arm", ""))
        if k not in keys:
            keys.append(k)

    series: dict = {}
    totals: dict = {}
    prep: dict = {}
    for k in keys:
        mine = [r for r in records if key_of(r.get("provider", ""),
                                             r.get("arm", "")) == k]
        provider, arm = mine[0].get("provider", ""), mine[0].get("arm", "")

        # Anything that is not a steady turn is preparation: a Gemini cache build,
        # an OpenAI conversation create. It is setup, not traffic. It is reported
        # below, in its own bucket, and never added to a total -- comparing one
        # arm's setup against another arm's traffic answers no question anyone
        # asked.
        steady = sorted((r for r in mine if r.get("phase") == "steady"),
                        key=lambda r: r.get("turn", 0))
        prepped = [r for r in mine if r.get("phase") != "steady"]

        per_up = [int(r.get("wire_sent") or 0) for r in steady]
        per_down = [int(r.get("wire_recv") or 0) for r in steady]
        # Uplink is the axis the arms actually differ on -- a resent history is
        # upload, a stored one is not -- so it stays on its own line rather than
        # being buried in a combined figure that the model's answer dominates.
        per_wire = [u + d for u, d in zip(per_up, per_down)]
        per_in = [int(r.get("input_tokens") or 0) for r in steady]

        measures = sorted({r.get("measure", "") for r in mine if r.get("measure")})

        series[k] = {
            "provider": provider,
            "arm": arm,
            "measure": measures[0] if len(measures) == 1 else (measure or ""),
            "turns": [r.get("turn", 0) for r in steady],
            "per_turn_wire_sent": per_up,
            "per_turn_wire_recv": per_down,
            "per_turn_wire": per_wire,
            "per_turn_input_tokens": per_in,
            "cum_wire_sent": _cumulative(per_up),
            "cum_wire_recv": _cumulative(per_down),
            "cum_wire": _cumulative(per_wire),
            "cum_input_tokens": _cumulative(per_in),
            **{f"per_turn_{m}": [_mark(r, m) for r in steady] for m in MARKS},
        }

        totals[k] = {
            "provider": provider,
            "arm": arm,
            "measure": series[k]["measure"],
            "turns": len(steady),
            "wire_sent": sum(per_up),
            "wire_recv": sum(per_down),
            "wire": sum(per_wire),
            **{t: sum(int(r.get(t) or 0) for r in steady) for t in _TOKENS},
            "marks": {m: _stats([_mark(r, m) for r in steady]) for m in MARKS},
            # Two clocks. call_ms is time spent inside the measured calls; wall_ms
            # is the steady stage start to finish -- the same window the pcap
            # covers. Neither includes prep or teardown, which is why prep has its
            # own bucket.
            "call_ms": sum(_elapsed(r) for r in steady),
            "wall_ms": wall.get(k, wall.get(arm, 0)),
            "errors": sum(1 for r in mine if r.get("error")),
        }

        if prepped:
            # Excluded from the totals, but not hidden: the build is real money and
            # real bytes, and an arm whose steady traffic is cheap because it paid
            # up front should be seen to have paid up front.
            prep[k] = {
                "provider": provider,
                "arm": arm,
                "phases": sorted({r.get("phase", "") for r in prepped}),
                "calls": len(prepped),
                "wire_sent": sum(int(r.get("wire_sent") or 0) for r in prepped),
                "wire_recv": sum(int(r.get("wire_recv") or 0) for r in prepped),
                "wire": sum(int(r.get("wire_sent") or 0)
                            + int(r.get("wire_recv") or 0) for r in prepped),
                **{t: sum(int(r.get(t) or 0) for r in prepped) for t in _TOKENS},
                "elapsed_ms": sum(_elapsed(r) for r in prepped),
            }

    # A run with a broken arm still produces plausible numbers, and a number from a
    # failed call is shaped exactly like a number from a good one. Name every
    # failing case so it cannot pass for a measurement.
    failures = [
        {"provider": r.get("provider", ""), "arm": r.get("arm", ""),
         "key": key_of(r.get("provider", ""), r.get("arm", "")),
         "phase": r.get("phase", ""), "turn": r.get("turn", 0),
         "error": r["error"]}
        for r in records if r.get("error")
    ]

    return {
        "measure": measure,
        "keys": keys,
        "series": series,
        "totals": totals,
        "prep": prep,
        "failures": failures,
    }
