"""Does `store: false` cost you the prompt cache?

The 2-turn run showed responses_stateless (store=false) reporting cached_tokens=0
on every call, while chat_stateless and responses_stateful hit the cache on the
same system prompt. Three explanations fit that observation:

  H1  store=false disables cache READ  -> you pay full price for every turn
  H2  store=false disables cache WRITE -> nothing to hit later
  H3  store=false still caches, but cached_tokens is simply not REPORTED
      (a measurement artifact, not a billing one)

They are separable. Send the identical large prefix N times in a row under each
store setting and watch cached_tokens across the repeats:

  H1 true  -> store=false stays 0 forever; store=true climbs after call 1
  H2 true  -> same shape as H1 from this vantage point... so we also cross-probe:
              warm the cache with store=false, then READ it with store=true.
              If that read hits, store=false wrote fine and the problem is read.
  H3 true  -> billing would disagree with the field; we cannot see billing here,
              so we can only flag it. But a store=false write that a store=true
              call can hit (cross-probe) argues the cache is real either way.

Each condition gets its OWN unique prefix marker at the very START of the system
prompt, so conditions cannot warm each other's cache. Prompt caching matches on
an exact prefix from byte zero.

Cost: ~15 calls x ~5k input tokens on gpt-4.1-nano ~= $0.008.
"""

from __future__ import annotations

import json
import time

import env  # noqa: F401
import fixture as fixture_mod
import openai_client as oc

REPEATS = 3
QUESTION = "Reply with the single word: ok"


def _probe(label: str, marker: str, store: bool, system: str,
           cache_key: str | None = None) -> list[dict]:
    """Send the same body `REPEATS` times to /v1/responses. Watch cached_tokens."""
    url = f"{oc.base_url()}/responses"
    tagged = f"[{marker}]\n{system}"
    items = [{"role": "system", "content": tagged},
             {"role": "user", "content": QUESTION}]
    body = oc._responses_body(oc.DEFAULT_MODEL, items, store=store)
    if cache_key:
        body["prompt_cache_key"] = cache_key

    rows = []
    for i in range(1, REPEATS + 1):
        data, sent, recv, req_b, resp_b, ms = oc._post(url, body)
        u = data.get("usage") or {}
        row = {
            "label": label,
            "store": store,
            "call": i,
            "input_tokens": u.get("input_tokens", 0),
            "cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
            "up_bytes": req_b,
            "ms": ms,
        }
        rows.append(row)
        print(f"  {label:<28} call {i}  in={row['input_tokens']:>6}  "
              f"cached={row['cached_tokens']:>6}  {ms:>5}ms", flush=True)
        time.sleep(1)  # give the cache a beat to become visible
    return rows


def main() -> None:
    fx = fixture_mod.load("perf")
    print(f"model={oc.DEFAULT_MODEL}  system={fx.system_chars} chars  "
          f"repeats={REPEATS}\n")

    rows: list[dict] = []

    # A: store=false, same body three times. Does it ever hit its own cache?
    print("A. store=false, repeated")
    rows += _probe("A store=false", "probe-a", False, fx.system)

    # B: store=true, same body three times. Control — this should hit.
    print("\nB. store=true, repeated  (control)")
    rows += _probe("B store=true", "probe-b", True, fx.system)

    # C: cross-probe. Warm with store=false, then read with store=true, on the
    #    SAME prefix. If C's store=true call hits, then store=false did write the
    #    cache and the deficit is on the read side, not the write side.
    print("\nC. warm with store=false, then read with store=true  (same prefix)")
    rows += _probe("C warm  store=false", "probe-c", False, fx.system)
    rows += _probe("C read  store=true", "probe-c", True, fx.system)

    # D: the routing hypothesis. A and B show cache hits appearing and vanishing
    #    under identical conditions, and C hits with store=false — so `store` is
    #    not the variable. What OpenAI's docs say governs hit rate is ROUTING:
    #    prompt_cache_key steers same-prefix requests to the same machine. If a
    #    stable key makes hits reliable, the earlier cached=0 was noise, not a
    #    property of store=false.
    print("\nD. store=false WITH a stable prompt_cache_key  (routing hypothesis)")
    rows += _probe("D key store=false", "probe-d", False, fx.system,
                   cache_key="probe-d-key")

    with open("results/probe_store_cache.json", "w") as f:
        json.dump(rows, f, indent=2)

    print("\n--- verdict inputs ---")
    a_last = [r["cached_tokens"] for r in rows if r["label"].startswith("A")][-1]
    b_last = [r["cached_tokens"] for r in rows if r["label"].startswith("B")][-1]
    c_read = [r["cached_tokens"] for r in rows if r["label"].startswith("C read")]
    print(f"A store=false, last call cached : {a_last}")
    print(f"B store=true,  last call cached : {b_last}")
    print(f"C store=true reading a prefix warmed by store=false: {c_read}")

    print("\nRead it:")
    if b_last > 0 and a_last == 0 and c_read and c_read[0] > 0:
        print("  store=false WRITES the cache but never READS it. A stateless client")
        print("  that declines server storage pays full input price on every turn.")
    elif b_last > 0 and a_last == 0 and c_read and c_read[0] == 0:
        print("  store=false neither writes nor reads. Cache is off entirely for it.")
    elif a_last > 0:
        print("  store=false does hit the cache. The 2-turn observation was a warm-up")
        print("  artifact, not a store= effect.")
    else:
        print("  inconclusive — even the store=true control did not hit. Cache may be")
        print("  cold or the prefix may be below the 1024-token floor.")


if __name__ == "__main__":
    main()
