# Context Caching — 3-Stage Stateful Traffic Experiment

**Date:** 2026-06-30
**Status:** Implemented (live Vertex caching + mock simulation)

## Goal

Prove that a server-side stateful pattern (Vertex **Context Caching**) lets the
client send **only the new question** per turn — far less traffic than stateless
full-history resend — by reproducing what the Interactions API would do, using
real caches (company pays Vertex). Measured signal: **request/response packet size
+ content length**. (Response quality is a later concern; we still store responses.)

## Three stages (explicit)

### Stage 1 — Scenario run (stateless)
- A 10-step follow-up Q&A. Each turn sends the **full accumulated history**
  (Q1,A1,…,Q(k-1),A(k-1),Qk) — stateless.
- Capture and store **every request and response** (text + sizes) as one JSON
  scenario document, keyed by `exec_id`.

### Stage 2 — Build cumulative caches
- From the stored scenario, create caches of growing prefixes:
  - cache₁ = [Q1,A1]
  - cache₂ = [Q1,A1,Q2,A2]
  - … cache_k = turns 1..k (Q&A)
- Each `cachedContents.create()` → `cache_id` (+ cached token count). Caches below
  the 2,048-token minimum are skipped and marked. Stored as a `cache_set` under
  the scenario `exec_id`.

### Stage 3 — Stateful replay (cache + question only)
- For turn k, call `generateContent(cachedContent=cache_(k-1), contents=[Qk])` —
  the prefix lives server-side, so **only the new question** is sent.
- Monitor **request/response packet size (wire bytes) + content length (payload
  bytes)** per turn. Compare against Stage-1 stateless sizes.
- Best-effort `cachedContents.delete()` afterwards (unless `KEEP_CACHE=1`).

## Data model (one scenario doc, exec_id)

```json
{
  "exec_id": "...", "mode": "caching-3stage", "mock": false,
  "model": "...", "params": {...},
  "scenario": [ {"turn":1,"question":"...","answer":"...",
                 "req_bytes":N,"resp_bytes":N,"wire_sent":N,"wire_recv":N}, ... ],
  "cache_set": [ {"k":1,"cache_id":"...","cached_tokens":N,"skipped":false}, ... ],
  "stateful": [ {"turn":2,"cache_id":"...","cached_tokens":N,
                 "req_bytes":N,"resp_bytes":N,"wire_sent":N,"wire_recv":N}, ... ],
  "summary": { "stateless_series": {...}, "stateful_series": {...},
               "totals": {"stateless_wire":N,"stateful_wire":N,"wire_ratio":R, ...} }
}
```

`summary.*_series` carry cumulative wire bytes + content length for the two charts;
the existing history viewer/graph plots them by `exec_id`.

## Code changes

- `gemini_client`: `CallResult.response_text`; capture it from candidates;
  `create_cache`/`delete_cache` (already added); `call_gemini(cached_content=...)`.
- `experiment.run_three_stage(model, request_name, turns)` → runs all three
  stages, returns the scenario doc above. Mock simulates caches + responses.
- `metrics`: build `stateless_series` / `stateful_series` (cum wire + content) and
  totals (wire_ratio, content_ratio, cached_tokens).
- `store`: save the combined doc under `exec_id` (Firestore + JSON), dummy backup.
- `app`: `/run` mode `caching-3stage` runs the pipeline; existing history routes.
- UI: mode option "caching-3stage (stateless→cache→stateful)"; result shows the
  two series overlaid (stateless vs stateful wire/content) + per-stage tables.

## Notes / honesty

- Live Vertex caching path is unverified here (no creds); validated via mock +
  the structure. Real run needs `GOOGLE_CLOUD_PROJECT` + ADC + large enough texts
  (>=2,048 tokens of accumulated prefix before a cache can form).
- Early caches (cache₁) may fall below the min-token bar → skipped; Stage 3 then
  sends full context for those turns (marked), since no cache exists yet.
