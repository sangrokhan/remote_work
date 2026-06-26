# Gemini API Traffic Experiment — Design Spec

**Date:** 2026-06-26
**Status:** Approved
**Branch:** worktree-gemini-traffic-test

## Goal

Prove that **stateless cumulative API usage** (resending the full conversation
history on every turn) produces dramatically more token + network traffic than a
**delta-only** approach (sending only the new turn), and that — because Gemini
bills on *all* tokens fed to inference — this directly inflates cost.

This is a **traffic measurement experiment**, not a chat-quality experiment. The
delta-only mode intentionally loses conversational context; we only compare bytes
and tokens, never answer correctness.

## Hypothesis

For an N-turn conversation:

- **Stateless (full-history resend):** turn `k` sends turns `1..k`. Cumulative
  input tokens grow as **O(N²)**.
- **Delta-only:** turn `k` sends only turn `k`. Cumulative input tokens grow as
  **O(N)**.

Therefore the stateless/delta ratio grows roughly linearly with N, and total
billed tokens (and wire bytes) diverge sharply. We expect a clear, visually
obvious gap on the charts.

## Why no account split is needed

Each call is **self-measured**:

- **Tokens** come from each response's `usageMetadata`
  (`promptTokenCount`, `candidatesTokenCount`, `totalTokenCount`) — scoped to that
  single call.
- **Wire bytes** come from a per-call socket byte counter we wrap around the
  request.

Every record is tagged `mode = stateless | delta`. Attribution is per-request in
our own code, so a single API key / account is sufficient. We do **not** rely on
Google's aggregate billing dashboard. (Optional later cross-check: run the two
modes in separate time windows and compare against the billing page.)

## Architecture

```
gemini_api_test/
├── app.py                  Flask: routes, serves UI, triggers experiment
├── experiment.py           runs the same conversation in both modes, builds records
├── gemini_client.py        Gemini call + socket byte counter + usageMetadata capture
├── metrics.py              record/summary builders, cumulative + ratio + cost stats
├── store.py                save/load run JSON files under data/runs/
├── templates/index.html    start button, params form, result tables
├── static/app.js           Chart.js cumulative-tokens + cumulative-bytes charts
├── static/style.css        minimal styling
├── data/runs/*.json        persisted run history (background collection)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml      passes GEMINI_API_KEY env into container
├── tests/test_metrics.py   unit tests for metrics math (no network)
├── PROJECT_GOAL.md         goal / hypothesis / method / proof (user-facing)
└── README.md               run instructions
```

## Components & interfaces

### gemini_client.py
- `call_gemini(model, contents, api_key) -> CallResult`
  - Sends a single `generateContent` request via `requests`.
  - Wraps the socket so raw bytes sent/received over TLS are counted
    (`wire_sent`, `wire_recv`) — real on-wire bytes, no tcpdump / NET_ADMIN.
  - Also records application payload size: `req_payload_bytes` (JSON body),
    `resp_payload_bytes` (JSON body).
  - Extracts `usageMetadata` → `prompt_tokens`, `resp_tokens`, `total_tokens`.
  - On error: returns `CallResult` with `error` set; never raises out.
- **Mock mode** (`GEMINI_MOCK=1` or no key): returns synthetic response with
  token counts proportional to input length, and synthetic byte sizes from the
  serialized payload. Lets the full flow + charts run without quota.

`CallResult` fields: `mode, turn, prompt_tokens, resp_tokens, total_tokens,
wire_sent, wire_recv, req_payload_bytes, resp_payload_bytes, error`.

### experiment.py
- `run_experiment(turns, message_chars, model, api_key) -> RunResult`
  - Builds N synthetic user messages of ~`message_chars` chars.
  - **Stateless run:** maintains a growing `contents` list; turn k sends 1..k.
  - **Delta run:** sends only the single turn-k message each call.
  - Collects per-turn `CallResult` for both modes.
  - Endpoint tagged as `generativelanguage.googleapis.com:443`.

### metrics.py
- `summarize(records) -> Summary` with per-mode cumulative tokens, cumulative wire
  bytes, cumulative payload bytes, final ratio (stateless/delta), and a USD cost
  estimate = `cumulative_tokens * price_per_token` (price configurable; default
  from a constant, clearly labeled as an estimate).
- Pure functions, fully unit-testable.

### store.py
- `save_run(run_result) -> path`: timestamped JSON in `data/runs/`.
- `list_runs() -> [summary,...]`: for history endpoint / trend.
- `aggregate() -> totals`: total tokens + traffic across all runs, per endpoint.

### app.py (Flask)
Routes:
- `GET /` → `index.html`.
- `POST /run` (JSON: turns, message_chars, model) → runs experiment, saves,
  returns full per-turn + summary JSON.
- `GET /history` → list of past run summaries + aggregate totals.

### Frontend (templates/index.html + static/app.js)
- Params form: turns, message chars, model, mock toggle.
- **Start button** → POST /run → render:
  - Two line charts (Chart.js): cumulative tokens (stateless vs delta),
    cumulative wire bytes (stateless vs delta).
  - Per-turn table + summary table (totals, ratio, cost estimate).
- History panel: past runs + aggregate token/traffic totals per endpoint.

## Data flow

UI Start → `POST /run` → `experiment.run_experiment` → per-turn
`gemini_client.call_gemini` (both modes) → `metrics.summarize` →
`store.save_run` → JSON response → `app.js` renders charts + tables.
Background collection = each run persisted to `data/runs/`, surfaced via
`/history` and `aggregate()`.

## Error handling
- Missing key & mock off → `/run` returns 400 with clear message; UI shows it.
- Per-turn API failure → record `error`, continue remaining turns, mark in table.
- Socket counter wraps the real call; counter failure must not break the request
  (fall back to payload bytes).

## Testing
- `tests/test_metrics.py`: cumulative growth (O(N²) vs O(N) shape), ratio, cost
  math — pure, no network.
- Manual: `GEMINI_MOCK=1` end-to-end run verifies UI, charts, persistence with no
  quota usage.

## Out of scope (YAGNI)
- Real packet capture (tcpdump / pcap).
- Context Caching API comparison (chosen baseline is delta-only).
- Multi-user, auth, DB. Flat JSON files are sufficient.
- Answer-quality evaluation of delta mode.
