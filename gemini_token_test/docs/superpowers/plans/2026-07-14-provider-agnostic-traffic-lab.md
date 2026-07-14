# Plan — one traffic lab, many providers

Date: 2026-07-14
Status: approved, not yet executed

## Why

Two projects now run the same experiment against different vendors, each with its
own socket counter, its own timing code, its own metrics, its own store, its own
Flask app:

| concern | `gemini_token_test/` | `openai_token_test/` |
|---|---|---|
| socket byte counter | inside `gemini_client.py` | `wire.py` |
| stream timings | `streaming.py` (5 marks) | `openai_client._post_stream` (ttft/ttlt) |
| metrics | `metrics.py` | `metrics.py` |
| store | `store.py` (JSON + Firestore) | `store.py` (JSON) |
| pcap | `capture.py` | `capture.py`, `verify_pcap.py` |
| front end | Flask `/compare` + SSE + charts | Flask `/run/stream` + CLI |

The arms are the *same question* asked of two vendors — who keeps the conversation,
and what does that cost in bytes and in seconds:

| concept | gemini arm | openai arm |
|---|---|---|
| client resends everything | `stateless` | `chat_stateless` |
| same, other endpoint (control) | `interaction_stateless` | `responses_stateless` |
| server keeps the history | `interaction` | `responses_stateful` |
| explicit cache | `cached` | — |
| system prompt inlined into turn 1 | `interaction_inline` | — |
| no context at all (lower bound) | `nocontext` | — |

A provider-agnostic core makes the vendors directly comparable and stops the two
codebases from drifting apart. It also makes a third vendor a day of work, not a
rewrite.

## Decisions taken (2026-07-14, with the user)

1. **Shared core + provider adapters.** New package; core owns wire counting,
   stream timing, metrics, store, capture, and the app. Providers declare their
   arms and build their payloads.
2. **Dual-pass measurement.** Bytes come from a non-streamed call; latency comes
   from a streamed call. OpenAI's SSE deltas carry `include_obfuscation` padding
   that corrupts a byte measurement, and TTFT cannot be read off a blocking call —
   so each turn is measured twice, deliberately, and each number comes from the
   call that can carry it honestly. This doubles API calls: it must be opt-in per
   run, and the default for a quick run is bytes-only.
3. **Dead code goes.** `run_experiment` / `run_three_stage` / `/run` / `/run/stream`,
   Firestore, `cost_usd` / `token_ratio` / `wire_ratio`, and the endpoint inspector
   (`/inspect`) are removed.

## What the experiment measures (the goal, restated)

For an N-turn conversation, per arm:

- **Client bytes** — `wire_sent` (the client's own upload, the axis the arms differ
  on) and `wire_recv`, per turn and cumulative. Socket-level, headers included.
- **Tokens** — input / cached / output / thought, per turn and cumulative.
- **Latency, five marks** — `req_sent` (request fully on the wire), `ttfb`,
  `ttft`, `ttlt` (answer complete — what a streaming user waits for), `turn_end`
  (server let go — what a blocking client waits for). `store_tail = turn_end - ttlt`.

Measured on Gemini, 2026-07-14, and the reason the goal is being restated: the
byte story held (a resent history is O(N²) upload) but the *latency* story turned
out to be about `store:true` — the write costs ~1.8 s per turn and lands after the
answer is already out, and a chained interaction cannot switch it off (400: "store
must be true when previous_interaction_id is set").

## Phases

### Phase 1 — remove dead code (gemini_token_test)

- `experiment.py`: delete `run_experiment`, `run_three_stage`, `MODES`. Keep
  `load_request`, `run_comparison`, the arms.
- `app.py`: delete `/run`, `/run/stream`, `/inspect`, `/download/transcript`,
  `_execute_run`, `_execute_three_stage`, and their imports.
- `metrics.py`: delete `summarize`, `summarize_three_stage`, `PRICE_PER_TOKEN`,
  `_ratio`, `_last`, `_series` if unused. Keep `summarize_comparison`.
- `store.py`: strip Firestore (`_firestore`, `firestore_active`, the dual write and
  the Firestore reads). Local JSON only. Drop `google-cloud-firestore` from
  requirements. Keep the DUMMY dataset only if the UI still needs it.
- delete `inspector.py`, `tests/test_inspector.py`.
- `templates/index.html`, `static/app.js`: remove the inspector panel, the Firestore
  status text, and any control that drove the deleted routes.
- tests: delete the ones that only covered deleted code; keep every test that still
  describes live behaviour.

Exit: the offline suite is green and nothing imports a deleted symbol.

### Phase 2 — docs match the code

- `PROJECT_GOAL.md`: rewrite. Goal = bytes × latency across the ways of keeping a
  conversation, on one host per provider. Record the store-tail finding.
- `README.md`: Developer API + API key, six arms, streaming, the outputs, how to
  run, how to keep the outputs from piling up.
- `docs/call-flow.md`: redraw for `run_comparison` and the six arms; the current
  diagram documents `run_three_stage`, which will no longer exist. Fix the system
  prompt size (20,653 chars, ~4.4k input tokens — not "9 KB").
- `docs/README.md`: the screenshot is of a UI that no longer exists; either
  regenerate it or say plainly that it is historical.

### Phase 3 — the shared core

Target layout (name to be confirmed):

```
token_traffic/
  core/
    wire.py         socket byte counter + the req_sent / ttfb marks
    streaming.py    SSE reader, the five marks, per-vendor text extractors
    record.py       the one per-turn record every arm produces
    metrics.py      series + totals + the five clocks + store_tail
    store.py        run JSON, retention policy, schema version
    capture.py      pcap per arm
    app.py          Flask: run, compare, download, history
  providers/
    gemini.py       arms: stateless, cached, interaction, interaction_inline,
                    interaction_stateless, nocontext
    openai.py       arms: chat_stateless, responses_stateless, responses_stateful
  fixtures/
    perf.json       the shared scenario (system prompt + N questions)
```

- A provider declares `ARMS`, builds a request body per turn, and knows how to read
  its own usage and its own answer text out of a response. Everything else is core.
- Records carry `provider` alongside `arm`, so one run can hold both vendors and the
  CSV stays one row per call.
- Dual-pass: `measure=bytes` (non-streamed), `measure=latency` (streamed), or
  `measure=both` (two calls per turn, labelled `pass` in the record).
- Outputs: schema version field, retention policy (keep N newest, mock runs
  excluded), and one CSV whose columns are the five marks + the byte split.

Exit: one run compares Gemini and OpenAI arms side by side on the same fixture.

## Rules for this work

- **No live API calls without asking.** The offline suite (mock mode) covers every
  arm. When a live run is needed, say what it costs in calls and wait.
- Every phase ends green: tests pass, nothing imports a deleted symbol.
