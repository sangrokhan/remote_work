# Interaction API vs stateless / cachedContents — traffic & latency comparison

Date: 2026-07-10
Status: host and model fixed by probe; arm implementation pending

## Goal

For the same 10-turn conversation, measure how much **network traffic** (wire
bytes), how many **tokens**, and how much **latency** each way of carrying context
costs — and keep the raw request/response of every call. One host, one key.

Four stages; the headline comparison is **1 vs 3 vs 4**:

1. **`stateless`** — resend the whole history (system prompt + all prior turns) every turn. **compared**
2. **`cachebuild`** — the `cachedContents` create/refresh calls that make stage 3 possible. Not a way of chatting; it is the *setup cost* of stage 3, measured and attributed to it. **setup bucket**
3. **`cached`** — stateless API + `cachedContent`: the prefix lives in an explicit cache, each turn sends only `cachedContent` + the new question. **compared** (its cost = stage 2 setup + its own per-turn)
4. **`interaction`** — Interactions API: history is held server-side via `previous_interaction_id`; the system prompt is re-sent every turn (it is not held). Each turn sends system prompt + new question. **compared**

`nocontext` (new question only) stays available as a lower-bound diagnostic but is
out of the headline comparison.

There is no fifth arm. "Interaction + explicit cache" was floated earlier; the
Interactions API supports only **implicit** caching, so it cannot reference a
`cachedContents` object. The `cached-sysonly` idea is likewise dropped.

## What each strategy actually saves (the whole point)

Three mechanisms, and they reduce different things. Conflating them is the mistake
the whole design guards against.

| Mechanism | History re-sent? | System prompt re-sent? | Wire bytes | Token bill |
|-----------|------------------|------------------------|-----------|------------|
| **implicit caching** (server-side KV, automatic on 2.5+) | **yes, every turn** | yes, every turn | **not reduced** | repeated prefix discounted |
| **explicit cache** (`cachedContents`) | uploaded **once** | uploaded once | **reduced** | cached tokens billed cheaper |
| **interaction** (`previous_interaction_id`) | **no** — server holds it | **yes, every turn** | history bytes saved only | implicit discount applies |

Consequences that shape the expected result:

- **Implicit caching does not cut wire traffic.** You still upload every token; the
  server just discounts the repeated prefix on the *bill*. So putting the system
  prompt inline in a `stateless` call and putting it in `interaction`'s
  `system_instruction` cost the **same bytes**. The only wire difference between
  `stateless` and `interaction` is the conversation **history**.
- **Only explicit caching cuts wire bytes** — upload the prefix once, then reference
  it by name. That saving is real but front-loaded: stage 2's one-time upload is the
  price, so `cached`'s true cost is stage 2 + its per-turn calls.
- Expected per-turn wire ranking: `stateless` (system + full history + q) >
  `interaction` (system + q) > `cached` (cache-ref + q) — with `cached` carrying the
  stage-2 upload as setup.

This is why the report keeps **wire bytes** and **tokens** on separate axes and
never collapses them into one "traffic = cost" number.

## Fixed by the probe

**Host: `https://generativelanguage.googleapis.com/v1beta` (Gemini Developer API).**
Auth is an API key (`x-goog-api-key`), not ADC.

**Model: `gemini-3.1-flash-lite`, on every arm.**

Vertex is out. The GEAP Interactions reference (`aiplatform.googleapis.com`) lists
only `lyria-3-clip-preview` and `lyria-3-pro-preview` under `model`, and only
`deep-research-preview-04-2026` under `agent` — music generation and one research
agent. No Gemini text model. A Google engineer confirmed the same on the developer
forum in January 2026. There is no pure-model Interactions target on Vertex today.

That forces every arm onto the Developer API. Leaving `stateless` and `cached` on
Vertex while `interaction` ran on the Developer API would put a different host,
auth path, and network route on either side of the comparison, and the latency
numbers would measure that difference instead of the strategies.

`gemini-3.1-flash-lite` is the cheapest model that (a) still serves new API keys and
(b) is accepted by `interactions`, `generateContent`, and `cachedContents` alike.
$0.25 / 1M input, $1.50 / 1M output.

**Retired for new keys** (404 `no longer available to new users`):
`gemini-2.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`. Also
`gemini-3-pro-preview` (404 `no longer available`). `ListModels` still advertises
all of them; only a generation call reveals the retirement, and only once billing
lets the request through.

## Measured, not assumed (probe run 2026-07-10)

| Question | Answer | How we know |
|---|---|---|
| Does `interactions` take a plain `model`? | **Yes** | 200 with `model: gemini-3.1-flash-lite` |
| Is `stream: true` supported? | **Yes** | 200, SSE, `usage` present |
| Is `stream: false` supported? | **Yes** | 200, single JSON, top-level `usage` |
| Does the response carry token counts? | **Yes** | `usage.{total_input_tokens, total_cached_tokens, total_output_tokens, total_thought_tokens, total_tool_use_tokens, total_tokens}` plus `input_tokens_by_modality` |
| Same model on `generateContent`? | **Yes** | 200 |
| Does `system_instruction` survive `previous_interaction_id`? | **No** | see below |

**Both stream modes work.** Choosing `stream: false` is a decision, not a
limitation: it makes wire bytes and total latency mean the same thing on the
`interaction` arm as on the `generateContent` arms — one request, one response, no
SSE framing, no ambiguity about whether "done" means first token or last.

## What `previous_interaction_id` actually carries

**Only the conversation history** — the user inputs and model outputs of the chained
interactions. Nothing else.

Everything else is *interaction-scoped*: it applies to the one interaction you are
generating right now, and it is gone on the next call. Per the reference, that
covers `system_instruction`, `tools`, and `generation_config` (including
`thinking_level` and `temperature`).
<https://ai.google.dev/gemini-api/docs/interactions-overview>

`system_instruction` is **not a required field** — a call without it returns 200. But
a stateful conversation that wants its system prompt in force must re-send it on
every turn, because the server does not keep it.

### How the probe established this

Asking the model to reveal a codeword from its system prompt does not work: the model
refuses, and a refusal is indistinguishable from the prompt having been dropped. The
first version of this probe did exactly that and drew no conclusion.

Nor does instructing an unconditional marker ("end every reply with `ZQ7`"). The
marker then appears in turn 1's *output*, which lands in the history that turn 2 can
see, and a model imitating its own previous format looks exactly like a server that
kept the instruction. That version reported `persisted` — wrongly.

The rule has to be **conditional and untriggered on turn 1**, so that nothing about it
reaches the history:

- `system_instruction`: "if the user's message is exactly `BANANA`, reply with only `ZQ7`".
- Control: same instruction and `BANANA` in one request → the model answers `ZQ7`. The rule is obeyed.
- Turn 1: `"Say hello."` with the instruction → `"Hello! How can I help you today?"`. No marker. The history learns nothing.
- Turn 2: `previous_interaction_id`, **no** `system_instruction`, message `BANANA` → `"BANANA! 🍌 That is a very enthusiastic fruit."`

The rule is gone. Twice, on two independent runs, with the control passing both
times. The docs are right.

### What that costs

The `interaction` arm uploads the system prompt on all 10 turns. Its byte savings
over `stateless` come from the conversation history alone — never from the system
prompt. That is why `cached` can beat `interaction` on bytes: an explicit cache holds
the system prompt, and `previous_interaction_id` does not.

## The two axes must stay separate

A stateful API moves bytes off the wire but does not necessarily move tokens off the
bill: the server still feeds the whole history to the model, and implicit caching
discounts the token bill without touching the wire. The report carries two
independent series and never collapses them:

- **wire bytes** — what crosses the socket, request and response, headers included.
- **input tokens** — what the model was fed (`total_input_tokens`), of which
  `total_cached_tokens` was served from cache (implicit or explicit).

`PROJECT_GOAL.md`'s implicit "traffic = cost" identity breaks for every arm that
caches, in either direction.

## The `cachebuild` stage (stage 2) and its cost

Stage 3 (`cached`) can only reference a cache that stage 2 created. Stage 2's
`cachedContents` create/refresh calls upload the prefix, and that upload is real
traffic — for a cumulative-prefix cache, summed over the turns, it approaches the
same O(N²) upload volume as `stateless`, just moved to a different endpoint and paid
up front.

So stage 2 is measured as `cached`'s **setup bucket** — every create/refresh call's
bytes, tokens, and milliseconds. `cached`'s reported cost is *stage 2 setup + its own
per-turn calls*. Hiding stage 2 would make `cached` look free when its savings are
merely front-loaded.

## Scenario fixture

New file `requests/perf.json`:

- **system prompt + persona**: ≥ 4,096 tokens. Gemini 3.x models require **4096**
  input tokens before caching engages (implicit *and* explicit); below that a
  `cachedContents` create is rejected and implicit caching never fires, so the whole
  comparison would be caching-free and pointless. Target **~5,000 tokens (~20,000
  characters)** for margin — a real, coherent system prompt and persona, padded with
  a deterministic block so runs reproduce.
- **steps**: 10 turns, 500 characters each.

`requests/default.json` is untouched.

## Non-goals

- Answer quality. `nocontext` deliberately produces worse answers.
- Cost in dollars. `cost_usd` is dropped from the comparison output. Tokens are still
  recorded, as a diagnostic axis, not the headline.
- Agent-mode interactions (remote sandbox, `background: true`). That code stays but
  `/compare` never exercises it.

## Bug fix: wire byte counting is wrong today

`gemini_client.py:320` clears `_active_counter` before every call, but
`_CountingHTTPSConnection.connect()` only installs a counter when a **new** TCP
connection opens. `requests` pools connections, and `reset_session()` is called once
per stage (`experiment.py:141,152`). So every turn after the first on a given socket
reads `counter is None` and silently falls back to `wire_sent = req_payload_bytes` —
the JSON body, no HTTP headers, no content-encoding. The existing stateless-vs-cached
wire numbers are already contaminated.

Fix: keep the `_CountingSocket` on the connection and snapshot/diff it around each
request instead of relying on `connect()` firing.

```python
@contextmanager
def wire_counter():
    """Bytes on the wire for the enclosed request, keep-alive or not."""
    before = _snapshot_all_counters()
    w = _Delta()
    yield w
    w.sent, w.recv = _diff(before, _snapshot_all_counters())
```

The experiment loop is single-threaded, so a global snapshot/diff is exact. All call
sites — `call_gemini`, `create_cache`, `_call_interaction` — go through it, so every
arm is measured by the same instrument. `inspector.py` reads `_active_counter`
directly and gets ported in the same change.

## Common record schema

Every arm emits the same per-turn dict:

```python
{
  "arm": str,             # stateless | cachebuild | cached | interaction | nocontext
  "turn": int,
  "wire_sent": int,       # HTTP request bytes on the socket, headers included
  "wire_recv": int,       # HTTP response bytes on the socket
  "elapsed_ms": int,      # request start -> last byte
  "input_tokens": int,
  "cached_tokens": int,   # implicit or explicit; nonzero even on stateless is expected
  "output_tokens": int,
  "thought_tokens": int,
  "total_tokens": int,
  "request_raw": str,     # exact request body sent (no auth headers)
  "response_raw": str,    # exact response body received
  "error": str,
}
```

Token field mapping:

| common | `generateContent` | `interactions` (non-stream) |
|--------|-------------------|-----------------------------|
| `input_tokens` | `usageMetadata.promptTokenCount` | `usage.total_input_tokens` |
| `cached_tokens` | `usageMetadata.cachedContentTokenCount` | `usage.total_cached_tokens` |
| `output_tokens` | `usageMetadata.candidatesTokenCount` | `usage.total_output_tokens` |
| `thought_tokens` | — | `usage.total_thought_tokens` |
| `total_tokens` | `usageMetadata.totalTokenCount` | `usage.total_tokens` |

With `stream: false` the `usage` object sits at the top level of the response body.
(Streaming also reports usage, on a terminal event; the comparison does not use it.)

`interaction_client.py` parses no usage at all today — it walks the payload for text
and discards the rest. It gains a usage reader.

`CallResult` in `gemini_client.py` gains `elapsed_ms`; it has no timing field today,
which is why no `generateContent` latency exists to compare against.

## Setup vs steady state

- **setup** — one-off costs charged before or during turn 1:
  - `cached`: the stage-2 `cachebuild` calls (create/refresh) plus the final delete.
  - `interaction`: turn 1, which opens the interaction.
  - `stateless`, `nocontext`: empty.
- **steady** — turns 2..N. Mean, median, min, max per arm.

`REPEATS = 1`. Traffic and token counts are deterministic for a fixed fixture, so one
run settles the primary question. Latency is reported as an observed value, not a
statistic, and the report says so. Generation settings are not pinned beyond the
model id, so output and thought tokens vary run to run; that variance lands in
latency.

Total wall-clock per arm (setup + all turns) is also reported, since that is what a
user waits.

## Endpoints and UI

- `POST /interaction/probe` → `probe.probe_interactions()`. Already shipped.
- `POST /compare`, `POST /compare/stream` → `experiment.run_comparison(model, turns, arms)`.
  Reuses the SSE progress plumbing from `/run/stream`.
- `metrics.summarize_comparison(records)` → per-arm series, plus `setup` and `steady`
  buckets. No `cost_usd`.
- UI: one table (arm × turn) and two charts sharing the turn axis — cumulative wire
  bytes, cumulative input tokens — with a toggle between them.

Each arm calls `reset_session()` before its first request, so it opens a fresh TCP
connection and a per-arm pcap stays attributable.

## Error handling

- An arm that errors records the error per turn and keeps going; the summary marks the
  arm partial rather than dropping the run.
- A `cached` create call returning `below_min` is a bug given the 12K system prompt;
  surface it loudly, do not silently fall back.

## Testing

Mock mode (`GEMINI_MOCK=1`) covers the wiring; no live calls in CI.

- `tests/test_wire_counter.py` — two sequential requests on one keep-alive connection
  each report nonzero, distinct, header-inclusive byte counts. This test fails against
  today's code, which is the point.
- `tests/test_compare_schema.py` — every arm emits the common record schema, including
  `request_raw` / `response_raw`.
- `tests/test_metrics_compare.py` — setup/steady bucketing; stage-2 `cachebuild` calls
  land in `cached`'s setup rather than vanishing.
- `tests/test_probe.py` — already shipped.

## Still open

- **Explicit-cache minimum token count for `gemini-3.1-flash-lite`.** The published
  table lists 4096 for Gemini 3.x (3.5 Flash, 3.1 Pro) and 2048 for 2.5, but
  **flash-lite is not listed**. `gemini_client.MIN_CACHE_TOKENS = 2048` is a stale
  Vertex constant. Verify with one live create before trusting the `cached` arm: if
  flash-lite's floor is 4096, the ~5,000-token fixture clears it, but a create must
  confirm rather than assume. If a create returns below-min, the fixture's system
  prompt is enlarged, not silently fallen back.
- **Interactions accepts only implicit caching.** Settled: the caching guide states
  explicit caching is unavailable on the Interactions API, so there is no
  `interaction` + `cachedContents` arm. The `cached_content` field in the reference is
  a `generateContent` construct, not an interactions one.
- **Latency baseline for the host.** Everything runs against
  `generativelanguage.googleapis.com`; absolute numbers are specific to that route.

## Build order

1. `wire_counter()` + `CallResult.elapsed_ms` + port `inspector.py`. Tests first.
2. Point `gemini_client` at the Developer API host; API-key auth; `cachedContents` there.
3. `requests/perf.json` fixture.
4. `interaction_client`: usage parsing, `stream: false`, `system_instruction` on every
   turn, drop warmup/`background` from the model path.
5. `experiment.run_comparison` with the four arms and setup/steady buckets.
6. `metrics.summarize_comparison`, `/compare`, UI table and charts.
