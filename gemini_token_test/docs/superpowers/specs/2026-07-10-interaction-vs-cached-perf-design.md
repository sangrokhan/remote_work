# Interaction API vs stateless / cachedContents — traffic & latency comparison

Date: 2026-07-10
Status: design, awaiting approval

## Goal

Measure, on one GCP project with one set of credentials, how much **network traffic**
and **wall-clock latency** each conversation-state strategy costs for the same
10-turn scenario:

1. `stateless` — resend the whole history every turn.
2. `cached` — Vertex `cachedContents` holds the prefix; each turn sends `cachedContent` + the new question.
3. `interaction` — Interactions API holds the history server-side; each turn sends `previous_interaction_id` + the new question.
4. `nocontext` — send only the new question. Lower bound; answer quality is irrelevant here.

Optional 5th arm, off by default: `cached-sysonly` (see "Cache shape" below).

## Non-goals

- Answer quality. `nocontext` deliberately produces worse answers.
- Cost in dollars. `cost_usd` is dropped from the comparison output. Tokens are
  still recorded, but as a diagnostic axis, not the headline.
- Agent-mode interactions (`INTERACTION_AGENT`, remote sandbox, `background=true`).
  That code stays but is never exercised by `/compare`.

## The two axes must stay separate

A stateful API moves bytes off the wire but does **not** necessarily move tokens off
the bill: the server still feeds the whole history to the model. The report therefore
carries two independent series and never collapses them:

- **wire bytes** — what crosses the socket, request and response, headers included.
- **input tokens** — what the model was fed (`total_input_tokens`), of which
  `total_cached_tokens` was served from cache.

`PROJECT_GOAL.md`'s implicit "traffic = cost" identity holds for `stateless` vs
`nocontext` and breaks for `cached` and `interaction`. The spec makes that visible
rather than hiding it.

## Cache shape — why `cached` may not save bytes

The existing `run_three_stage` builds a **cumulative prefix cache per turn**: before
turn k it POSTs the full prefix (system + turns 1..k-1) to `cachedContents`. That
upload is real traffic. Summed over 10 turns it is the same O(N²) upload volume as
`stateless`, just moved to a different endpoint.

So the `cached` arm gets an explicit **setup bucket** that counts every
`cachedContents` create call — bytes, tokens, and milliseconds. Hiding those calls
would make the arm look free when it is not.

`cached-sysonly` (optional arm) caches only the 12K-char system prompt, once, and
resends the conversation history in `contents` each turn. It is the only cache usage
that actually reduces upload bytes. Default off; enable with `COMPARE_ARMS`.

## Scenario fixture

New file `requests/perf.json`:

- **system prompt**: 10,000–15,000 characters (target 12,000). Padded with a
  deterministic, non-random filler block so runs are reproducible. At ~4 chars/token
  this is ~3,000 tokens, comfortably above `MIN_CACHE_TOKENS = 2048`, so the `cached`
  arm's cache is valid from turn 1 instead of falling back to a full prefix for the
  first eight turns.
- **steps**: 10 turns, 500 characters each.

`requests/default.json` is untouched; the existing `/run` flow keeps its fixture.

## Capability probe

### What the docs already settle

- **`model` is a documented field on the GEAP surface.** The Interactions API
  reference describes `model` as "The model that will complete your prompt" and marks
  `agent` as "Required if `model` is not provided" — the two are alternatives, not
  agent-only. The agent guide page happens to show only `agent` + `environment`
  examples, which is what made it look unsupported.
  <https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/interactions-api>

- **`system_instruction` must be re-sent every turn.** Under the heading *Server-side
  state management*: "The `previous_interaction_id` parameter preserves only the
  conversation history (inputs and outputs) […] The other parameters are
  **interaction-scoped** and apply only to the specific interaction you are currently
  generating" — listing `tools`, `system_instruction`, and `generation_config` —
  "This means you must re-specify these parameters in each new interaction if you want
  them to apply."
  <https://ai.google.dev/gemini-api/docs/interactions-overview>

- **`usage` is returned on the `interaction.complete` event**, carrying
  `total_tokens`, `total_input_tokens`, `total_output_tokens`, `total_thought_tokens`.
  <https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/managed-agents/interact-with-agents>

### What remains unmeasured

`stream:false` on the GEAP host, a regional location, whether `usage` is populated for
a **model** interaction (documented examples are agent + streaming), and which model
ids both APIs accept. Those are measured, not assumed.

Probe 5 (system-prompt persistence) stays even though the docs answer it, because the
answer is load-bearing for the byte math and the API is preview-stage. Doc says
re-send; the probe confirms the server actually behaves that way.

`probe.py` exposes one function, `probe_interactions()`, reachable two ways:

- `POST /interaction/probe` — a button in the UI. This is the only user-facing entry
  point. Cloud Run has no shell, so a Makefile target would be useless there.
- called once automatically at the start of `/compare`; the result is cached in
  memory and embedded in the saved run JSON as `capabilities`, so any later reading
  of the numbers can be audited against what the API actually supported that day.

### Probes

| # | Probe | Question |
|---|-------|----------|
| 1 | `model` + `stream:true` | Sanity check — the field is documented; confirm it returns 200 on this project. |
| 2 | `model` + `stream:false` | Is a single-JSON, non-streaming response available? |
| 3 | `locations/us-west1/interactions` | Is a regional location accepted? |
| 4 | `system_instruction` field | Is the field accepted (vs. 400)? |
| 5 | system-prompt persistence | Send `system_instruction` containing a secret codeword on turn 1; on turn 2 send only `previous_interaction_id` + "what is the codeword?". If the model answers correctly, the system prompt persists server-side and need not be resent. |
| 6 | `usage` in `interaction.complete` | Are `total_input_tokens` / `total_cached_tokens` / `total_output_tokens` / `total_thought_tokens` present? |
| 7 | model-id support matrix | For each candidate model id, does `generateContent` accept it, and does `interactions` accept it? |

Per the doc quoted above, `system_instruction` is interaction-scoped: the `interaction`
arm uploads the 12K system prompt on all 10 turns, and its byte savings come from the
history alone. Probe 5 confirms the server matches the doc before the numbers are
trusted. Design the arm for re-sending; if probe 5 shows persistence, revisit.

### Verdict logic

Auth failures and schema failures must not be confused. Classify on status:

| status | body signal | verdict |
|--------|-------------|---------|
| 401 | UNAUTHENTICATED | `environment` — bad/absent/expired token. Abort, report nothing about schema. |
| 403 | PERMISSION_DENIED | `environment` — token valid, IAM role missing (`roles/aiplatform.user`). |
| 403 | SERVICE_DISABLED | `environment` — `aiplatform.googleapis.com` not enabled on the project. |
| 403 | allowlist / preview wording | `unavailable` — the project is not admitted to the Interactions preview. This *is* the finding; report it with the response body. |
| 400 | INVALID_ARGUMENT | `unsupported` — the field is genuinely rejected. This is the signal we want. |
| 404 | — | `unsupported` for probe 3 (no such regional resource); otherwise a path/revision typo. |
| 200/201 | — | `supported`. |

The probe result renders verbatim in the UI, response bodies included. No summarizing
away the error text.

### Fallbacks driven by the probe

- Probe 1 fails → the comparison cannot run on `aiplatform`. That is itself the
  answer: "no pure model interaction on Vertex today." Report and stop.
- Probe 2 fails → `interaction` streams; total latency is measured to the arrival of
  the `interaction.complete` event. `generateContent` stays non-streaming. The SSE
  framing overhead (event names, blank lines, `data:` prefixes) is reported in its own
  column, `sse_overhead_bytes`, so the byte asymmetry is stated rather than buried.
- Probe 3 fails → both arms run at `locations/global` (set `VERTEX_LOCATION=global`),
  keeping the endpoint family identical. If it succeeds, both run at `us-west1`.
- Probe 5 fails (no persistence) → `interaction` resends `system_instruction` every
  turn, and the report says so.
- Probe 7 → the run picks a model id supported by **both** APIs. If only `gemini-3-*`
  works on Interactions, `stateless`, `cached`, and `nocontext` all use that same
  `gemini-3-*` id. Same model on every arm, always. The support matrix ships in the
  run JSON.

## Bug fix: wire byte counting is wrong today

`gemini_client.py:320` clears `_active_counter` before every call, but
`_CountingHTTPSConnection.connect()` only installs a counter when a **new** TCP
connection is opened. `requests` pools connections, and `reset_session()` is called
only once per stage (`experiment.py:141,152`). So every turn after the first on a
given socket reads `counter is None` and silently falls back to
`wire_sent = req_payload_bytes` — the JSON body with no HTTP headers and no
content-encoding. The existing `stateless` vs `cachedContents` wire numbers are
already contaminated by this.

Fix: keep the `_CountingSocket` alive on the connection and snapshot/diff it around
each request instead of relying on `connect()` firing.

```python
@contextmanager
def wire_counter():
    """Bytes on the wire for the enclosed request, keep-alive or not."""
    before = _snapshot_all_counters()
    w = _Delta()
    yield w
    w.sent, w.recv = _diff(before, _snapshot_all_counters())
```

The experiment loop is single-threaded, so a global snapshot/diff is exact. All three
call sites — `call_gemini`, `create_cache`, and `_call_interaction` — go through it,
so every arm is measured by the same instrument. `_call_interaction` currently uses
`_session()` already, so it inherits the fix for free once the counter is correct.

`inspector.py` reads `_active_counter` directly; it gets ported to `wire_counter()`
in the same change.

## Common record schema

Every arm emits the same per-turn dict, so the summary code has one shape to handle:

```python
{
  "arm": str,             # stateless | cached | interaction | nocontext | cached-sysonly
  "turn": int,
  "wire_sent": int,       # HTTP request bytes on the socket, headers included
  "wire_recv": int,       # HTTP response bytes on the socket
  "sse_overhead_bytes": int,   # 0 for non-streaming arms
  "elapsed_ms": int,      # request start -> last byte
  "first_byte_ms": int,   # request start -> first response byte / first SSE event
  "input_tokens": int,
  "cached_tokens": int,
  "output_tokens": int,
  "thought_tokens": int,
  "total_tokens": int,
  "error": str,
}
```

Token field mapping:

| common | generateContent | interactions |
|--------|-----------------|--------------|
| `input_tokens` | `usageMetadata.promptTokenCount` | `interaction.usage.total_input_tokens` |
| `cached_tokens` | `usageMetadata.cachedContentTokenCount` | `interaction.usage.total_cached_tokens` |
| `output_tokens` | `usageMetadata.candidatesTokenCount` | `interaction.usage.total_output_tokens` |
| `thought_tokens` | — (absent) | `interaction.usage.total_thought_tokens` |
| `total_tokens` | `usageMetadata.totalTokenCount` | `interaction.usage.total_tokens` |

`interaction_client.py` today parses no usage at all — it walks the SSE payload for
text and discards everything else. It gains a `_usage(events)` reader for the
`interaction.complete` event.

`CallResult` in `gemini_client.py` gains `elapsed_ms` and `first_byte_ms`; it has no
timing fields today, which is why no `generateContent` latency exists to compare
against.

## Setup vs steady state

Latency and traffic are reported in two buckets, per the agreed measurement rule:

- **setup** — one-off costs charged before or during turn 1:
  - `cached`: every `cachedContents` create call (all 10 of them, for the cumulative
    variant), plus the delete.
  - `interaction`: turn 1, which opens the interaction.
  - `stateless`, `nocontext`: empty. Their turn-1 cost is an ordinary turn.
- **steady** — turns 2..N. Reported as mean, median, min, max per arm.

`REPEATS = 1` by default, so p95 is not computed; it would be noise. The UI exposes no
repeat field. Traffic and token counts are deterministic for a fixed fixture, so a
single run settles the primary question; latency is reported as an observed value, not
a statistic, and the report labels it that way.

The run also reports plain total wall-clock per arm (setup + all turns), since that is
what a user actually waits.

Model generation settings are **not** pinned (same model id only, per decision). Output
and thought tokens will vary run to run, and that variance lands in latency. Stated in
the report header.

## Endpoints and UI

- `POST /interaction/probe` → `probe.probe_interactions()`. Button renders the raw
  verdict table plus every response body.
- `POST /compare` and `POST /compare/stream` → `experiment.run_comparison(model, turns, arms)`.
  Reuses the existing SSE progress plumbing from `/run/stream`.
- `metrics.summarize_comparison(records)` → per-arm series keyed by the common schema,
  plus `setup` and `steady` buckets. No `cost_usd`.
- UI: one table (arm × turn) and two charts sharing the turn axis — cumulative wire
  bytes, and cumulative input tokens. A toggle switches the chart between the two.
  The existing chart component is reused.

Each arm calls `reset_session()` before its first request, so it opens a fresh TCP
connection and a per-arm pcap stays attributable. The existing capture toggle applies.

## Error handling

- Any arm that errors records the error per turn and keeps going; the summary marks
  the arm partial rather than dropping the run.
- A `cached` create call returning `below_min` is a bug given the 12K system prompt;
  it is surfaced loudly, not silently fallen back.
- Probe verdict `environment` aborts `/compare` before any billable call.

## Testing

Mock mode (`GEMINI_MOCK=1`) covers the wiring; no GCP calls in CI.

- `tests/test_wire_counter.py` — two sequential requests on one keep-alive connection
  each report nonzero, distinct, header-inclusive byte counts. This test fails against
  today's code, which is the point.
- `tests/test_probe.py` — each status/body combination in the verdict table maps to the
  right verdict; 401 aborts.
- `tests/test_compare_schema.py` — all four arms emit the common record schema with the
  same keys.
- `tests/test_metrics_compare.py` — setup/steady bucketing, and `cached`'s create calls
  land in `setup` rather than vanishing.
- `preflight.sh` gains none of these as new steps; `make test` already discovers them.

## Risks

- **Probe 1 may still fail** despite `model` being documented — the API is preview and
  the project may not be admitted. Then there is no pure-model Interactions target on
  this project and the comparison is impossible as specified. The fallback is not to
  switch to `generativelanguage.googleapis.com` — different auth, different network
  path, different billing — because that would silently change what is being measured.
  It becomes the reported finding instead.
- **`usage` may be absent for model interactions.** Every documented `usage` example is
  agent + streaming. If probe 6 comes back empty, the token axis dies for the
  `interaction` arm and only the wire-byte and latency axes survive — which still
  answers the primary question.
- **`us-west1` may not exist for interactions**, in which case both arms move to
  `global` and the regional question goes unanswered.
- **Single run**, unpinned generation settings: latency is indicative, not conclusive.
- **`cached` may show no byte savings.** Expected, and the reason `cached-sysonly`
  exists as an option.

## Build order

1. `wire_counter()` + `CallResult` timing + port `inspector.py`. Tests first.
2. `probe.py` + `POST /interaction/probe` + UI button. Run it against the real project.
3. Branch on the probe result before writing anything else.
4. `requests/perf.json` fixture.
5. `interaction_client` usage parsing; drop warmup/`background` from the model path.
6. `experiment.run_comparison` with the four arms and setup/steady buckets.
7. `metrics.summarize_comparison`, `/compare`, UI table and charts.
