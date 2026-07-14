# Project Goal — what a conversation costs, per way of keeping it

## The question

A multi-turn conversation with an LLM has to be kept somewhere. The client can keep
it and resend it every turn; the server can keep it and be handed a pointer; the
prefix can be parked in an explicit cache; or nothing can be kept at all. Each of
those choices sends a different number of bytes up the wire and makes the user wait
a different amount of time.

This project measures that, for one scenario, on one host, with one auth, against
one model — so that the only thing varying between arms is **who keeps the
conversation**. It answers, per arm and per turn:

- what the **client uploads** (`wire_sent`), and what comes back (`wire_recv`);
- what the model is **billed for** (input / cached / output / thought tokens);
- and **when things happen** — five marks on every turn, not one elapsed number,
  because one number cannot tell a history still going up the wire apart from a
  model thinking apart from a server persisting the turn.

It is a traffic and latency experiment, not a chat-quality one. Answer correctness
is never scored; the `nocontext` arm deliberately has no memory at all.

## Where it runs

Everything talks to the **Gemini Developer API**
(`generativelanguage.googleapis.com`), authenticated with `x-goog-api-key`. That is
the only host that serves plain-model Interactions, and every arm has to sit on one
host, one auth and one network path or the latency numbers compare nothing. Vertex
AI and ADC are gone from the project entirely. The model is
`gemini-3.1-flash-lite` (`gemini_client.DEFAULT_MODEL`); `GEMINI_MOCK=1` runs the
whole flow offline against synthetic responses shaped like the real ones.

## The scenario

One fixture, `requests/perf.json`: a **20,653-character system prompt** (persona +
tool descriptions — roughly 4.4k input tokens, comfortably above the cache floor)
and **10 questions**. Every arm replays the same system prompt and the same
questions in the same order. `experiment.run_comparison()` is the whole experiment.

## The arms

| arm | what each turn sends | who keeps the history |
|---|---|---|
| `stateless` | system prompt + every prior Q and A + the new question, via `generateContent` | the client |
| `nocontext` | the bare question (the system prompt rides turn 1 alone) | nobody |
| `cached` | the new question, referencing an explicit `cachedContents` resource holding the prefix | an explicit cache |
| `interaction` | `system_instruction` + the new question + `previous_interaction_id`, `store:true` | the server |
| `interaction_inline` | the same, but the system prompt is inlined into the **first user turn** so it becomes part of the stored history; later turns send only their question | the server |
| `interaction_stateless` | the whole conversation as `Step[]` on the interactions endpoint, `store:false`, no `previous_interaction_id` | the client |

Why each one exists:

- **`stateless`** is the baseline every SDK chat loop actually implements, and the
  arm whose upload grows quadratically.
- **`nocontext`** is the lower bound. It is not a usable chat; it is the floor that
  says how much of a turn's cost is the *question* rather than the *context*.
- **`cached`** is the only arm that can take the system prompt off the wire *and*
  out of the per-turn input bill. Caches are built in a prep stage from the real
  `stateless` transcript (a cache of a conversation that never happened measures
  nothing), outside the measured window, and reported separately as `cachegen_*`.
- **`interaction`** is `previous_interaction_id` used the way the docs show it. It
  saves the *history* bytes and nothing else: `system_instruction`, `tools` and
  `generation_config` are interaction-scoped and re-uploaded on every single turn
  (measured — `docs/interactions-api-fields.md`), so a 20 KB system prompt goes up
  ten times anyway.
- **`interaction_inline`** exists precisely because of that. Move the system prompt
  into the first user message and the server stores it as history; every later turn
  is then just a question. Same content reaches the model; a different party keeps
  it.
- **`interaction_stateless`** is the control that isolates what
  `previous_interaction_id` actually buys: same endpoint, same body shape, but the
  client resends everything and the server stores nothing.

## What is measured, and how

**Wire bytes.** A counting socket (`gemini_client._CountingSocket` /
`_CountingReader`, installed into the `requests` pool via a custom connection class)
tallies every byte of the HTTP exchange, headers and content-encoding included, on
both the `send` and the `makefile` read paths. It is post-decryption HTTP framing,
not TLS ciphertext. `wire_sent` — the client's own uplink — is the axis the arms
differ on; `wire_recv` is roughly the same work whoever keeps the history. Reported
per turn and cumulative (`metrics.summarize_comparison`).

**Tokens.** Straight from each response's usage: `promptTokenCount` /
`cachedContentTokenCount` / `candidatesTokenCount` / `thoughtsTokenCount` on
`generateContent`, and `total_input_tokens` / `total_cached_tokens` /
`total_output_tokens` / `total_thought_tokens` on `/interactions`. No cost model,
no USD estimate, no ratio metric — the raw counts are the finding.

**Latency, five marks.** Every arm streams (`:streamGenerateContent?alt=sse` and
`stream:true` on interactions), because TTFT cannot be read off a blocking call.
`streaming.read_stream` times each turn:

| mark | what it is |
|---|---|
| `req_sent_ms` | the request is fully on the wire — the client's history has finished uploading |
| `ttfb_ms` | the first response byte comes back — network + queue, no tokens yet |
| `ttft_ms` | the first event carrying **answer** text (a `thought` part does not start this clock) |
| `ttlt_ms` | the last event carrying answer text — **what a streaming user waits for** |
| `turn_end_ms` | the stream closes — **what a blocking client waits for** |

`store_tail_ms = turn_end_ms − ttlt_ms`. On the `generateContent` arms it is zero:
nothing happens after the last token. On a stored interaction it is not, and that
gap is the whole latency finding below.

**Packet capture (optional).** `capture.py` runs `tcpdump` filtered to the API host,
one pcap per arm, around the arm's steady stage only. Each arm opens a fresh TCP
connection and closes it inside its own capture window (`_close_connection` with a
settle), so a pcap is one self-contained SYN..FIN conversation and can be
cross-checked against the socket counter. Needs `NET_RAW`; the experiment runs fine
without it.

**Scope discipline.** `wall_ms` and the pcap cover only the **steady** stage — the
measured turns. The `cached` arm's cache builds (prep) and cache deletes (teardown)
run outside both windows and are never folded into an arm's totals.

## What the probes settled

`probe.py` exists to replace assumption with measurement. Live against
`gemini-3.1-flash-lite`; the tables are in `docs/interactions-api-fields.md`.

- **The endpoint and auth.** Vertex was probed and never served plain-model
  Interactions. `POST https://generativelanguage.googleapis.com/v1beta/interactions`
  with `x-goog-api-key` is the one target; no interactions method is ever advertised
  in the model catalog, so calling it was the only way to know.
- **Instruction fields do not persist.** `GET /interactions/{id}` on a stored
  interaction returns `steps` and nothing else: `system_instruction: None`,
  `tools: null`. A conditional-rule probe (turn 1 states the rule and does not
  trigger it; turn 2 chains with no `system_instruction` and triggers it) confirms
  the rule is gone on turn 2. So a chained conversation re-sends its instructions
  every turn.
- **`input` accepts a client-supplied history.** `store:false`, no
  `previous_interaction_id`, three steps including a `model_output` the client wrote
  → 200, and the model resolved "And of Italy?" to `Rome.` — it read the step. That
  is the `interaction_stateless` arm's mechanism, confirmed.
- **The model's turn comes back as two steps.** A `thought` step carrying an
  encrypted `signature`, then `model_output` (on `generateContent`: a part with
  `thoughtSignature` alongside the text). Echoing it verbatim is accepted, costs **0
  extra input tokens**, and — per `probe_hidden_state` (`signature_carries_nothing`)
  — restores no reasoning the text had not already carried. Dropping it is also
  accepted. What it changes is the **upload**: ~1 KB per turn. The arms echo it
  because a real client does, and an arm that quietly omits it reports a smaller
  upload than the client it claims to measure.
- **What `store` costs, and where it lands.** See below.

## Findings

### The byte story held

A client-kept history is O(N²) upload: turn *k* re-sends turns 1..k−1. The arms that
hand the history to the server or to a cache send the new question and nothing else.
That was the original hypothesis and the measurement backs it.

But it is narrower than it looks. `previous_interaction_id` removes only the
*history* from the wire. The system prompt is not history — it is an
interaction-scoped instruction field — so the `interaction` arm re-uploads all 20,653
characters of it on all ten turns, and saves almost nothing against `stateless` in
this scenario. Only `cached` (prefix in a cache) and `interaction_inline` (prefix
inlined into turn 1, hence stored) actually take the system prompt off the wire.
That is why those two arms exist.

### The latency story did not

The original hypothesis had bytes driving everything. Measured, latency is driven by
a field that has nothing to do with bytes.

**Which field buys the seconds** (`probe_latency_matrix`, `stream:false`, decoding
pinned, medians of 7):

| cell | median |
|---|---|
| `generateContent`, full history | 601 ms |
| interactions, client history, `store:false` | 854 ms |
| interactions, client history, **`store:true`** | **2,685 ms** |
| interactions, **`previous_interaction_id`** + `store:true` | 2,699 ms |

`store` costs **+1,831 ms**. `previous_interaction_id` costs **+14 ms** — nothing.
And the store cost is **constant, not proportional to what is stored**: adding the
20,653-character system prompt moves it by ~280 ms (856→957 ms unstored, 2,372→2,653
ms stored). It is a fixed per-turn write, not a function of the payload.

**Where the 1.8 s sits** (`probe_stream_ttft`, `stream:true`, medians of 4, ~18-char
answer):

| | first text | stream closed |
|---|---|---|
| `store:false` | 1,131 ms | 1,335 ms |
| `store:true` | **951 ms** | **2,800 ms** |

The answer reaches a streaming client at the same time either way — if anything
sooner with `store:true`. The write happens **after the last text delta**: the SSE
stream simply stays open (`step.stop` → `interaction.completed` → `[DONE]`) while
the server persists the interaction. So:

- a **streaming** client never waits for the write — which is why the published
  interactions examples, all of them streamed, look fast;
- a **blocking** client pays the full ~1.8 s on every single turn;
- and a chained conversation **cannot opt out**: `store:false` together with
  `previous_interaction_id` is rejected with
  `400 "store must be true when previous_interaction_id is set."`

That is why every arm here streams and why every turn carries five marks instead of
one. Reporting a single `elapsed` would charge the interaction arms for a write
their user never waits for; reporting only TTFT would hide the write entirely. The
`store_tail_ms` column is the number that separates the two, and it is the reason
the goal of this project is now *bytes × latency*, not bytes alone.

### The honest-client correction

Until 2026-07-14 the client-history arms rebuilt the model's turn from its answer
text, which dropped the thought step and its signature — about 1 KB of upload per
turn that a real client pays. The arms now echo the model's turn exactly as the
server sent it. The token bill and the answers are unchanged (measured); the
reported upload is not.

## Next: one lab, many providers

The same question — who keeps the conversation, and what does it cost — is being
asked of OpenAI in `openai_token_test/` with a second copy of the socket counter,
the timings, the metrics and the app. The plan is a provider-agnostic core (wire,
streaming, record, metrics, store, capture, app) with thin provider adapters that
declare their arms and build their bodies, plus **dual-pass measurement**: bytes
from a non-streamed call, latency from a streamed one, because OpenAI's SSE deltas
carry `include_obfuscation` padding that corrupts a byte count while TTFT cannot be
read off a blocking call. Details, phases and the arm-to-arm mapping:
`docs/superpowers/plans/2026-07-14-provider-agnostic-traffic-lab.md`.
