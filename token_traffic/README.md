# token_traffic

What a multi-turn conversation costs on the wire, per way of keeping it, across two
providers, on the same fixture.

A conversation has to be kept somewhere. The client can keep it and resend it every
turn; the server can keep it and be handed a pointer; the prefix can be parked in an
explicit cache; or nothing can be kept at all. Each choice puts a different number of
bytes up the wire, gets billed a different number of input tokens, and makes the user
wait a different amount of time. This lab measures those three things — and nothing
else. Answer quality is never scored.

It is the provider-agnostic successor to `gemini_token_test/` and `openai_token_test/`:
one core (socket byte counter, SSE clock, records, metrics, store, capture, runner,
web app) and two thin provider adapters that declare their arms and build their bodies.

## What it measures

For every (provider, arm, turn):

- **Uplink and downlink bytes** (`wire_sent`, `wire_recv`), counted on the socket
  itself — headers and content-encoding included, after TLS decryption. `len(json)` of
  the body would compare the arms on a quantity none of them pays. Uplink is the axis
  the arms actually differ on: a resent history is upload; a stored one is not.
- **Tokens**, straight from the provider's own usage block, translated into one neutral
  vocabulary (`input_tokens`, `cached_tokens`, `output_tokens`, `reasoning_tokens`,
  `total_tokens`). Gemini's "thought" tokens and OpenAI's "reasoning" tokens are the
  same column here. No prices, no ratios: a dollar figure built on a per-token rate
  nobody verified is a guess dressed up as a result.
- **Five marks, not one elapsed number** — `req_sent_ms`, `ttfb_ms`, `ttft_ms`,
  `ttlt_ms`, `turn_end_ms`, plus the derived `store_tail_ms` (`turn_end − ttlt`). One
  number cannot tell a history still going up the wire apart from a model thinking
  apart from a server persisting the turn.
- **Optionally, the packets** — one pcap per arm, via tcpdump, so the in-process byte
  count can be checked by somebody who does not trust this code.

The scenario is one fixture (`fixtures/perf.json`): a 20,653-character system prompt
(persona, tool descriptions, operating policy — over 4k tokens, so both implicit and
explicit caching engage) and ten questions that lean on each other through pronouns and
ellipsis. Every arm of every provider replays the same system prompt and the same
questions in the same order, or the comparison means nothing.

## The arms

### Gemini — `generativelanguage.googleapis.com`, API key

One host, one auth, one network path. Vertex and ADC are not options here: a different
host on a different route with different latency, and it does not serve plain-model
Interactions at all, so half the arms would be measuring a network the other half never
touched.

| arm | what each turn sends | what it is evidence of |
|---|---|---|
| `stateless` | `generateContent` with system + every prior Q and A + the new question | the baseline every SDK chat loop implements, and the arm whose upload grows O(N²) |
| `nocontext` | `generateContent` with the bare question; the system prompt rides turn 1 alone | the floor. Not a way anyone runs a chat — it says how much of a turn is the *question* rather than the *context*. Out of the headline set |
| `cached` | `generateContent` with the new question and `cachedContent` pointing at a `cachedContents` resource holding the prefix | the only arm that takes the system prompt off the wire **and** out of the per-turn input bill |
| `interaction` | `/interactions` with `system_instruction` + the new question + `previous_interaction_id`, `store: true` | what `previous_interaction_id` actually buys: the *history* bytes and nothing else. `system_instruction` is interaction-scoped and is re-uploaded on every turn — so the 20 KB prompt goes up ten times anyway |
| `interaction_inline` | the same, with the system prompt inlined into the first user turn so the server stores it; later turns send only their question | the fix for the line above. Same content reaches the model; a different party keeps it |
| `interaction_stateless` | `/interactions` with the whole conversation as `Step[]`, `store: false`, no `previous_interaction_id` | the control. Same endpoint, same body shape, server-side state removed — so the gap against `interaction` is `previous_interaction_id` alone |

Two rules the Gemini arms keep, and why:

1. **Echo what the server sent.** When an arm keeps the history client-side, the model's
   turn goes back on the wire exactly as it came off it — the `thought` step with its
   signature, or the parts carrying `thoughtSignature`. Measured: echoing it is
   accepted, costs zero extra input tokens, and restores no reasoning the answer text
   did not already carry. What it changes is roughly 1 KB of **upload** per turn — which
   a real client pays. An arm that rebuilds the turn from the answer text reports a
   smaller upload than the client it claims to measure.
2. **The answer is the answer.** A `thought` part never enters the transcript, never
   starts the TTFT clock, and never lands in a cache built from it.

### OpenAI — `api.openai.com`, API key

| arm | what each turn sends | what it is evidence of |
|---|---|---|
| `chat_stateless` | `POST /v1/chat/completions`, `messages = [system, u1, a1, …, uk]` | the classic chat loop: O(N²) upload |
| `responses_stateless` | `POST /v1/responses`, `store: false`, `input = [system, u1, a1, …, uk]` | the control: the same payload as `chat_stateless` on a different endpoint, so any byte gap against `responses_stateful` is about server-side state and not about which endpoint the bytes went through |
| `responses_stateful` | `POST /v1/conversations` once (seeded with the system prompt), then `POST /v1/responses` with `conversation=conv_…`, `input = [uk]` | the finding: the uploaded bytes collapse to O(N) while `usage.input_tokens` does not. OpenAI bills every previous input token in the chain. The bytes can already be saved; the billing does not follow |

Two measurement choices that are not incidental:

- The `requests` library, never the official SDK. The SDK rides on httpx, and the socket
  counter is an http.client/urllib3 subclass that cannot attach to it.
- The system prompt is byte-identical on every turn and `prompt_cache_key` is pinned per
  arm. OpenAI's prompt cache matches on an exact prefix, so a timestamp anywhere in the
  system prompt would miss the cache every turn and turn `cached_tokens` into noise; and
  unkeyed, whether an identical prefix hits depends on which node the call lands on
  (measured live: 2/5 unkeyed, 4/5 keyed). The key is distinct per arm so that whichever
  arm runs first cannot warm the cache for the next one and make it look cheaper for no
  reason but its position.

## The two rules that keep the numbers honest

### 1. Dual-pass measurement

Two of the measurements want opposite things from the same call, and no single request
can serve both.

**Bytes want a blocking call.** OpenAI's streamed deltas carry `include_obfuscation`
padding — a side-channel mitigation, on by default — which inflates the SSE frames by an
amount that has nothing to do with the conversation. Both providers frame their streams.
A streamed byte count is a measurement of the vendor's framing policy, not of the
conversation.

**Latency wants a streamed call.** Time-to-first-token cannot be read off a blocking
response at all.

So `measure` says what a turn pays for:

| `measure` | calls per turn | what the record carries |
|---|---|---|
| `bytes` (default) | 1, blocking | bytes and tokens; the marks are 0, and `measure` on the row is what says they are absent rather than instantaneous |
| `latency` | 1, streamed | the five marks and tokens; the bytes are the streamed framing, and are not the same quantity |
| `both` | 2 | bytes and body from the blocking pass, marks from the streamed pass. The streamed pass's byte counts are dropped, not added: their sum is a number no client ever pays |

`both` doubles the API bill, so it is never a default. And on **`openai:responses_stateful`
it is not merely expensive but wrong**: every pass carries the conversation id, OpenAI
appends each of them to the server-side history (`store: false` is not allowed alongside
a conversation), so the second call of turn *k* makes turn *k+1*'s `input_tokens` count
turn *k* twice. `core.runner.warnings_for()` refuses to let that go out unannounced — the
CLI prints the warning before the dry run ends, and `POST /api/preflight` returns it.
Run that arm with `bytes` or `latency`.

### 2. The measurement window

`wall_ms` and the pcap cover the **steady** turns only.

Prep is not traffic. A Gemini cache build re-uploads the whole prefix, so building a
cache per turn costs O(N²) in setup alone; an OpenAI conversation create uploads the
system prompt once. Both are real calls, both are recorded — an API call the run made and
did not report is a hole in the evidence — but they carry a prep phase (`cachegen`,
`setup`) and `core.metrics` never folds a prep phase into a total. Teardown (the cache
DELETEs) is outside the window too.

The window is defined by the arm's own progress events: the runner opens the pcap and
starts the `wall_ms` clock on the arm's first `steady` event, and closes both on a
`teardown` event (or when the arm returns). An arm with neither prep nor teardown is
captured whole, which is correct — all of it is traffic.

Around that window, `core.wire.reset_session()` drops the pooled TLS connection, twice:
once before the capture opens, so the pcap starts from a handshake instead of onto an
established connection with prep's FIN in it; and once before tcpdump is stopped, so the
FIN that ends the last measured turn is inside the capture that measured it. Each arm
therefore gets a fresh connection, and one arm's teardown cannot land in the next arm's
pcap.

## Running it

Nothing is billable unless you say so twice: mock mode is how the suite is meant to be
run, and `python cli.py` without `--go` calls nothing at all.

```sh
pip install -r requirements.txt

# A dry run: what would be called, how many calls that is, and every warning —
# and then it stops. This is the default.
python cli.py

# Offline, synthetic, no keys, no quota. Shaped like the real thing.
TRAFFIC_MOCK=1 python cli.py --go

# The only thing that spends money.
GEMINI_API_KEY=... OPENAI_API_KEY=... python cli.py --go --measure bytes

# One provider, chosen arms, a shorter thread.
python cli.py --go --providers gemini --arms stateless,cached --turns 3

# The web UI (host/port from TRAFFIC_HOST / TRAFFIC_PORT).
python cli.py --serve
```

`cli.py` flags: `--providers`, `--arms` (of a single provider), `--measure`
(`bytes`|`latency`|`both`), `--fixture`, `--turns` (truncates the thread — it never
cycles: a repeated question is answered from context and costs nothing like a new one),
`--capture`, `--pause SEC` (a gap between arms, to stay under a rate limit), `--go`,
`--serve`. It exits non-zero if any record carried an error.

The web UI is a thin skin over the same runner. `GET /api/config` is the preflight (what
is ready, what capture would do, the retention limit); `POST /api/preflight` reports what
a specific selection would cost in calls and warnings before anything goes out;
`POST /api/run/stream` runs it as server-sent events, because a ten-turn comparison
across nine arms takes minutes and a UI with no progress is a UI the operator reloads
mid-run — which abandons the request but not the calls.

Packet capture needs the `tcpdump` binary and `NET_RAW`. Locally:
`sudo setcap cap_net_raw,cap_net_admin+eip $(which tcpdump)`. In Docker:
`--cap-add=NET_RAW`. Where it is unavailable, `core.capture.available()` says why and the
run proceeds without it — the byte counts come from the socket and stand on their own.

In the container the app is served by gunicorn with **one worker**: the socket byte
counter is a module-global tally and a second worker would interleave two requests into
one count.

### Make targets

The pre-merge gate is `make preflight` (unit tests, image build, mock-container smoke);
`make test` runs the unit tests alone, `make build` builds the image, `make clean` removes
what they made. None of them makes a live API call.

## Environment

Every knob below is read from the environment at the point named. Anything not listed
here does not exist.

| variable | default | what it does |
|---|---|---|
| `TRAFFIC_MOCK` | unset | `1`, `true`, `yes` or `on` puts the whole suite in mock mode: no call leaves the process, and the run is stored in the mock bucket and labelled everywhere it appears. Every module reads the flag through `core.config`, so there is one answer — two parsers once disagreed about `true`, and half a run was billed while the other half was synthetic and the whole thing was filed as live |
| `GEMINI_MOCK` | unset | mocks Gemini alone, while OpenAI still talks to its API |
| `OPENAI_MOCK` | unset | the same, for OpenAI alone. A run with any synthetic call in it is filed as mock: its numbers cannot be charted against measured ones |
| `GEMINI_API_KEY` | unset | sent as `x-goog-api-key`. Without it, and without a mock flag, the Gemini arms report `not_ready` and are skipped rather than failing the run |
| `OPENAI_API_KEY` | unset | sent as `Authorization: Bearer`. Same not-ready behaviour |
| `TRAFFIC_DATA_DIR` | `data/runs` | where run JSON is written. Mock runs go in the `mock/` subdirectory of it |
| `TRAFFIC_RETENTION_KEEP` | `20` | how many runs to keep **per bucket**. `save_run` prunes on every write. Mock runs cannot evict live ones |
| `TRAFFIC_PCAP_DIR` | `data/pcaps`, or a temp dir | where pcaps are written, re-read on every capture |
| `TRAFFIC_PCAP_DISABLE` | unset | `1` reports capture unavailable without probing for it |
| `TRAFFIC_PCAP_IFACE` | `any` | the interface tcpdump listens on |
| `TRAFFIC_PCAP_SNAPLEN` | `100` | bytes kept per packet. The TLS payload is encrypted and useless to store; the L2–L4 and TLS record headers are not. Each frame still records its original on-wire length, so packet sizes stay exact, and truncating slashes the disk I/O that causes kernel drops |
| `TRAFFIC_HOST` | `127.0.0.1` | `--serve` bind address |
| `TRAFFIC_PORT` | `8080` | `--serve` port. (The container's gunicorn uses `PORT` instead) |
| `KEEP_CACHE` | unset | `1` leaves the Gemini `cached` arm's `cachedContents` resources behind instead of deleting them at teardown. A cache left behind is a bill nobody is watching, so this is for debugging only |

The pcap directory default has a trap worth knowing about. Ubuntu's tcpdump AppArmor
profile carries `audit deny @{HOME}/.*/** mrwkl`, and in AppArmor deny beats allow — so a
checkout living under a dot-directory in `$HOME` cannot host the pcap directory at all:
tcpdump gets `NET_RAW`, opens its socket, then dies on the output file with a bare
"Permission denied" that reads exactly like the capability never took. `core.capture`
detects that case, falls back to a temp directory, and — if `TRAFFIC_PCAP_DIR` points
somewhere the profile forbids — reports it as the reason capture is unavailable.

Provider-local knobs, for completeness: `GEMINI_API_HOST`, `GEMINI_API_SCHEME`,
`CACHE_TTL_SECONDS` (1800), `MIN_CACHE_TOKENS` (2048 — below it, a cache build is skipped
rather than eating a 400, and the record says so), `OPENAI_BASE_URL`, `OPENAI_MODEL`
(`gpt-4.1-nano`), `OPENAI_MAX_OUTPUT_TOKENS` (400), `OPENAI_REASONING_EFFORT` (empty; the
parameter must not be sent at all to a non-reasoning model, or the call 400s),
`OPENAI_TIMEOUT` (180), `TRAFFIC_CAPTURE_PROBE_TTL` (30).

## What a run produces

One JSON document per run, `records.csv`, `summary.csv`, and optionally one pcap per arm.
A mock run is stored in its own bucket, listed separately, tagged in its CSV filename,
and is never charted or averaged with a live one. See `docs/outputs.md`.

## Mock mode is shaped like the real thing

A mock that let the expensive arms look free would be worse than no mock at all. So:

- A Gemini stored interaction holds the stream open ~1.8 s after its last token, because
  the real one does (measured — see below).
- A model turn comes back with a `thought` step carrying a realistic-length signature,
  because that blob is most of what echoing a model turn costs.
- Gemini's input tokens are counted from **what the payload actually carries** — a prefix
  that lives in a cache is billed as cached, a history that was never sent is billed to
  nobody.
- OpenAI's mock keeps the server-side conversation and bills `input_tokens` from the
  **full** history, never from what came up the wire. A mock that billed the upload would
  erase the entire finding.
- `measure=both` against `openai:responses_stateful` double-appends in the mock too. The
  point of a mock here is to reproduce what the numbers will look like, and that includes
  the ways they can be wrong.

## What the predecessors already measured

Live against `gemini-3.1-flash-lite`, 2026-07-10 to 2026-07-14. These are the results
this package was built to carry forward and re-measure across two providers.

**`system_instruction` does not persist.** `GET /interactions/{id}` on a stored
interaction returns `steps` and nothing else — `system_instruction: None`, `tools: null`.
A conditional-rule probe confirms the rule is gone on the chained turn. So a chained
conversation re-sends its instructions every single turn, and `previous_interaction_id`
saves only the history bytes. That is why `interaction_inline` and `cached` exist.

**The thought signature costs upload, not tokens.** Turn 2 of a two-turn conversation,
sent three ways: echoing the response's steps verbatim → 1,634 request bytes; a rebuilt
`model_output` step with the thought step dropped → 632 bytes; chained via
`previous_interaction_id` → 457 bytes. All three billed the same 62 input tokens, all
three answered correctly. The echo buys nothing except honesty about what a real client
uploads.

**`store: true` costs ~1.8 s, and it is a tail cost.** Medians of 7, decoding pinned:
`generateContent` with full history 601 ms; interactions with a client history and
`store:false` 854 ms; the same with `store:true` **2,685 ms**; chained via
`previous_interaction_id` (which forces `store:true`) 2,699 ms. So `store` costs
+1,831 ms and `previous_interaction_id` costs +14 ms — nothing. The cost is constant, not
proportional to what is stored. Streamed (medians of 4), the last text delta arrives at
951 ms with `store:true` versus 1,131 ms without; the stream then stays open until
2,800 ms while the server persists the interaction. A streaming client never feels it —
which is why the published examples, all streamed, look fast. A blocking client waits out
the whole tail on every turn. `store_tail_ms` is the column that separates the two.

And a chained conversation cannot opt out: `store:false` with `previous_interaction_id`
is rejected — `400 "store must be true when previous_interaction_id is set."`

## Layout

```
token_traffic/
  cli.py             dry run by default; --go spends money; --serve starts the UI
  core/
    wire.py          counting socket: bytes, and when the request left / the reply began
    streaming.py     SSE reader: the answer, and the marks that bracket it
    call.py          one turn on the wire, in one or two passes
    record.py        the per-turn record every arm produces, and its schema version
    metrics.py       per-(provider, arm) series and totals; prep never folded in
    store.py         one run, one JSON file; retention; a wall between live and mock
    capture.py       tcpdump around one arm's steady stage
    runner.py        the plan, the fresh connection, the measurement window
    scenario.py      the fixture every arm replays
    export.py        records.csv and summary.csv
    app.py           Flask: preflight, run, history, download
  providers/
    base.py          the Provider protocol; get(name) is the only way to reach one
    gemini.py        stateless, nocontext, cached, interaction, interaction_inline,
                     interaction_stateless
    openai.py        chat_stateless, responses_stateless, responses_stateful
  fixtures/perf.json the shared scenario
  docs/
    core-contracts.md  what every provider may rely on, and must supply
    outputs.md         what a run produces, and how it is managed
```
