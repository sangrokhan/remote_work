# Gemini traffic lab

A measurement rig for one question: **what does it actually cost to carry a
conversation forward?** The same N-turn conversation is replayed across six
mechanisms — resend the whole history, resend nothing, cache the history, let the
server keep it — and every turn is measured at the socket, at the token counter,
and at five points on the clock.

Everything runs against the **Gemini Developer API**
(`generativelanguage.googleapis.com`) with an API key. There is no Vertex, no ADC,
no service account, no Firestore, and no cost estimate. Every arm sits on one host,
one auth, one network path; otherwise the latency numbers would be comparing the
route rather than the mechanism.

See [`PROJECT_GOAL.md`](PROJECT_GOAL.md) for why the question matters and
[`docs/call-flow.md`](docs/call-flow.md) for the per-arm request diagrams.

## What it measures

One run replays one scenario (`requests/perf.json`: a large system prompt, then a
list of user turns) across six arms, back to back, on the same model. Each arm gets
a fresh TCP connection so its traffic is separately attributable — and, with capture
on, separately capturable.

| Arm | What the client sends each turn | Who holds the history |
|-----|--------------------------------|-----------------------|
| `stateless` | system prompt + every prior question and answer + the new question | the client, in full, every turn |
| `nocontext` | the new question alone (the system prompt rides turn 1 only) | nobody — the floor, not a real strategy |
| `cached` | a `cachedContent` reference + the new question | the server, in an explicit cache built beforehand |
| `interaction` | `previous_interaction_id` + the new question | the server (`store:true`) |
| `interaction_inline` | same, but the system prompt is the first user turn instead of `system_instruction` | the server, system prompt included |
| `interaction_stateless` | the whole conversation as `Step[]`, `store:false`, no chaining | the client — the stateless bargain, made on the Interactions endpoint |

`stateless`, `nocontext` and `cached` go to `:streamGenerateContent`; the three
`interaction*` arms go to the Interactions endpoint. **Every arm streams**, because
TTFT cannot be measured any other way — and without TTFT the stored arms get charged
for a write their user never actually waits for.

The `cached` arm builds its caches in a **prep** phase before the measured window
opens (from the transcript `stateless` produced, so all arms answer the same
conversation) and deletes them in a teardown phase after it closes. Those calls are
recorded with `phase=cachegen` and are reported separately — never folded into the
totals, because each build re-sends the whole system prompt and n turns of setup
cost O(n²) bytes that no measured turn pays.

### Per turn

- **`wire_sent` / `wire_recv`** — bytes of the HTTP request and response as they
  cross the TLS stream, **headers included**, body content-encoded (i.e. the real
  transferred size, not the decoded JSON size). Counted by wrapping the socket, so
  no tcpdump or `NET_RAW` is needed. This is post-decryption HTTP framing, not raw
  ciphertext; for true packet sizes, use the pcap.
- **Tokens** — `input_tokens`, `cached_tokens`, `output_tokens`, `thought_tokens`,
  `total_tokens`, from the response's usage metadata.
- **Five latency marks**, not one `elapsed`, because a single number cannot tell
  "my history is still going up the wire" apart from "the model is thinking":

  | Mark | When |
  |------|------|
  | `req_sent_ms` | the client's request is fully on the wire — a resent history pays here |
  | `ttfb_ms` | the first response byte comes back |
  | `ttft_ms` | the answer's first token |
  | `ttlt_ms` | the answer's last token — what a streaming user waits for |
  | `turn_end_ms` | the server closes the stream — what a blocking client waits for |

  and one derived: **`store_tail_ms` = `turn_end_ms` − `ttlt_ms`**. It is ~0 on the
  `generateContent` arms (nothing happens after the last token) and ~1.8 s on a
  stored interaction, where the write lands *after* the answer is already out. A
  streaming client never waits for that tail; a blocking one pays it every turn.

## Running it

### Mock first (no key, no quota, no bill)

```bash
pip install -r requirements.txt
GEMINI_MOCK=1 python app.py        # http://localhost:8080
```

`GEMINI_MOCK=1` runs the whole flow synthetically: every arm, every route, every
export, every download. The mock timings are fixed but shaped like the real thing —
the upload mark scales with payload size and the stored arms still pay their
`store_tail_ms` — so a mock run cannot pretend the stored arms are free. Use it for
anything that is not specifically about live numbers.

### Live

```bash
export GEMINI_API_KEY=...
python app.py                      # http://localhost:8080
```

Open the page, pick a model and a turn count, hit start. The run streams per-arm
progress over SSE (`POST /compare/stream`) and lands a JSON document in
`data/runs/`.

> **House rule: no live run without a reason.** The project key has a monthly spend
> cap. Every live call is billable, and a 6-arm run multiplies whatever you were
> going to spend by six. Mock mode exists precisely so that nothing but a genuine
> measurement question ever reaches the network.

### Docker

```bash
cp .env.example .env               # or just: GEMINI_MOCK=1
docker compose up --build          # http://localhost:8080
```

Compose auto-loads `.env` and already adds `NET_RAW` for packet capture. Build args
`BASE_IMAGE`, `PIP_INDEX_URL`, `PIP_TRUSTED_HOST` and `APT_MIRROR` let the image
build behind a mirror; see `.env.example`.

## Packet capture (optional)

Tick **capture packets** before starting and `tcpdump` runs around each arm's
measured window — **one pcap per arm**, each a self-contained SYN..FIN conversation.
That is what lets the socket-level byte counter be cross-checked against what
actually went out on the wire. Download links appear per arm when the run finishes.

The TLS payload is encrypted; packet **sizes and timing** are the proof, which is
why the default snaplen is 100 bytes (the original on-wire length is still recorded,
so sizes stay exact). Requires raw-socket capability: `sudo`, or
`setcap cap_net_raw+ep $(which tcpdump)`, or Docker (already configured). In mock
mode no real traffic leaves the process, so the pcap is empty.

## What comes out

Every run is written to `data/runs/<exec_id>.json` — params, summary, and one record
per call including the raw request and response bodies.

| Endpoint | What it gives you |
|----------|-------------------|
| `GET /download/comparison/<exec_id>.csv` | **metrics CSV** — one row per call |
| `GET /download/comparison/<exec_id>-responses.csv` | **responses CSV** — one row per turn, one column per arm |
| `GET /download/comparison/<exec_id>.json` | **cases JSON** — every call with its raw request and response parsed back into objects |
| `GET /download/chat/<exec_id>` | **chat JSON** — per-turn question/answer plus the raw bodies |
| `GET /download/run/<exec_id>` | the stored run document, verbatim |
| `GET /download/pcap/<name>` | a captured pcap |
| `GET /history` | every stored run: `exec_id`, timestamp, mode, mock flag, totals |
| `GET /history/<exec_id>` | one run document |
| `DELETE /history/<exec_id>` | delete one run |

(`GET /download/compare/<exec_id>` still exists and serves a turn/query/stateless/
stateful CSV, but only an older 3-stage run has the fields it reads. Nothing
produces those any more.)

### The metrics CSV, column by column

`arm`, `phase`, `turn`, `wire_sent`, `wire_recv`, `req_sent_ms`, `ttfb_ms`,
`ttft_ms`, `ttlt_ms`, `turn_end_ms`, `store_tail_ms`, `elapsed_ms`, `input_tokens`,
`cached_tokens`, `output_tokens`, `thought_tokens`, `total_tokens`, `error`.

- `arm` — which of the six.
- `phase` — `steady` for a measured turn, `cachegen` for one of the `cached` arm's
  prep builds. Filter to `steady` for anything you intend to compare.
- `turn` — 1-based step in the conversation.
- `wire_sent` / `wire_recv` — socket bytes, headers included (above).
- The five marks and `store_tail_ms` — as defined above; `elapsed_ms` is the whole
  call and is kept only so a row is never marks-only.
- Token columns — from usage metadata. `input_tokens` includes `cached_tokens`.
- `error` — empty on a good call. A run with a broken arm still returns numbers, and
  numbers from a failed call look exactly like numbers from a good one, so check
  this column (and `summary.failures`) before believing a chart.

### The responses CSV

One row per turn; columns `turn`, `question`, then `<arm>_response` and
`<arm>_request` for each arm. The metrics CSV says what each arm *spent*; it cannot
say whether the arms were having the same conversation. An arm that quietly degraded
— a cache that never hit, a history the server dropped — still produces perfectly
reasonable-looking bytes. Reading the answers side by side is the only way to catch
it. `cachegen` rows are excluded: a cache build answers nothing.

### Clearing the runs

There is **no retention policy**. Runs pile up in `data/runs/` — a 3-turn, 6-arm run
is a few hundred KB, because the raw request and response bodies of every call are
stored. Delete them yourself:

```bash
rm -rf data/runs/*.json data/pcaps/*        # or: curl -X DELETE localhost:8080/history/<exec_id>
```

## Tests

230 tests, all offline (mock mode), no key and no network:

```bash
python -m pytest tests -q
```

Two caveats about the existing wrappers, both stale as of this writing:

- `make test` runs `python -m unittest discover -s tests`. The suite is
  pytest-style (plain `def test_*` functions); `unittest` discovers only the one
  `TestCase` class in `tests/test_capture.py`, so this reports success while running
  almost nothing. Use `pytest`.
- `./preflight.sh` (also `make preflight`) runs unit tests, builds the Docker image,
  starts a mock container, then smoke-tests `POST /run` and `POST /inspect` — two
  routes that no longer exist in `app.py`. It will fail at that step.

## Probes

`probe.py` is a set of one-shot diagnostics, run from the shell. **Every probe is a
real, billable call** — they exist to answer questions the comparison run cannot,
not to be run on a whim.

```bash
python3 -c "import json, probe; print(json.dumps(probe.probe_step_echo(), indent=2))"
```

- `probe_interactions()` — which Interactions surface, if any, serves a plain Gemini
  model. The model catalog never advertises an interactions method, so calling it is
  the only way to know. This one is also wired to `GET|POST /interaction/probe` and
  runs on page load, cached (`PROBE_CACHE_TTL`).
- `probe_step_echo()` — will the server accept a client-built `Step[]` history?
- `probe_signature_echo()` — what happens to the model's own thought step (and its
  ~1 KB signature) when the client rebuilds the history from response text alone.
- `probe_hidden_state()` — does that signature carry reasoning the answer text never
  showed? Turn 1 hides a number, turn 2 asks for it back.
- `probe_latency_matrix()` — holds the conversation fixed and varies one field at a
  time (`store`, `previous_interaction_id`, client history) to find which of them
  buys the seconds.
- `probe_stream_ttft()` — does the `store` write happen before the first token or
  after the last one? (After. Which is why `store_tail_ms` is a column.)

Findings are written up in [`docs/interactions-api-fields.md`](docs/interactions-api-fields.md).

## Configuration

Every variable the code actually reads, and nothing else.

| Var | Default | Meaning |
|-----|---------|---------|
| `GEMINI_API_KEY` | — | Developer API key. Required for a live run. |
| `GEMINI_MOCK` | `0` | `1` = synthetic responses; no key, no network, no bill. |
| `GEMINI_API_HOST` | `generativelanguage.googleapis.com` | Override to point tests at a local server. |
| `GEMINI_API_SCHEME` | `https` | Override to drive a TLS-less local server. |
| `GEMINI_DATA_DIR` | `data/runs` | Where run JSON is written. |
| `PORT` | `8080` | Server port. |
| `CACHE_TTL_SECONDS` | `1800` | TTL on the caches the `cached` arm builds. |
| `KEEP_CACHE` | — | `1` = skip the teardown that deletes them. |
| `MIN_CACHE_TOKENS` | `2048` | Minimum cacheable prefix; below this the API refuses a cache. |
| `INTERACTION_TIMEOUT` | `180` | Per-call timeout, seconds, on the Interactions arms. |
| `PCAP_DIR` | *(auto)* | Normally `data/pcaps`; a temp dir if the checkout sits under a dot-directory in `$HOME` (tcpdump's AppArmor profile denies `@{HOME}/.*/**`). Set only to override. |
| `PCAP_IFACE` | `any` | tcpdump capture interface. |
| `PCAP_SNAPLEN` | `100` | Bytes captured per packet. Header-only; the TLS payload is encrypted anyway, and truncating cuts the disk I/O that causes kernel drops. Original packet length is still recorded, so sizes stay exact. |
| `PCAP_DISABLE` | `0` | `1` = hide the capture toggle. |
| `PCAP_SETTLE_SECONDS` | `1.0` | Beat between an arm's socket closing and its capture stopping, so the FIN makes it into the file. |
| `CAPTURE_PROBE_TTL` | `30` | How long the "is capture available?" answer is cached. |
| `PROBE_MODELS` | *(the default model)* | Comma-separated model ids for `probe_interactions`. |
| `PROBE_TIMEOUT` | `60` | Per-call timeout, seconds, in the probes. |
| `PROBE_THINKING_LEVEL` | `high` | Thinking level for the signature probe — a signature is only worth echoing if there is thinking behind it. |
| `PROBE_CACHE_TTL` | `600` | How long the page-load probe result is held before a refresh re-bills it. |
