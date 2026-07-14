# Core contracts

What every provider may rely on, and what every provider must supply. Written
before the code so that the two provider adapters and the core can be built against
the same spec instead of against each other; kept current with it since.

## The shape of the thing

```
token_traffic/
  cli.py           dry run by default; --go spends money; --serve starts the UI
  core/
    config.py      one reading of the mock switch, for everybody
    wire.py        counting socket: bytes, and when the request left / the reply began
    streaming.py   SSE reader: the answer, and the marks that bracket it
    call.py        one turn on the wire, in one or two passes
    record.py      the per-turn record every arm produces, and its schema version
    metrics.py     per-(provider, arm) series and totals
    store.py       one run, one JSON file, with a retention policy
    capture.py     tcpdump around an arm
    scenario.py    the fixture every arm replays
    runner.py      replay one scenario across providers x arms
    export.py      records.csv and summary.csv
    app.py         Flask: preflight, run, download, history
  providers/
    base.py        the Provider protocol
    gemini.py      stateless, nocontext, cached, interaction, interaction_inline,
                   interaction_stateless
    openai.py      chat_stateless, responses_stateless, responses_stateful
  fixtures/
    perf.json      the shared scenario: one system prompt, N questions
  tests/
```

What a run produces, and how it is stored and pruned: `docs/outputs.md`.

## Why two passes

Two measurements want opposite things from the same call.

**Bytes** want a blocking call. OpenAI's streamed deltas carry `include_obfuscation`
padding, which pads the SSE frames and destroys the byte measurement the experiment
exists to make.

**Latency** wants a streamed call. Time-to-first-token cannot be read off a blocking
response, and on a stored Gemini interaction the difference matters enormously: the
last token of the answer lands ~950 ms in, and the stream then stays open ~1.8 s
longer while the server persists the interaction. A blocking client waits for all of
it; a streaming one does not. Reporting one number would hide which.

So `measure` selects what a turn pays for:

| `measure` | calls per turn | what the record carries |
|---|---|---|
| `bytes` | 1, blocking | wire_sent / wire_recv / tokens; marks are 0 |
| `latency` | 1, streamed | the five marks / tokens; bytes are the streamed framing, not comparable across providers |
| `both` | 2 | bytes from the blocking pass, marks from the streamed pass, tokens from the blocking pass |

`both` doubles the API bill, so it is never the default. A run records its `measure`
in `params`, and the CSV carries it per row: a bytes column from a streamed pass and
one from a blocking pass are not the same measurement and must never be averaged
together.

On one arm `both` is not merely expensive but wrong. On `openai:responses_stateful`
every pass carries the conversation id, and OpenAI appends each of them to the
server-side history (`store: false` is not allowed alongside a conversation), so the
second call of turn *k* makes turn *k+1*'s `input_tokens` count turn *k* twice.
`core.runner.warnings_for()` is where that is caught, and nothing may run it without
saying so first.

## `core.wire`

```python
@contextmanager
def wire_counter() -> _WireDelta:
    """Count HTTP bytes on the socket for the enclosed request, headers and
    content-encoding included, and stamp when the request finished going out and
    when the first byte came back."""

class _WireDelta:
    sent: int              # bytes the client put on the wire
    recv: int              # bytes it read back
    last_send_at: float | None    # monotonic: request fully written
    first_recv_at: float | None   # monotonic: first response byte

def session() -> requests.Session:
    """Session whose pools use the counting connection classes."""

def reset_session() -> None:
    """Drop pooled sockets so the next call opens a fresh TCP connection. Every arm
    calls this before its capture window, or the pcap opens onto an established
    connection and misses the handshake."""
```

The counter must survive keep-alive: it counts on the socket, not per connection.

## `core.streaming`

```python
@dataclass
class StreamResult:
    status: int
    req_sent_ms: int      # set by core.call, from the wire marks
    ttfb_ms: int
    ttft_ms: int          # first event carrying ANSWER text
    ttlt_ms: int          # last event carrying answer text
    turn_end_ms: int      # stream closed
    text: str
    events: list          # parsed `data:` payloads, in order
    raw: str              # the SSE body, for the audit trail
    error: str

def read_stream(resp, text_of, t0) -> StreamResult
def since(t0, mark, fallback=0) -> int
```

`text_of(event) -> str` is the provider's: it pulls the answer text out of one event
and nothing else. Reasoning text (a Gemini `thought` part, an OpenAI reasoning
summary) is not the answer and must not start the TTFT clock.

## `core.call`

```python
@dataclass
class Exchange:
    status: int
    error: str
    wire_sent: int
    wire_recv: int
    req_payload_bytes: int
    resp_payload_bytes: int
    req_sent_ms: int
    ttfb_ms: int
    ttft_ms: int
    ttlt_ms: int
    turn_end_ms: int
    elapsed_ms: int
    text: str             # the answer
    response: dict        # the body a blocking call would have returned
    request_json: str
    response_json: str

def send(url, headers, body, *, measure, text_of, stream_body=None,
         stream_url=None, rebuild=None, timeout=180) -> Exchange
```

- `body` is the blocking request. `stream_body` is the same request with whatever the
  provider needs to make it stream (`stream: true`); when `measure` is `latency` or
  `both`, that is what goes out for the streamed pass. `stream_url` is where it goes
  when streaming lives at a different endpoint — Gemini's is `:streamGenerateContent`,
  not `:generateContent`, so the two passes of a `both` turn do not even share a URL.
- `rebuild(events) -> dict` turns the streamed events back into the body a blocking
  call would have returned. It is not a convenience: the Gemini interactions endpoint
  streams the model's steps and its completed event does **not** carry them, so the
  steps a client-side history must echo exist only as deltas that went past.
- `send` never raises. A failed call comes back with `error` set and the marks pinned
  to the moment it ended, because a zero mark reads as "instant" rather than "never".

## `core.record`

One record per (provider, arm, turn, pass). The CSV is one row per record.

```python
SCHEMA_VERSION = 1

def turn_record(provider, arm, phase, turn, question, measure, exchange, usage,
                extra=None) -> dict
```

| field | meaning |
|---|---|
| `schema_version` | so a run from an older layout can be told apart, not silently charted |
| `provider`, `arm`, `phase`, `turn` | `phase` is `steady` or a prep phase (e.g. `cachegen`, `setup`) |
| `measure` | `bytes`, `latency`, or `both` |
| `wire_sent`, `wire_recv` | socket bytes; uplink is the axis the arms differ on |
| `req_payload_bytes`, `resp_payload_bytes` | decoded body sizes |
| `req_sent_ms`, `ttfb_ms`, `ttft_ms`, `ttlt_ms`, `turn_end_ms` | the five marks |
| `store_tail_ms` | `turn_end - ttlt`: what a blocking client waits for after the answer |
| `input_tokens`, `cached_tokens`, `output_tokens`, `reasoning_tokens`, `total_tokens` | from the provider's own usage block |
| `question`, `response_text` | so a run can be audited: did the arms have the same conversation? |
| `request_raw`, `response_raw` | the evidence |
| `error` | empty when the call succeeded |

`reasoning_tokens` is the provider-neutral name (Gemini calls it thought tokens).

## `providers.base`

```python
class Provider(Protocol):
    NAME: str                       # "gemini" | "openai"
    DEFAULT_MODEL: str
    ARMS: tuple[str, ...]
    HEADLINE_ARMS: tuple[str, ...]  # what a default run includes

    def ready() -> tuple[bool, str]
    def api_host() -> str
    def run_arm(arm, model, system, steps, measure, on_progress) -> list[dict]

def names() -> tuple[str, ...]
def get(name) -> module          # KeyError on an unknown name: get(user_input) must
                                 # not be a way to import an arbitrary module
def progress(on_progress, provider, arm, phase, turn, turns) -> None
```

`run_arm` owns the conversation: it decides what turn k sends, it echoes the model's
turn back when the arm keeps the history client-side, it builds and tears down any
server-side state (a Gemini cache, an OpenAI conversation), and it returns records
built with `core.record.turn_record`. Everything else — bytes, marks, storage,
metrics, capture — is core's.

`api_host()` is the host the arms talk to. Capture needs it to filter tcpdump down to
this run's traffic, and core must not have to know which provider is running to build
that filter.

`ready()` returns `(False, reason)` and never a bare `False`: a run that dies on a
missing key must say which key. A provider that is not ready is skipped with a
`not_ready` record, not fatal to the run.

`progress(event)` is announced **before** each call, not after — a UI has to be able
to say "turn 3 of 10, in flight" while the call is still out. The event carries
`{provider, arm, phase, turn, turns}`, and `phase` must be exact, because the runner
uses it to bound the measurement window (below). A provider that mislabels a phase does
not produce a wrong number — the socket counter and the records still phase correctly —
it produces a pcap that disagrees with them, which is worse, because the pcap is what
the numbers are checked against.

Two rules a provider must not break:

1. **Echo what the server sent.** When an arm keeps the history client-side, the
   model's turn goes back on the wire exactly as it came off it: Gemini's `thought`
   step with its signature, OpenAI's reasoning item. Rebuilding it from the answer
   text under-reports what a real client uploads.
2. **The answer is the answer.** Reasoning text never enters the transcript, never
   starts the TTFT clock, and never lands in a cache built from it.

## `core.store`

One run, one JSON file. No second datastore.

```python
def data_dir() -> Path            # $TRAFFIC_DATA_DIR, default data/runs
def retention_keep() -> int       # $TRAFFIC_RETENTION_KEEP, default 20
def save_run(run) -> dict
def get_run(exec_id) -> dict | None
def list_runs() -> dict           # {"runs": [...], "mock_runs": [...], keep, dir}
def delete_run(exec_id) -> dict
def prune(keep=None) -> dict      # None means retention_keep()
```

Retention exists because the last layout had none: 122 files and 17 MB of mostly
synthetic runs accumulated, indistinguishable from live ones. `save_run` prunes on
every write — retention in a cron job or a cleanup route is retention nobody runs.

A mock run lives in a `mock/` subdirectory, not behind a flag in its filename: a flag
is a rule a future reader has to remember, a directory is one the filesystem enforces.
Each bucket has its own keep budget, so a week of offline development cannot evict the
one live run somebody paid for. `list_runs()` returns two lists rather than one flagged
list, because a caller that has to remember to filter will forget to filter.

The environment is read on every call, never at import: the tests point `TRAFFIC_DATA_DIR`
at a tmpdir, and a module-level constant would have frozen the real one into them.

Column-by-column: `docs/outputs.md`.

## `core.metrics`

`summarize(run)` returns series and totals keyed by `(provider, arm)`:

- per-turn and cumulative `wire_sent`, `wire_recv`, `input_tokens`
- mean/median/min/max for each of the five marks, plus `store_tail_ms`
- prep cost (e.g. Gemini's cache builds, OpenAI's conversation create) reported
  separately and never folded into the totals: it is setup, not traffic, and it
  would drown everything else
- `failures`: every record with an error, named. A run with a broken arm still
  produces numbers, and numbers from a failed call look like numbers from a good one.

No cost estimate. A dollar figure built on a guessed per-token rate is not evidence.

## `core.scenario`

The conversation the arms replay. Every arm answers the same questions against the same
system prompt, or the comparison means nothing: an arm that looks cheap because it was
asked something shorter is not cheap. So the scenario is one file, loaded once per run,
and the runner hands the same `system` and `steps` to every provider.

```python
FIXTURE_DIR = <package>/fixtures
DEFAULT = "perf"

def names() -> list[str]                             # the fixtures on disk
def load(name=DEFAULT, turns=None) -> dict           # {name, description, system, steps}
```

A fixture is JSON: `{name, description, system, steps}`.

- `system` is a **list of paragraphs**, joined with blank lines. It is a list rather than
  one string because the perf fixture's prompt is deliberately large — 20,653 characters,
  over 4k tokens, so that implicit and explicit caching both engage — and a prompt that
  size is unreadable and unreviewable as a single line.
- A **step may be a bare string or `{"text": ...}`**; the perf fixture uses the latter, to
  leave room for per-step annotations. `load()` normalizes both to `list[str]` and refuses
  anything else loudly, at the only door into a run. A provider takes `list[str]`, and a
  dict that slips past becomes `{"text": {"text": "..."}}` on the wire — a malformed
  request the API would accept the shape of and answer from nothing.
- `turns` **truncates**, never cycles. The steps lean on each other through pronouns and
  ellipsis ("roll that back"), so the first n of them is a coherent conversation and a
  resampled n is not. Asking for more turns than the fixture has is an error, not a
  repeat: a repeated question is answered from context and costs nothing like a new one.

## `core.runner`

The harness around the arms. The arms are the experiment; the runner owns everything that
would make two of them incomparable if each did it differently — the connection they open,
the window a capture covers, the clock that says how long the measured stage took, and the
order they run in.

```python
MEASURES = ("bytes", "latency", "both")

def plan(providers=None) -> list[tuple[str, str]]      # the (provider, arm) pairs, in order
def warnings_for(pairs, measure) -> list[str]
def run(providers=None, *, system, steps, measure="bytes", models=None,
        want_capture=False, pause_seconds=0, timestamp="", on_progress=None) -> dict
```

`plan()` maps `{provider: [arm, ...]}` to an ordered pair list; `None` for a provider's
arms means its `HEADLINE_ARMS`, and `None` for the whole map means every provider's. Arms
run **grouped by provider**, so one provider's rate limit cannot be tripped by the other's
burst. An unknown arm raises rather than being skipped.

`run()` returns `{params, records, pcaps, wall_ms}`. Records carry `provider` and `arm`, so
one run holds both vendors and a reader can group either way.

Three rules the arms depend on and cannot enforce themselves:

**A fresh connection per arm.** The session pools TLS connections, so without a reset the
second arm rides the first one's socket: its pcap opens onto an established connection with
no handshake in it, and the first arm's teardown lands inside the second arm's capture.
`core.wire.reset_session()` is called before the first arm, again when each arm's window
opens, and again before its capture is stopped.

**Prep runs outside the window.** A Gemini cache build re-uploads the whole system prompt,
so a run of n turns costs O(n²) in setup alone. Counting that as the arm's traffic would
drown every number the arm exists to produce. Prep records are phased, `core.metrics` keeps
them out of the totals, and the capture window opens only after prep has finished and its
connection has been closed.

**`wall_ms` covers the measured stage only.** It is the same window the pcap covers, so the
two can be read against each other.

The window is bounded by the arm's own progress events: it **opens on the first `steady`
event** and **closes on a `teardown` event**, or when the arm returns (whichever is first;
closing is idempotent). Everything before the first steady call is prep; everything after a
teardown event is cleanup (a cache DELETE). Neither is traffic the arm's turns produced,
and a pcap holding either cannot be read as evidence of what a turn cost. An arm with no
prep and no teardown announces `steady` first and never announces teardown, so it is
captured whole — which is right: all of it is traffic.

`warnings_for()` is what the operator is told **before** the calls go out, not after they
are billed. It refuses `measure="both"` on `openai:responses_stateful` (see "Why two
passes"). A capture that cannot start does not stop a run — the byte counts come from the
socket and stand on their own — but the reason is appended to the same list, or a run with
no pcaps looks like a run that was never asked for one.

`pause_seconds` inserts a gap between arms (never after the last one: it would delay
nothing but the operator), and ticks once a second while doing so — a "pausing" event
followed by a minute of silence is indistinguishable from a hang.

## `core.export`

```python
RECORD_COLUMNS  = [...]
SUMMARY_COLUMNS = [...]

def records_csv(run) -> str      # one row per record, prep rows included
def summary_csv(run) -> str      # one row per (provider, arm), steady turns only
```

Four columns ride at the front of every record row, before any number, because they are
what stops two incomparable numbers being averaged: `provider` and `arm` (a run holds both
vendors, so `stateless` alone names nothing), `phase` (a cache build is not a turn), and
`measure` (bytes off a streamed pass are padded and framed; bytes off a blocking pass are
not).

The raw request and response bodies are not in the CSV. They are the evidence and they are
in the run's JSON; a 40 KB history echo in a spreadsheet cell makes the file unopenable and
the numbers unreadable. Full column tables: `docs/outputs.md`.

## `core.app`

Flask, and a thin skin over `core.runner`. The rules that make a run mean something live in
the runner, not in a route handler, because a rule enforced only by the UI is a rule the CLI
does not have.

| route | what it does |
|---|---|
| `GET /` | the UI |
| `GET /api/config` | the preflight: every provider's arms and whether it is ready, the measures, whether capture is available and why not, the fixtures, the retention limit, and whether this process is in mock mode |
| `POST /api/preflight` | what **this exact selection** would cost: the pairs, the turns, the billable call count (0 in mock mode), and every warning — before anything goes out |
| `POST /api/run` | run and return the whole document. Fine for a short run |
| `POST /api/run/stream` | the same run as server-sent events, ending with the run document. A ten-turn comparison across nine arms takes minutes, and a UI with no progress is a UI the operator reloads mid-run — which abandons the request but not the calls, and bills for a run nobody will ever see |
| `GET /api/runs` | history: live and mock, kept apart |
| `GET /api/runs/<exec_id>` | one run document |
| `DELETE /api/runs/<exec_id>` | drop one |
| `GET /api/runs/<exec_id>/records.csv` | `core.export.records_csv`. A mock run's CSV says so in its filename |
| `GET /api/runs/<exec_id>/summary.csv` | `core.export.summary_csv`, likewise |
| `GET /api/pcaps/<name>` | the pcap, name-validated against traversal |

`main()` binds `TRAFFIC_HOST` (default `127.0.0.1`) and `TRAFFIC_PORT` (default `8080`).

Two things this layer owns and the runner does not: telling the operator what is about to be
billed, and not lying about mock data. A mock run is stored in its own bucket, listed in its
own list, and labelled everywhere it appears. The last iteration of this lab accumulated 122
synthetic runs indistinguishable from live ones; that is the failure this layer is built to
make impossible.

The run is summarized **before** it is saved, not per page view: recomputing a stored run's
summary later means its numbers change when the metrics code changes, which is how a chart
quietly comes to disagree with the CSV beside it.

## `cli.py`

The lab from a terminal, on a machine with the network but no browser — and, mostly, so that
the confirmation step exists.

```
python cli.py                            # what would run, and how many calls that is
python cli.py --go --measure bytes       # run it
python cli.py --go --providers gemini --arms stateless,cached --turns 3
python cli.py --serve                    # the web UI instead
```

**`--dry-run` is the default and there is no way to spend money by accident.** Without
`--go` it prints the fixture, the pairs, the call count (`pairs × turns × 2 if both`), every
`warnings_for()` warning, every not-ready provider, whether capture is available — and stops.
Two paid APIs sit behind a monthly cap, and a full comparison is between fifty and a hundred
calls.

Flags: `--providers`, `--arms` (of one provider only — arm names are not unique across them),
`--measure`, `--fixture`, `--turns`, `--capture`, `--pause SEC`, `--go`, `--serve`. It exits
non-zero if the run produced any failure.

## `core.config`

```python
def flag(name: str) -> bool          # "1" | "true" | "yes" | "on"
def is_mock(provider: str = "") -> bool
```

The mock switch is parsed here and nowhere else. There were once two parsers:
`TRAFFIC_MOCK=true` satisfied one provider and not the other, so half a run was
synthetic and half of it was billed — and the run was then filed in the *live* bucket,
because the flag that picks the bucket had a third reading of its own. Mock data
indistinguishable from measured data is the failure this package exists to prevent, and
a disagreement between truthy-parsers produces exactly that.

A run is filed as mock if **any** provider in it was mocked. It is not a live run with
a caveat: the mocked provider's numbers were never measured, and nothing must ever chart
them next to numbers that were.

## House rules

- **No live API call without the operator asking for one.** The suite runs entirely
  in mock mode; both provider keys have monthly spend caps.
- Mock mode is shaped like the real thing: a stored interaction pays its write tail,
  a resent history pays its upload, and a signature costs bytes but no tokens.
