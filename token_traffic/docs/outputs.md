# What a run produces

A run leaves behind one JSON document, CSVs derived from it on demand, and — if capture
was asked for and was available — one pcap per (arm, kind). Nothing else. There is no
second datastore: a run is a few hundred kilobytes and this experiment runs on one
machine.

The CSVs are `records.csv` and `summary.csv` always, plus `cwnd.csv` and
`cwnd_summary.csv` when congestion monitoring was on. All of them, the run document, and
every pcap are in `GET /api/runs/<exec_id>/bundle.zip`.

## The run document

One file per run, written by `core.store.save_run()`:

```
$TRAFFIC_DATA_DIR/            exec_20260714T101530Z_a1b2c3d4.json     (live)
$TRAFFIC_DATA_DIR/mock/       exec_20260714T101602Z_9f8e7d6c.json     (mock)
```

`TRAFFIC_DATA_DIR` defaults to `data/runs`. The `exec_id` is a UTC timestamp plus four
random bytes.

| key | what it holds |
|---|---|
| `exec_id` | the run's id, and the filename |
| `schema_version` | currently `1`. A run written under an older layout has to be identifiable as such, not silently charted next to a current one whose columns mean something else |
| `timestamp` | when the run started, UTC ISO-8601 |
| `mock` | `true` if the run made no network call. Every consumer keys off this |
| `params` | `mode`, `measure`, `pairs` (`["gemini:cached", "openai:responses_inline", …]`), `providers`, `models`, `turns`, `fixture`, `capture`, `cache_bust`, and `warnings` — everything the operator was told before the calls went out, kept with the numbers they produced |
| `params.cache_bust.prefix_drift` | `true` when a turn counter (`[turn 001]`) was put in **front** of the system prompt on every turn. The negative control: a prefix cache matches from the first token, so a marker that moves every turn misses every turn and the server re-prefills the whole prompt — and the history behind it — from scratch. Expect `cached_tokens` near zero and TTFT that grows with the prompt. It is the runnable form of "the system prompt must not change during a multi-turn or agentic task"; anything genuinely per-turn belongs *after* the stable prefix. Arms that send the prompt once (`gemini:cached`, `gemini:interaction_inline`, `openai:responses_inline`) have no per-turn send for it to ride, and the run names them in `warnings` |
| `params.cache_bust` | `{enabled, tags, prefix_drift}`. `tags` is `{"gemini:stateless": "3f9a1c2e7b40d5a6", …}`: the marker each arm's system prompt actually carried. Recorded rather than merely flagged, because a run that came back suspiciously warm can only be explained if the prefixes it sent are recoverable. `enabled: false` means the arms shared a prefix and an arm's `cached_tokens` and TTFT may belong to the arm before it — the run says so in `warnings` too |
| `records` | one row per (provider, arm, turn, pass), including prep. See below |
| `summary` | `core.metrics.summarize()`, computed once, at save time, and stored |
| `pcaps` | keyed `provider:arm`; what each capture actually got |
| `cwnd` | keyed `provider:arm`, then kind; the congestion samples and what they add up to. Absent unless monitoring was on |
| `wall_ms` | keyed `provider:arm`; how long the arm's steady stage took, start to finish — the same window the pcap covers, so the two can be read against each other |

The summary is computed **before** saving rather than per page view, because recomputing
it later means an old run's numbers change when the metrics code changes — which is how a
chart quietly comes to disagree with the CSV beside it.

Each record carries `request_raw` and `response_raw`: the bodies exactly as they went out
and came back. They are the evidence. They are not in the CSVs — a 40 KB history echo in
a spreadsheet cell makes the file unopenable and the numbers unreadable.

### Retention

`save_run()` prunes its own bucket back to `TRAFFIC_RETENTION_KEEP` (default 20) on every
write. Pruning at write time rather than in a cron job or a cleanup route means retention
holds even if nobody remembers it exists — which is exactly what failed last time: the
previous layout accumulated 122 files and 17 MB, most of them synthetic runs sitting in
the same directory as the live ones, under the same naming scheme, charting identically.

Each bucket gets its own budget. A week of offline development cannot evict the one live
run somebody paid for, and live runs do not evict mock ones either, so a fixture run stays
reproducible.

### The rule about mock runs

A mock run lives in its own subdirectory rather than behind a flag in a filename, because
a flag in a filename is a rule a future reader has to remember and a directory is one the
filesystem enforces. `list_runs()` returns `runs` and `mock_runs` as two lists rather than
one flagged list, because a caller that has to remember to filter is a caller that will
forget to filter. A downloaded CSV from a mock run has `mock_` in front of its filename: a
number lifted out of a spreadsheet has no other way of remembering it was never measured.

**A mock run is never charted or averaged with a live one.** Its bytes are shaped like
real bytes and its timings are shaped like real timings — that is the point of it — which
is exactly why the two must never end up on one axis.

## `records.csv`

`GET /api/runs/<exec_id>/records.csv`, or `core.export.records_csv(run)`. One row per
record — prep rows included, phased, not dropped. A reader who wants only the steady turns
can filter; a reader who is never shown the cache build cannot discover what the arm paid
before its first question.

The first five columns exist so that two rows which are not comparable never look
comparable. They ride at the front, before any number.

| column | meaning |
|---|---|
| `provider` | `gemini` or `openai`. A run holds both vendors, and `stateless` alone names nothing |
| `arm` | which strategy. Unique only together with `provider` |
| `phase` | `steady` for the turns that count; `cachegen` (a Gemini cache build or transcript replay) or `setup` (an OpenAI conversation create) for prep. A cache build is not a turn, and `core.metrics` never folds one into a total |
| `kind` | on a prep record only: `transcript`, `cache_create`, or `conversation_create`. A phase is **not** a kind — `cachegen` holds transcript calls *and* cache builds, and they do not measure the same thing. Summing their token columns produced 19071 for `gemini:cached`: two real input counts (4479 + 4762) added to two cache sizes (4659 + 5171). A number describing nothing |
| `billed` | on a prep record only. `true` means the token columns are tokens billed for an answer; `false` means nothing was billed and a `0` in those columns means **not billed**, not *not sent*. `POST /v1/conversations` returns no usage object at all, and the 21 KB prompt it stores is billed as input on **every turn** — the zeros are honest and the note beside them is what says what they mean |
| `cache_tokens` | on a `cache_create` record: the **size** of the prefix now held in the cache. A size, not a bill, which is why it is not in `input_tokens` |
| `turn` | 1-based within the arm; `0` on a prep record that belongs to no turn |
| `measure` | `bytes`, `latency`, or `both`. Bytes off a streamed pass are framed (and, on OpenAI, obfuscation-padded); bytes off a blocking pass are not. They are different measurements wearing the same column name, and averaging them is the mistake this column exists to prevent |
| `wire_sent`, `wire_recv` | socket bytes, headers and content-encoding included. `wire_sent` is the axis the arms differ on |
| `req_payload_bytes`, `resp_payload_bytes` | the decoded body sizes, for reference. Not what anyone pays |
| `req_sent_ms` | the request's last byte went out — the client's history has finished uploading |
| `ttfb_ms` | the response's first byte came back: network and queue, no tokens yet |
| `ttft_ms` | the first event carrying **answer** text. A reasoning delta is not the answer and does not start this clock |
| `ttlt_ms` | the last event carrying answer text — what a streaming user waits for |
| `turn_end_ms` | the stream closed — what a blocking client waits for |
| `store_tail_ms` | `turn_end − ttlt`, floored at zero. On a stored Gemini interaction this is the ~1.8 s the server spends persisting the turn after the answer is already out |
| `input_tokens`, `cached_tokens`, `output_tokens`, `reasoning_tokens`, `total_tokens` | from the provider's own usage block, translated into one vocabulary. `reasoning_tokens` is what Gemini bills as thought tokens |
| `error` | empty when the call succeeded |

On a `measure=bytes` row the five marks are `0`. That is not "instant" — it is "not
measured", and `measure` is the column that says so. A row from a *failed* call is
different: its marks are pinned to the moment the turn ended, because a zero mark would
chart as the fastest turn in the run.

## `summary.csv`

`GET /api/runs/<exec_id>/summary.csv`, or `core.export.summary_csv(run)`. One row per
(provider, arm), over the **steady** turns only.

| column | meaning |
|---|---|
| `provider`, `arm`, `measure` | the identity of the series, and what kind of measurement it is |
| `turns` | how many steady turns went into the row |
| `wire_sent`, `wire_recv`, `wire` | totals over the steady turns. `wire` is the sum of the two, kept separate from `wire_sent` because the model's answer dominates it and would bury the difference the arms are actually about |
| `input_tokens`, `cached_tokens`, `output_tokens`, `reasoning_tokens`, `total_tokens` | totals |
| `<mark>_mean`, `<mark>_median` | for each of `req_sent_ms`, `ttfb_ms`, `ttft_ms`, `ttlt_ms`, `turn_end_ms`, `store_tail_ms`. Both, always: a mean alone hides the one turn that took eight seconds; a median alone hides that it happened at all |
| `call_ms` | time spent inside the measured calls |
| `wall_ms` | the steady stage, start to finish — the same window the pcap covers. Neither clock includes prep or teardown |
| `prep_calls`, `prep_wire_sent`, `prep_wire_recv` | what the arm paid before its first measured turn. **Zero, not blank**, for an arm with no prep: blank reads as "not measured", and a stateless arm's zero setup cost is a measurement and the point of the row |
| `errors` | how many records of this arm carried one |

Prep is excluded from the totals but never hidden. A Gemini cache build re-uploads the
whole prefix — build one per turn and the setup alone costs O(N²) — so folding it into the
totals would drown every number the arm exists to produce. But an arm whose steady traffic
is cheap because it paid up front should be seen to have paid up front, which is what the
three `prep_*` columns are for.

The run's `summary` also carries a `failures` list, naming every record with an error. A
run with a broken arm still produces plausible numbers, and a number from a failed call is
shaped exactly like a number from a good one.

## `cwnd.csv` and `cwnd_summary.csv`

`GET /api/runs/<exec_id>/cwnd.csv` and `.../cwnd_summary.csv`, or
`core.export.cwnd_csv(run)` / `cwnd_summary_csv(run)`. Present only when the run was
asked to monitor; an unmonitored run returns a header and no rows, which says "monitored
nothing" rather than "saw nothing".

`cwnd.csv` is the raw series: one row per (arm, tick, socket), a hundred rows a second
per socket. It is meant to be plotted, not read. `snd_cwnd` against `t_ms` is the
picture; `snd_ssthresh` says where slow start hands off to congestion avoidance, and
`rtt_us` says what one re-earned round trip is worth in milliseconds. `ca_state` is the
column that keeps the reading honest — a window that shrank while it says `recovery`
shrank because of loss, which is a different finding with a different fix.

`cwnd_summary.csv` is one row per monitored arm:

| column | what it is |
|---|---|
| `interval_ms` | the sampling period actually requested |
| `samples`, `ticks`, `seconds` | how much was collected, over how long. `ticks` well below `seconds × 1000 / interval_ms` means the box could not keep up |
| `sockets` | every local `ip:port` that matched the API host. More than one is normal: a pooled client may open several |
| `peak_cwnd`, `final_cwnd` | the widest window the arm earned, and where it ended |
| `idle_resets` | how many times a window that had grown past 10 segments went back to 10 or below while `ca_state` was `open`. This is the number the monitoring exists to produce |
| `truncated` | the arm hit `TRAFFIC_CWND_MAX_SAMPLES` and the tail is missing. Reported, never silent |
| `error` | why the arm has no samples, when it has none |

`peak_cwnd` well above 10 with `idle_resets` at zero is a real result too: on that path,
the idle gaps cost nothing.

Samples come from `native/cwnd_monitor`, a C helper reading netlink `sock_diag` —
unprivileged, the same interface `ss -ti` uses, and it never touches the client's
sockets. Sampling at 10 ms resolves an idle gap of one RTO (200 ms and up) into dozens
of points; it cannot resolve an event shorter than a tick, which is why the loopback
tests check the monitor against `ss` instead of asserting a reset that completes in
microseconds there.

A mock run produces no traffic to the API host, so its monitor comes back with zero
samples and no error — the same shape as a live arm whose connection went somewhere
else.

## The pcaps

One per (provider, arm, **kind**), written to `TRAFFIC_PCAP_DIR`, downloadable at
`GET /api/pcaps/<name>`:

```
capture_gemini_interaction_inline_bytes_2026-07-14T10-15-30-837905-00-00_9f8e7d6c5b4a3f21.pcap
```

Every pcap's entry carries an `offload` block, and it is not decoration: without it a
reader cannot tell a 64 KB kernel super-packet from a jumbo frame, or a slow-start burst
that is missing from one that never happened.

| field | what it is |
|---|---|
| `iface` | the device the capture's traffic actually leaves by, resolved from the routing table (`TRAFFIC_PCAP_IFACE` defaults to `any`, which is not a device) |
| `during_capture` | `{tso, gso, gro}` as they were while packets were being recorded. Any `true` here means the packet sizes in the file are not wire frames |
| `before` | what they were before the capture, and what they were restored to |
| `disabled` | which ones this capture turned off. Empty unless `TRAFFIC_PCAP_NO_OFFLOAD` is set |
| `fixed` | features the driver will not let anyone change |
| `error` | why the state is unknown or could not be changed. A `RESTORE FAILED` here means the machine was left altered and needs attention |

The `log` lines lead with the same thing in a sentence, before the packet counts,
because "3412 captured" means nothing until you know whether those were packets.

The label is `provider_arm_kind`, where `kind` is the measure the pcap holds — `bytes` or
`latency`. The timestamp is the run's, with every non-alphanumeric character squeezed to a
dash — the run stamps itself with `datetime.isoformat()`, which carries a `.` and a `+`,
and a filename holding either of them is a filename the download route's own validator
rejects. It used to: tcpdump wrote a good pcap, the run recorded it, and
`GET /api/pcaps/<name>` answered 404 for a file sitting on disk. The authoritative
timestamp is in the run document, next to the pcap's entry; the one in the name is a label.
The trailing 64-bit token means two concurrent captures cannot collide and a download URL
is not guessable from another one. A label that cannot be spelled safely in a filename is
**refused**, not substituted — the predecessor renamed unspellable labels to a default,
which is how an arm once shipped a pcap claiming to be a different arm.

`run["pcaps"]` is `{"provider:arm": {"bytes": {...}, "latency": {...}}}` — a map of kind to
capture result. A single-measure run has one kind; a `measure=both` run has two.

**Why `both` captures twice.** In a `both` run each turn sends a blocking pass (the bytes)
and a streamed pass (the marks), and the two interleave on the same host and port. One
capture spanning the arm would hold both passes, and its packet total would match neither
the recorded `wire_sent` (which drops the streamed frames) nor the latency number — the
pcap could verify nothing, which is its whole job. So the arm runs twice: the whole
conversation in `bytes` under one capture, then the whole conversation in `latency` under
another, and the per-turn records are merged back the way `core.call.send` merges the two
passes of one turn — bytes and body from the bytes sweep, marks from the latency sweep. The
latency sweep re-runs prep too (a second cache build, a second conversation create); those
duplicate setup records are dropped, and only the bytes sweep's prep is kept.

A byte count taken inside the process is a claim; a pcap taken on the interface is the
thing the claim is about. The TLS payload is encrypted and is not the point: packet sizes
and timing are exactly the traffic being argued over, and they can be opened in Wireshark
by somebody who does not trust this code. One capture per (arm, kind), because a single
pcap spanning either more arms or both passes cannot be attributed after the fact.

The capture covers the arm's steady stage only — it opens on the arm's first `steady`
progress event, after the pooled connection has been dropped and the peer's FIN has been
waited out, and closes on `teardown`. So each pcap is one self-contained SYN…FIN
conversation containing the measured turns and nothing else.

`run["pcaps"]["gemini:cached"]` records what the capture actually got: `ok`, `file`,
`bytes`, the `host` and resolved `ips`, the tcpdump `filter`, the `snaplen`, tcpdump's own
`stats` (captured / received by filter / dropped), a `dropped` total and a `log`. A lossy
pcap announces itself rather than being read as a complete record of the run: dropped
packets mean the capture was overloaded, the pcap will show "previous segment not
captured", and the fix is a quieter host or a smaller `TRAFFIC_PCAP_SNAPLEN`.

A mock run produces no packets. Its capture, if one was asked for, comes back with
`ok: false` and the note "no packets captured (a mock run makes no real traffic)".
