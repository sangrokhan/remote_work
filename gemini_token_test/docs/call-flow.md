# Implementation & call flow

How the app drives Gemini four different ways, and what actually goes on the wire
each turn. `N` = number of turns; `q_k` = the k-th question; `S` = the large system
prompt (~9 KB, ~1.7k–2.3k tokens).

The point of the demo: **the same N questions, four transports, wildly different
bytes sent and answer quality.**

---

## 1. Components

```mermaid
flowchart LR
  UI["Browser<br/>static/app.js"]
  APP["Flask<br/>app.py"]
  EXP["experiment.py<br/>run_three_stage()"]
  GC["gemini_client.py<br/>call_gemini / create_cache"]
  IC["interaction_client.py<br/>run_interaction()"]
  CAP["capture.py<br/>tcpdump -s 100"]
  ST["store.py<br/>JSON + Firestore"]

  V["Vertex generateContent<br/>{loc}-aiplatform.googleapis.com"]
  C["Vertex cachedContents"]
  I["Interactions API<br/>aiplatform.googleapis.com<br/>locations/global"]

  UI -- "POST /run/stream (SSE)" --> APP
  UI -- "POST /interaction/test (SSE)" --> APP
  APP --> EXP
  APP --> IC
  EXP --> GC
  EXP --> CAP
  GC --> V
  GC --> C
  IC --> I
  APP --> ST
  APP -- "progress / done events" --> UI
```

All Google calls use the same **ADC bearer token** (service account / `gcloud`).
No Gemini API key anywhere.

---

## 2. The 3-stage run (`/run/stream`)

`run_three_stage()` executes four stages in order, each with its own pcap, and
pauses between them so a burst of turns doesn't trip Vertex per-minute quotas.

```mermaid
flowchart TD
  A["Stage 1: stateless<br/>resend S + full history each turn"] --> P1["pause<br/>STAGE_PAUSE_SECONDS (60s)"]
  P1 --> B["Stage 2: cachebuild<br/>create N cachedContents"]
  B --> P2["pause"]
  P2 --> C["Stage 3: stateful<br/>cache_(k-1) + q_k"]
  C --> P3["pause"]
  P3 --> D["Stage 4: nocontext<br/>S+q_1, then bare q_k"]
  D --> E["delete caches · summarize · save_run"]
```

Each stage: `reset_session()` → start tcpdump → warmup 2s → turns → close socket →
drain 1s → stop tcpdump. The fresh socket makes every pcap start on a clean TCP
handshake.

---

### 2a. Stateless — full resend

Client keeps the transcript and replays all of it every turn. Bytes grow
quadratically; answers are coherent.

```mermaid
sequenceDiagram
  participant C as client
  participant V as generateContent
  Note over C: history = [S]
  C->>V: POST contents = [S, q₁]
  V-->>C: a₁
  Note over C: history += q₁, a₁
  C->>V: POST contents = [S, q₁, a₁, q₂]
  V-->>C: a₂
  C->>V: POST contents = [S, q₁, a₁, q₂, a₂, q₃]
  V-->>C: a₃
  Note over C,V: turn k sends S + all prior Q&A + q_k
```

### 2b. No-context — pure question

System prompt rides the **first** query only; every later turn sends the bare
question. The server is stateless, so turns 2..N have no context at all —
pronoun/ellipsis questions ("bring it back", "that same check") go ambiguous.
This is the control case.

```mermaid
sequenceDiagram
  participant C as client
  participant V as generateContent
  C->>V: POST contents = [S, q₁]
  V-->>C: a₁
  C->>V: POST contents = [q₂]
  V-->>C: a₂ (ambiguous — no history)
  C->>V: POST contents = [q₃]
  V-->>C: a₃ (ambiguous)
```

### 2c. Cache-based stateful

Two phases. First build one cache per prefix, then answer each turn with
`cachedContent` + the new question only. The prefix lives server-side; cached
input tokens bill at ~10%.

```mermaid
sequenceDiagram
  participant C as client
  participant K as cachedContents
  participant V as generateContent

  rect rgb(240,240,240)
  Note over C,K: Stage 2 — cachebuild
  C->>K: POST create(contents = S + [q₁,a₁])
  K-->>C: cache₁ (+cached_tokens)
  C->>K: POST create(contents = S + [q₁,a₁,q₂,a₂])
  K-->>C: cache₂
  end

  rect rgb(240,240,240)
  Note over C,V: Stage 3 — stateful replay
  C->>V: POST contents = [q₁] (no cache yet → sends S)
  V-->>C: a₁
  C->>V: POST cachedContent = cache₁, contents = [q₂]
  V-->>C: a₂
  C->>V: POST cachedContent = cache₂, contents = [q₃]
  V-->>C: a₃
  end
  Note over C: caches deleted unless KEEP_CACHE=1
```

Turn `k` uses `cache_(k-1)`; turn 1 has no cache and falls back to sending the
prefix. A prefix below `MIN_CACHE_TOKENS` (2048) is skipped by the API, and that
turn falls back too.

---

## 3. Interaction API (`/interaction/test`)

Genuinely stateful: the **server** stores the conversation. The client sends only
the new question plus `previous_interaction_id`.

Two targets, chosen by `INTERACTION_AGENT`:

| | `model` mode (default) | `agent` mode (`INTERACTION_AGENT` set) |
|---|---|---|
| body field | `model: gemini-2.5-flash` | `agent: antigravity-preview-05-2026` |
| sandbox | none | remote container, provisioned on demand |
| `background` | `false` (foreground stream) | **`true` — required**; the service rejects `false` |
| warmup | not needed | yes, before turn 1 |
| speed | fast | slow (container + agent work) |

`agent` is only required when `model` is absent. Model mode skips the sandbox
entirely, which is why it's much faster.

The warmup below applies to **agent mode only**: the container is provisioned on
demand, so we warm it up *before* asking anything, otherwise turn 1 pays for (or
fails on) provisioning.

```mermaid
sequenceDiagram
  participant C as client
  participant I as Interactions API

  rect rgb(240,240,240)
  Note over C,I: init stage — warmup_environment() [agent mode only]
  loop until env_id, or WARMUP_TIMEOUT
    C->>I: POST environment={type:remote}, input="ready"
    alt sandbox still provisioning
      I-->>C: 400 "resource setup has just started"
      Note over C: sleep WARMUP_INTERVAL, retry
    else ready
      I-->>C: SSE … interaction.complete {environment_id}
    end
  end
  Note over C: env_id captured (warmup turn NOT chained)
  end

  rect rgb(240,240,240)
  Note over C,I: scenario turns (model mode: no environment field)
  C->>I: POST model=…, store=true, input=[S, q₁]
  I-->>C: SSE stream → a₁, interaction.id = id₁
  C->>I: POST model=…, previous_interaction_id=id₁, input=[q₂]
  I-->>C: SSE → a₂ (server recalled the history)
  C->>I: POST model=…, previous_interaction_id=id₂, input=[q₃]
  I-->>C: SSE → a₃
  end
```

Set `INTERACTION_ENV_ID` to skip warmup entirely and reuse a sandbox (TTL 7 days,
reset on each interaction).

Each response is an **SSE event stream**; events are parsed as they arrive and
timestamped, so `first_event_ms` (time to the server's first event) separates
provisioning/queueing from the agent actually working.

---

## 4. What goes on the wire, per turn

| Case | Sent on turn k | Context the model has | Sent bytes |
|---|---|---|---|
| stateless | `S` + all prior Q&A + `q_k` | full | grows ~O(k) per turn, O(N²) total |
| no-context | `q_k` (turn 1: `S + q₁`) | none after turn 1 | flat, tiny |
| stateful (cache) | `cachedContent` ref + `q_k` | full (server-side prefix) | flat, tiny |
| interaction | `previous_interaction_id` + `q_k` | full (server-side history) | flat, tiny |
| interaction_stateless | `S` + all prior Q&A + `q_k` (`store: false`, no `previous_interaction_id`) | full (client-resent, server keeps nothing) | grows ~O(k) per turn, tracks stateless |

Coherent answers: stateless, stateful, interaction, interaction_stateless.
Ambiguous: no-context. Small requests: no-context, stateful, interaction. So
**stateful and interaction are the two ways to get "small request + coherent
answer"** — one caches the prefix, the other keeps the whole conversation
server-side. `interaction_stateless` fills the remaining cell of the endpoint
× who-keeps-the-history matrix: the interactions endpoint, but the client
keeps the history, like stateless. A live 3-turn run (2026-07-13,
gemini-3.1-flash-lite) shows its `input_tokens` tracking `stateless` turn for
turn (4459/4825/5337 vs 4459/4885/5413) rather than `interaction`'s
(4459/4886/5465), which grows at the same rate — proof the server actually
reads the client-supplied history rather than accepting and ignoring it.
Against `interaction`, the wire gap (21701/23342/25600 vs 21700/21755/21722)
is exactly what `previous_interaction_id` buys: ~3.9 KB by turn 3, widening
every turn — a **bytes** saving, not a token one. `interaction`'s
input_tokens (4459/4886/5465) grow at the same rate as every other arm's,
because the server replays and bills the stored history; the ~2% residual
against `interaction_stateless` (5337 vs 5465 at turn 3) is answer-length
variance, not a saving the mechanism produces.

---

## 5. Progress & evidence

- **SSE progress** — both endpoints emit `{stage, turn, turns}` per turn (plus
  `{stage:"pause", turn:secs_left}` and `{stage:"provisioning", attempt}`), with a
  `: keepalive` heartbeat so long runs don't idle out through a proxy.
- **Per-stage pcap** — `tcpdump -i any -s 100 -U` filtered to tcp/443. Snaplen 100
  keeps headers and the true packet length while cutting the disk I/O that causes
  kernel drops; the drop counters are parsed from tcpdump's stderr and shown in the
  UI.
- **Exports** — chat JSON (question, answer, raw request/response per turn) and a
  comparison CSV (`turn, query, stateless_response, nocontext_response,
  stateful_response`).
