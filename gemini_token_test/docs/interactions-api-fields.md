# Interactions API — request fields, and what belongs in each

Endpoint used by this project (fixed by the probe on 2026-07-10):

```
POST https://generativelanguage.googleapis.com/v1beta/interactions
Header: x-goog-api-key: <API key>
Header: Content-Type: application/json
```

Two sources are kept apart below:

- **Docs** — the published field reference
  (<https://ai.google.dev/api/interactions-api>,
  <https://ai.google.dev/gemini-api/docs/interactions-overview>).
- **Measured** — what this project's probe actually observed against
  `gemini-3.1-flash-lite`, including a `GET /interactions/{id}` on a stored
  interaction. Where the two differ, Measured wins.

---

## Required fields

Exactly one of `model` / `agent`, plus `input`. Nothing else is required.

| Field | Type | What goes in it |
|-------|------|-----------------|
| `model` | string | The Gemini model id, e.g. `"gemini-3.1-flash-lite"`. Required **if `agent` is absent**. This project always uses `model`. |
| `agent` | string | A managed agent id (`deep-research-preview-04-2026`, `antigravity-preview-05-2026`, …). Required **if `model` is absent**. Not used by this project. |
| `input` | Content \| array \| Step[] \| Turn[] \| string | The new user turn. Only the *new* message — never the prior history (see `previous_interaction_id`). |

`model` and `agent` are mutually exclusive. Sending an agent id in `model` is
rejected: `"'X' refers to an agent, but was provided in the 'model' field."`

### Shape of `input`

The canonical form this project sends:

```json
"input": [
  { "type": "user_input",
    "content": [ { "type": "text", "text": "What is the capital of France?" } ] }
]
```

A bare string is also accepted (`"input": "..."`). Content blocks other than
`text` exist (`image`, `audio`, `document`) but this project sends text only.

---

## Optional fields — and the critical distinction

Everything below is optional. The distinction that matters for a stateful
conversation:

> `previous_interaction_id` carries **only the conversation history** — the user
> inputs and model outputs of the chained interactions. Every other field is
> **interaction-scoped**: it applies to the single interaction you are generating
> right now and is neither stored on it nor carried to the next turn.

So a multi-turn conversation re-sends the instruction fields on **every** turn.

| Field | Type | What goes in it | Persists across `previous_interaction_id`? |
|-------|------|-----------------|--------------------------------------------|
| `previous_interaction_id` | string | The `id` of the previous interaction. Tells the server to prepend that interaction's history. | — (this *is* the history mechanism) |
| `system_instruction` | string | The system prompt **and** persona — both are just prose in this one string. "You are a terse assistant. Never apologise." | **No.** Re-send every turn. |
| `tools` | Tool[] | Tool/function **declarations** (name, description, JSON-schema parameters) the model may call. This is where "tool description" lives — *not* in `system_instruction`. | **No.** Re-send every turn. |
| `generation_config` | object | Decoding settings: `temperature`, `top_p`, `seed`, `stop_sequences`, `max_output_tokens`, `thinking_level` (`minimal`/`low`/`medium`/`high`), `thinking_summaries`. Only valid when `model` is set. | **No.** Re-send every turn. |
| `cached_content` | string | Name of an explicit `cachedContents` resource to use as context. Format `projects/{p}/locations/{l}/cachedContents/{id}`. Docs and overview page disagree on whether Interactions accepts it — unverified here. | n/a |
| `stream` | bool | `true` → SSE. `false` → one JSON body. **Both work** (measured). This project uses `false` to match `generateContent`. | per-call |
| `store` | bool | `true` → the interaction is retrievable later by `GET /interactions/{id}` and chainable via `previous_interaction_id`. | per-call |
| `background` | bool | Run the interaction in the background (long tasks). Not used here. | per-call |
| `response_format` | ResponseFormat | Force a JSON response matching a supplied JSON schema. Not used here. | per-call |
| `response_modalities` | enum[] | Which output types to return (`text`/`image`/`audio`/`video`/`document`). | per-call |

### `system_instruction` is *not* required

A call with no `system_instruction` returns 200. It is optional. The point is only
that if you *want* a system prompt in force on turn 2, you must send it again on
turn 2 — the server did not keep it.

---

## What "persona / system prompt / tool description" map to

The three things you might think of as "the model's setup" are **two** fields, not
one, and none of them are the history:

- **system prompt** → `system_instruction` (string)
- **persona** → also `system_instruction` — it is a sentence inside that same string, not a separate field
- **tool description** → `tools[]` (a separate array; each tool has its own `description`)

All are interaction-scoped. All are re-sent every turn.

---

## Measured proof that instruction fields do not persist

Probe, 2026-07-10, `gemini-3.1-flash-lite`, two independent runs.

**They are not stored on the interaction.** Turn 1 sent `system_instruction` and
`tools` with `store: true`. `GET /interactions/{id}` returned:

```
keys: [created, id, model, object, service_tier, status, steps, updated, usage]
system_instruction: None
tools: null
```

The stored resource holds `steps` (the user/model turns) and nothing else. The
instruction fields are absent from it entirely.

**They do not apply on the next turn.** A conditional rule proves it without the
history leaking the answer:

- `system_instruction`: "if the user's message is exactly `BANANA`, reply with only `ZQ7`."
- Turn 1 — instruction present, message `"Say hello."` → `"Hello! How can I help you today?"` (rule not triggered; history learns nothing about it).
- Turn 2 — `previous_interaction_id`, **no** `system_instruction`, message `BANANA` → `"BANANA! 🍌 …"`.

The rule is gone. (An earlier probe using an *unconditional* marker wrongly read
`persisted`, because turn 1's own output carried the marker into the history and the
model imitated it. The conditional design removes that confound.)

---

## Measured: `input` accepts a client-supplied history (Step[] echo)

Probe, 2026-07-13, `gemini-3.1-flash-lite`, `probe.probe_step_echo()`.

`POST /v1beta/interactions` with `store: false`, no `previous_interaction_id`, and
an `input` of three steps — `user_input` ("What is the capital of France?"),
`model_output` ("Paris."), `user_input` ("And of Italy? Answer in one word.").

- HTTP status: `200`
- Verdict: `supported`
- Answer: `Rome.`

The model answered `Rome.` — it could only resolve "of Italy?" by reading the
`model_output` step the client supplied, so the interaction_stateless arm's core
mechanism (resending the whole conversation, including the model's own prior
answers, as `Step[]`) is confirmed to work.

This is the shape the `interaction_stateless` arm sends on every turn: the whole
conversation, the model's own prior answers included, with the server storing
nothing.

---

## Consequence for the traffic experiment

The `interaction` arm must re-upload `system_instruction` (the ~12K-char system
prompt) and `tools` on all 10 turns. `previous_interaction_id` saves only the
re-transmission of the **history** (past questions and answers). The instruction
payload is uploaded identically to `stateless`.

This is why an explicit cache can beat `previous_interaction_id` on bytes: a cache
can hold the system prompt; `previous_interaction_id` cannot.
