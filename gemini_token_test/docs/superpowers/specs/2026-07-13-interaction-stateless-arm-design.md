# `interaction_stateless` — the Interactions API with server-side state turned off

Date: 2026-07-13
Status: approved, not yet implemented

## The gap this fills

The experiment currently runs four arms. Two axes decide what an arm is: which
endpoint it calls, and who keeps the conversation history.

| | history kept by the server | history kept by the client |
|---|---|---|
| `POST /v1beta/models/{model}:generateContent` | `cached` (explicit `cachedContents`) | `stateless` |
| `POST /v1beta/interactions` | `interaction`, `interaction_inline` (`previous_interaction_id`) | **missing** |

The empty cell is the control the experiment needs. Today, when `interaction`
beats or loses to `stateless`, two things changed at once: the endpoint *and* the
history mechanism. `interaction_stateless` holds the endpoint fixed and takes only
the server-side history away, so the difference between it and `interaction` is
exactly what `previous_interaction_id` buys — and the difference between it and
`stateless` is exactly the protocol overhead of `/interactions` over
`generateContent`.

## What the arm does

Endpoint: `POST https://generativelanguage.googleapis.com/v1beta/interactions`.

Every turn:

- `store: false` — the interaction is not persisted server-side.
- **No `previous_interaction_id`, ever.** There is no chain to point at.
- `system_instruction` carries the ~12K-char system prompt, re-sent on every turn.
  This matches the `interaction` arm, so the two differ in one respect only.
- `input` carries the **whole conversation so far** as a `Step[]`: the accumulated
  user questions and the model's own prior answers, ending with the new question.

Turn *k* body:

```json
{
  "model": "gemini-3.1-flash-lite",
  "stream": false,
  "store": false,
  "system_instruction": "<12K-char system prompt>",
  "input": [
    {"type": "user_input",   "content": [{"type": "text", "text": "q1"}]},
    {"type": "model_output", "content": [{"type": "text", "text": "a1"}]},
    {"type": "user_input",   "content": [{"type": "text", "text": "q2"}]},
    {"type": "model_output", "content": [{"type": "text", "text": "a2"}]},
    "…",
    {"type": "user_input",   "content": [{"type": "text", "text": "qk"}]}
  ]
}
```

`input` has `2k - 1` steps on turn *k*.

The history accumulates from **this arm's own answers**, not from a transcript
borrowed off another arm. `_arm_stateless` already works this way; an arm whose
history is a conversation it never had measures nothing.

Expected shape of the result: input tokens grow O(N²) across the run, wire bytes
per turn track `stateless` closely, and `interaction_id` comes back but is never
used (and may be absent — `store: false`).

## Grounding in the published API

Two pages, both fetched 2026-07-13:

- `https://ai.google.dev/api/interactions-api` — `input` is typed
  `Content | array(Content) | array(Step) | array(Turn) | string`. The `Step[]`
  form is what lets a client hand the server a full history, prior `model_output`
  steps included. `store` is `boolean`, "Input only. Whether to store the response
  and request for later retrieval."
- `https://ai.google.dev/gemini-api/docs/interactions-overview` — with
  `store: false` the interaction is not retained, which "prevents using
  `previous_interaction_id` for subsequent turns" and rules out `background: true`.
  For a stateless conversation you "manually send the complete conversation history
  in each request", built from `user_input` steps for user turns and `model_output`
  steps for model turns.

The two pages disagree on whether an unstored interaction stays chainable. The
disagreement does not touch this arm: it never chains.

## Open risk, and the probe that closes it

Nothing in this project has ever *sent* a `model_output` step. The 2026-07-10 probe
only ever read them back out of a stored interaction with `GET /interactions/{id}`.
The echo shape is therefore doc-stated and unverified, and it is the one thing that
can invalidate the whole arm.

**Phase 0 is a probe, and it gates everything after it.** Send a three-step `input`
(`user_input`, `model_output`, `user_input`) with `store: false` against
`gemini-3.1-flash-lite`, and record the status code and the raw body — the error
body especially, if it is a 4xx. Write the finding into
`docs/interactions-api-fields.md` under a Measured heading, in the style of the
existing Measured sections there.

If the API rejects `model_output` inside `input`, stop and re-decide between the
`Turn[]` and `Content[]` forms. Do not paper over a rejection with a silent retry
on a different shape: an arm that quietly changed its wire format is an arm whose
numbers mean nothing.

## Changes, by file

**`probe.py`** — the Phase 0 echo-steps probe described above.

**`docs/interactions-api-fields.md`** — a Measured section recording what the probe
found, plus a row for the client-side-history use of `input`.

**`interaction_client.py`** — the arm's mechanics:

- `_step_user(text)` / `_step_model(text)` builders for the two step types.
- `interaction_body(model, text, system_instruction, prev_id, store=True, history=None)`
  — when `history` is given it becomes `input` verbatim and `prev_id` is ignored;
  otherwise the body is byte-for-byte what it is today.
- `run_interaction(..., client_history=False)` — when true: `store: false`, no
  `previous_interaction_id`, and the step list grows by the question and the answer
  after each turn.

The existing `interaction` and `interaction_inline` arms must come out of this
unchanged, byte for byte. Their numbers are already published; a shifted baseline
would silently rewrite them.

**`experiment.py`** — `interaction_stateless` joins `DEFAULT_ARMS` and
`COMPARE_ARMS`, and gets a dispatch branch in `run_arm` that calls
`_arm_interaction` with `client_history=True`.

**`static/app.js`** — label ("Interaction (client-side history, store:false)"),
series color, dash pattern, and point marker, alongside the existing four.

**`templates/index.html`** — a sentence in the explainer saying what the arm
isolates, in the register of the `interaction_inline` paragraph already there.

**`tests/`** — a `test_interaction_stateless.py` mirroring
`test_interaction_inline.py`:

- body shape: `store` is `false`, `previous_interaction_id` is absent, `input` has
  `2k - 1` steps on turn *k*, and the step types alternate correctly;
- the arm is present in both `DEFAULT_ARMS` and `COMPARE_ARMS`;
- a mock comparison run emits records for the arm;
- the existing arms' bodies are unchanged (a regression guard on the refactor);
- pcap naming survives the new arm name (`test_compare_capture.py` already burned
  once on `_` in arm labels).

## Verification

1. Mock comparison run — arm appears, records are well formed, no exception.
2. Live three-turn run against `gemini-3.1-flash-lite`.
3. Assert the arm's `wire_sent` grows turn over turn, and that its per-turn
   `input_tokens` track the `stateless` arm within noise. If they do not, the
   history is not reaching the model and the arm is measuring the wrong thing.
