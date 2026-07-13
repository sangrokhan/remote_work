# `interaction_stateless` Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fifth arm, `interaction_stateless`, that calls the Interactions API with `store: false`, never uses `previous_interaction_id`, and resends the whole conversation as a client-accumulated `Step[]` in `input`.

**Architecture:** The arm is a third mode of the existing `interaction_client.run_interaction`, not a new client. Two new parameters carry it: `store` (body field) and `client_history` (accumulate `Step[]` instead of chaining). `experiment.py` gains an arm name and a dispatch branch; the UI gains a label, a color, and a dash/marker. Every byte of the existing `interaction` and `interaction_inline` arms stays exactly as it is — their numbers are already published.

**Tech Stack:** Python 3, Flask, `requests`, plain-JS Chart.js front end. Tests are pytest-style module-level functions with `monkeypatch`, matching `tests/test_interaction_inline.py`.

**Spec:** `docs/superpowers/specs/2026-07-13-interaction-stateless-arm-design.md`

## Global Constraints

- Project directory is `gemini_token_test/`. Every path below is relative to it. Never write outside it.
- Endpoint: `POST https://generativelanguage.googleapis.com/v1beta/interactions`, header `x-goog-api-key`. Do not touch the Vertex path.
- The arm sends `store: false`, and **never** sends `previous_interaction_id`.
- `system_instruction` carries the system prompt on **every** turn of this arm.
- `input` on turn *k* is a `Step[]` of length `2k - 1`: `user_input`, `model_output`, … , `user_input`.
- History accumulates from **this arm's own** answers, never from another arm's transcript.
- The `interaction` and `interaction_inline` request bodies must not change. Task 2 has a regression test that pins this.
- Every test run is offline: `GEMINI_MOCK=1`. The only live calls in this plan are Task 1's probe and Task 5's verification run, and both are called out as billable.
- Run tests with `python3 -m pytest tests/ -v` (module-level `test_*` functions with `monkeypatch` are pytest, not unittest).

---

### Task 1: Probe the `Step[]` echo shape — the gate

Nothing in this project has ever *sent* a `model_output` step; the 2026-07-10 probe only read them back out of a stored interaction. The whole arm rests on the API accepting one. Settle it with one live call before writing the arm.

**This task gates the rest of the plan.** If the probe reports `unsupported`, stop, report the raw error body, and re-decide between the `Turn[]` and `Content[]` forms with the user. Do not silently switch shapes.

**Files:**
- Modify: `probe.py` (add a function at the end, before the `# --- cached probe ---` section)
- Modify: `docs/interactions-api-fields.md` (append a Measured section)

**Interfaces:**
- Produces: `probe.probe_step_echo() -> dict` with keys `url`, `status`, `verdict`, `body`, `sent_steps`. Nothing else consumes it; it is a one-shot diagnostic run from the shell.

- [ ] **Step 1: Write the probe function**

Add to `probe.py`, immediately above the `# --- cached probe ---` comment block:

```python
# --------------------------------------------------------------------------
# Step-echo probe: can a client hand the server a history it already has?


def _step_user(text: str) -> dict:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def _step_model(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def probe_step_echo(model: str = "") -> dict:
    """Does `input` accept a client-supplied history, `model_output` steps included?

    The docs type `input` as Content | Content[] | Step[] | Turn[] | string and the
    overview page says a stateless conversation resends the whole history as steps.
    Neither claim has ever been tested here: the 2026-07-10 probe only ever *read*
    model_output steps back out of a stored interaction with GET /interactions/{id}.

    Sends a three-step history with store:false and no previous_interaction_id --
    exactly the shape the interaction_stateless arm depends on. One live call.
    """
    model = model or (PROBE_MODELS[0] if PROBE_MODELS else DEFAULT_MODEL)
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    if not API_KEY:
        return {"url": url, "status": 0, "verdict": "environment",
                "body": "GEMINI_API_KEY not set", "sent_steps": 0}

    steps = [
        _step_user("What is the capital of France?"),
        _step_model("Paris."),
        _step_user("And of Italy? Answer in one word."),
    ]
    body = {"model": model, "stream": False, "store": False, "input": steps,
            "system_instruction": "You are a test fixture. Answer in one word.",
            "generation_config": {"max_output_tokens": 16}}
    try:
        resp = _session().post(url, data=json.dumps(body),
                               headers=_headers("apikey"), timeout=PROBE_TIMEOUT)
        status, text = resp.status_code, resp.text
    except Exception as exc:
        return {"url": url, "status": 0, "verdict": "error", "body": str(exc),
                "sent_steps": len(steps)}

    return {"url": url, "status": status, "verdict": classify(status, text),
            "body": text[:2000], "sent_steps": len(steps)}
```

- [ ] **Step 2: Run the probe — this is a live, billable call**

Run:

```bash
cd gemini_token_test && python3 -c "import json, probe; print(json.dumps(probe.probe_step_echo(), indent=2))"
```

Expected on success: `"status": 200`, `"verdict": "supported"`, and a `body` whose `steps` contain a `model_output` saying `Rome`.

A `Rome` answer is the strong signal: the model could only know Italy was the subject by reading the history the client supplied. If the answer ignores the history, the steps were accepted but not used — record that too; it fails the arm just as surely as a 400 does.

**Gate:** if `verdict` is `unsupported` (a 400 rejecting the field), stop the plan here and report the raw `body`. If it is `environment`, the key or quota is the problem, not the shape — fix that and re-run.

- [ ] **Step 3: Record what the probe found**

Append to `docs/interactions-api-fields.md`, after the existing "Measured proof that instruction fields do not persist" section. Fill the quoted values in from the actual probe output — do not copy the sample below verbatim:

```markdown
## Measured: `input` accepts a client-supplied history (Step[] echo)

Probe, 2026-07-13, `gemini-3.1-flash-lite`, `probe.probe_step_echo()`.

`POST /v1beta/interactions` with `store: false`, no `previous_interaction_id`, and
an `input` of three steps — `user_input` ("What is the capital of France?"),
`model_output` ("Paris."), `user_input` ("And of Italy? Answer in one word.").

- HTTP status: `<status>`
- Verdict: `<verdict>`
- Answer: `<the model's answer>`

<One sentence: did the model use the supplied history, and what that means for the
interaction_stateless arm.>

This is the shape the `interaction_stateless` arm sends on every turn: the whole
conversation, the model's own prior answers included, with the server storing
nothing.
```

- [ ] **Step 4: Commit**

```bash
cd gemini_token_test && git add probe.py docs/interactions-api-fields.md
git commit -m "probe(interactions): can input carry a history the client already has?

The Step[] echo was doc-stated and never tested — the earlier probe only read
model_output steps back out of a stored interaction, never sent one. The
interaction_stateless arm rests entirely on the API accepting one, so settle it
before writing the arm."
```

---

### Task 2: Teach `interaction_client` to keep the history itself

**Files:**
- Modify: `interaction_client.py:44-66` (`_input`, `interaction_body`), `interaction_client.py:171-...` (`run_interaction`), and the module docstring
- Test: `tests/test_interaction_stateless.py` (create)

**Interfaces:**
- Consumes: nothing from Task 1 (the probe is a gate, not a dependency).
- Produces:
  - `interaction_client._step_user(text: str) -> dict`
  - `interaction_client._step_model(text: str) -> dict`
  - `interaction_client.interaction_body(model: str, text: str, system_instruction: str, prev_id: str | None, store: bool = True, history: list | None = None) -> dict`
  - `interaction_client.run_interaction(model, request_name="perf", turns=None, on_progress=None, inline_system=False, stage="interaction", client_history=False) -> dict` — same `{"params": {...}, "interaction_records": [...]}` return as today, with `params["client_history"]` and `params["store"]` added.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_interaction_stateless.py`:

```python
"""A fifth arm: the Interactions API with server-side state switched off.

The four existing arms vary two things at once -- which endpoint they call, and who
keeps the conversation history -- so `interaction` vs `stateless` never said which
of the two moved the numbers. This arm holds /interactions fixed and takes
previous_interaction_id away: store:false, and the client resends the whole
conversation as a Step[] every turn. What is left between it and `interaction` is
exactly what previous_interaction_id buys.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import interaction_client as ic


def _bodies(monkeypatch, turns=3, **kw):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=turns, **kw)
    return [json.loads(r["request_raw"]) for r in out["interaction_records"]]


# --- the wire shape -------------------------------------------------------

def test_it_never_stores_the_interaction(monkeypatch):
    for b in _bodies(monkeypatch, client_history=True):
        assert b["store"] is False


def test_it_never_chains_on_a_previous_interaction(monkeypatch):
    for b in _bodies(monkeypatch, client_history=True):
        assert "previous_interaction_id" not in b


def test_it_sends_the_system_prompt_every_turn(monkeypatch):
    system, _, _ = experiment.load_request("perf")
    for b in _bodies(monkeypatch, client_history=True):
        assert b["system_instruction"] == system


def test_the_history_grows_by_a_question_and_an_answer_each_turn(monkeypatch):
    bodies = _bodies(monkeypatch, client_history=True, turns=4)
    assert [len(b["input"]) for b in bodies] == [1, 3, 5, 7]


def test_the_steps_alternate_user_then_model(monkeypatch):
    steps = _bodies(monkeypatch, client_history=True, turns=3)[-1]["input"]
    kinds = [s["type"] for s in steps]
    assert kinds == ["user_input", "model_output", "user_input",
                     "model_output", "user_input"]


def test_the_history_carries_this_arms_own_answers(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = ic.run_interaction("gemini-3.1-flash-lite", turns=2, client_history=True)
    recs = out["interaction_records"]
    sent = json.loads(recs[1]["request_raw"])["input"]
    answered = recs[0]["response_text"]
    assert sent[1]["content"][0]["text"] == answered


def test_the_questions_go_in_in_order(monkeypatch):
    _, steps, _ = experiment.load_request("perf")
    sent = _bodies(monkeypatch, client_history=True, turns=3)[-1]["input"]
    asked = [s["content"][0]["text"] for s in sent if s["type"] == "user_input"]
    assert asked == steps[:3]


# --- the existing arms must not move --------------------------------------

def test_the_interaction_arm_still_chains_and_stores(monkeypatch):
    for k, b in enumerate(_bodies(monkeypatch), start=1):
        assert b["store"] is True
        assert isinstance(b["input"], list) and len(b["input"]) == 1
        if k > 1:
            assert b["previous_interaction_id"]


def test_the_inline_arm_still_chains_and_stores(monkeypatch):
    for k, b in enumerate(_bodies(monkeypatch, inline_system=True), start=1):
        assert b["store"] is True
        assert len(b["input"]) == 1
        if k > 1:
            assert b["previous_interaction_id"]


def test_client_history_is_off_by_default(monkeypatch):
    b = _bodies(monkeypatch, turns=1)[0]
    assert b["store"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd gemini_token_test && python3 -m pytest tests/test_interaction_stateless.py -v`

Expected: the `client_history=True` tests fail with `TypeError: run_interaction() got an unexpected keyword argument 'client_history'`. The three "must not move" tests pass already — that is the point of them; they are the regression guard, and they must still pass after Step 3.

- [ ] **Step 3: Implement it**

In `interaction_client.py`, replace the `_input` / `interaction_body` block:

```python
def _step_user(text: str) -> dict:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def _step_model(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def _input(text: str) -> list:
    return [_step_user(text)]


def interaction_body(model: str, text: str, system_instruction: str,
                     prev_id: str | None, store: bool = True,
                     history: list | None = None) -> dict:
    """One interaction request body.

    Two ways to give the server a conversation. Chained (the default): `input` is
    the new question alone and `previous_interaction_id` points at the interaction
    that holds everything before it. Client-side (`history`): `input` is the whole
    conversation as a Step[] -- prior questions, prior model answers, new question
    last -- and no previous_interaction_id is sent, because there is nothing stored
    to point at.

    system_instruction rides every turn either way: the server does not keep it.
    """
    body: dict = {
        "model": model,
        "stream": False,
        "store": store,
        "input": list(history) if history is not None else _input(text),
    }
    if system_instruction:
        body["system_instruction"] = system_instruction
    if prev_id and history is None:
        body["previous_interaction_id"] = prev_id
    return body
```

Then thread it through the two call sites. `_call_interaction` and `_mock_interaction` both build a body; give each the same two new parameters:

```python
def _mock_interaction(turn: int, text: str, system: str, prev_id: str | None,
                      store: bool = True, history: list | None = None) -> dict:
    iid = f"mock_interaction_{turn:03d}"
    ans = f"(mock interaction answer, turn {turn}, prev={prev_id or 'none'}) " + ("lorem ipsum " * 12)
    body = interaction_body("mock-model", text, system, prev_id, store=store,
                            history=history)
    req = json.dumps(body)
    inp = _text_tokens([{"parts": [{"text": (system if turn == 1 else '') + text}]}])
    usage = {"input_tokens": inp, "cached_tokens": 0, "output_tokens": 40,
             "thought_tokens": 0, "total_tokens": inp + 40}
    resp = json.dumps({"id": iid, "status": "completed",
                       "steps": [{"type": "model_output",
                                  "content": [{"type": "text", "text": ans}]}]})
    return _record(turn, text, ans, iid, usage,
                   len(req) + 200, len(resp) + 200, 0, req, resp, "")


def _call_interaction(model: str, text: str, system_instruction: str,
                      prev_id: str | None, turn: int = 1,
                      store: bool = True, history: list | None = None) -> dict:
    """One interaction request. Never raises; errors land in the record's error.
    Returns the common per-turn record."""
    body = interaction_body(model, text, system_instruction, prev_id, store=store,
                            history=history)
    req_raw = json.dumps(body)
    ...
```

Leave the rest of `_call_interaction` exactly as it is — only its signature and its first line change.

Then `run_interaction`:

```python
def run_interaction(model: str, request_name: str = "perf",
                    turns: int | None = None, on_progress=None,
                    inline_system: bool = False, stage: str = "interaction",
                    client_history: bool = False) -> dict:
```

Add to its docstring, after the `inline_system=True` paragraph:

```
    client_history=True: the server stores nothing (store:false) and there is no
    previous_interaction_id. The client keeps the conversation and resends it whole
    every turn as a Step[] in `input` -- its own prior questions and the model's own
    prior answers. This is the stateless arm's bargain, made on the interactions
    endpoint: what separates it from the chained arm is exactly what
    previous_interaction_id buys.
```

And its loop:

```python
    records = []
    prev_id: str | None = None
    history: list = []              # only used when client_history
    for k, q in enumerate(steps, start=1):
        if on_progress:
            on_progress({"stage": stage, "turn": k, "turns": n})
        if inline_system:
            # The prompt goes in the user turn, once, and the server holds it after.
            text = f"{system}\n\n{q}" if (k == 1 and system) else q
            sys_instruction = ""
        else:
            text, sys_instruction = q, system

        if client_history:
            history.append(_step_user(text))
            sent_history, store = list(history), False
        else:
            sent_history, store = None, True

        if is_mock():
            r = _mock_interaction(k, text, sys_instruction, prev_id,
                                  store=store, history=sent_history)
        else:
            r = _call_interaction(model, text, sys_instruction, prev_id, turn=k,
                                  store=store, history=sent_history)
        r["turn"] = k
        r["question"] = q          # the step, never the prompt bolted onto it
        if client_history:
            # The arm's own answer, not another arm's transcript. A history of a
            # conversation that never happened measures nothing.
            history.append(_step_model(r.get("response_text") or ""))
        elif r.get("interaction_id"):
            prev_id = r["interaction_id"]     # chain the next turn onto this one
        records.append(r)

    return {
        "params": {"mode": stage, "turns": n, "model": model,
                   "stream": False, "endpoint": interactions_url(),
                   "inline_system": inline_system,
                   "client_history": client_history, "store": not client_history,
                   "request_source": source},
        "interaction_records": records,
    }
```

Finally, extend the module docstring's body sample so it documents both shapes — add below the existing `body:` block:

```
  With client_history (the interaction_stateless arm) the trade flips: store is
  false, previous_interaction_id is gone, and `input` carries the whole
  conversation as steps.

  body: {"model": "...", "stream": False, "store": False,
         "system_instruction": "...",           # still re-sent every turn
         "input": [{"type":"user_input","content":[{"type":"text","text":"q1"}]},
                   {"type":"model_output","content":[{"type":"text","text":"a1"}]},
                   {"type":"user_input","content":[{"type":"text","text":"q2"}]}]}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd gemini_token_test && python3 -m pytest tests/test_interaction_stateless.py tests/test_interaction_inline.py tests/test_interaction_devapi.py -v`

Expected: PASS, all of them. `test_interaction_inline.py` and `test_interaction_devapi.py` passing unchanged is the proof that the existing arms did not move.

- [ ] **Step 5: Commit**

```bash
cd gemini_token_test && git add interaction_client.py tests/test_interaction_stateless.py
git commit -m "feat(interactions): let the client keep the history, and the server keep nothing

store:false, no previous_interaction_id, and the whole conversation resent as a
Step[] every turn. Same endpoint as the chained arms, so what separates them is
exactly what previous_interaction_id buys.

The history is built from this arm's own answers — a history of a conversation
that never happened measures nothing."
```

---

### Task 3: Wire the arm into the comparison

**Files:**
- Modify: `experiment.py:259-260` (`DEFAULT_ARMS`, `COMPARE_ARMS`), `experiment.py:375-405` (`_arm_interaction`), `experiment.py:427-440` (`_run_arm`)
- Test: `tests/test_interaction_stateless.py` (append)

**Interfaces:**
- Consumes: `interaction_client.run_interaction(..., client_history=True)` from Task 2.
- Produces: the arm name `"interaction_stateless"` in `experiment.DEFAULT_ARMS` and `experiment.COMPARE_ARMS`; per-turn records with `arm == "interaction_stateless"`, `phase == "steady"`, and the shared record fields every other arm emits.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_interaction_stateless.py`:

```python
# --- wired into the comparison --------------------------------------------

def test_the_arm_exists_and_is_a_headline_arm():
    assert "interaction_stateless" in experiment.COMPARE_ARMS
    assert "interaction_stateless" in experiment.DEFAULT_ARMS


def test_the_arm_produces_the_shared_record(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2,
                                    arms=["interaction_stateless"])
    recs = [r for r in out["records"] if r["arm"] == "interaction_stateless"]
    assert [r["turn"] for r in recs] == [1, 2]
    assert all(r["phase"] == "steady" for r in recs)
    assert all("wire_sent" in r and "input_tokens" in r for r in recs)


def test_all_three_interaction_arms_run_side_by_side(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison(
        "gemini-3.1-flash-lite", turns=1,
        arms=["interaction", "interaction_inline", "interaction_stateless"])
    assert {r["arm"] for r in out["records"]} == {
        "interaction", "interaction_inline", "interaction_stateless"}


def test_the_arm_resends_a_growing_history_through_the_comparison(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=3,
                                    arms=["interaction_stateless"])
    recs = sorted((r for r in out["records"] if r["arm"] == "interaction_stateless"),
                  key=lambda r: r["turn"])
    sizes = [len(json.loads(r["request_raw"])["input"]) for r in recs]
    assert sizes == [1, 3, 5]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd gemini_token_test && python3 -m pytest tests/test_interaction_stateless.py -v -k "comparison or headline or side_by_side or shared_record"`

Expected: FAIL — `assert "interaction_stateless" in experiment.COMPARE_ARMS`, and the run_comparison tests find no records (`_run_arm` returns `[]` for an unknown arm).

- [ ] **Step 3: Implement it**

In `experiment.py`, the arm tuples:

```python
DEFAULT_ARMS = ("stateless", "cached", "interaction", "interaction_inline",
                "interaction_stateless")
COMPARE_ARMS = ("stateless", "cached", "interaction", "interaction_inline",
                "interaction_stateless", "nocontext")
```

`_arm_interaction` gains the flag and passes it through — signature and call only, the record mapping below them is untouched:

```python
def _arm_interaction(model, request_name, turns, on_progress,
                     arm: str = "interaction", inline_system: bool = False,
                     client_history: bool = False) -> list:
    """Interactions API arm, mapped into the shared per-turn record.

    inline_system moves the system prompt out of system_instruction and into the
    first user turn, so the server-side history carries it and later turns send only
    their question. Same content reaches the model; a different party stores it.

    client_history takes the server-side history away entirely: store:false, no
    previous_interaction_id, and the whole conversation resent as steps every turn.
    """
    from interaction_client import run_interaction   # lazy: avoids import cycle
    out = run_interaction(model, request_name=request_name, turns=turns,
                          on_progress=on_progress, inline_system=inline_system,
                          stage=arm, client_history=client_history)
```

And `_run_arm` gains a branch, after the `interaction_inline` one:

```python
    if arm == "interaction_stateless":
        return _arm_interaction(model, request_name, n, on_progress,
                                arm="interaction_stateless", client_history=True)
```

- [ ] **Step 4: Run the whole suite**

Run: `cd gemini_token_test && python3 -m pytest tests/ -v`

Expected: PASS. Pay attention to `tests/test_compare_capture.py` — it burned once on `_` in arm labels, so a fresh underscored arm name is exactly what it exists to catch.

- [ ] **Step 5: Commit**

```bash
cd gemini_token_test && git add experiment.py tests/test_interaction_stateless.py
git commit -m "feat(arm): interaction_stateless — the interactions endpoint, no server-side state

Fills the empty cell of the 2x2. The four existing arms varied endpoint and
history-owner at once, so interaction-vs-stateless never told us which one moved
the numbers. This one holds the endpoint fixed and takes only the server-side
history away."
```

---

### Task 4: Give the arm a line on the chart and a sentence on the page

Five arms now overlay on two axes. Without its own color, dash, and marker the new arm is drawn under `stateless` — which is exactly where the interesting comparison is, and exactly where a hidden line is worst.

**Files:**
- Modify: `static/app.js:35-51` (`ARM_LABELS`, `ARM_COLORS`), `static/app.js:256-259` (`ARM_DASH`, `ARM_POINT`)
- Modify: `templates/index.html:104-111` (the interaction-arms paragraph)

**Interfaces:**
- Consumes: the arm name `"interaction_stateless"` from Task 3. The arm checkboxes are rendered from the server's arm list (`{% for a in arms %}`), so no template change is needed to make it selectable.

- [ ] **Step 1: Label and color it**

In `static/app.js`:

```javascript
const ARM_LABELS = {
  stateless: "Stateless (full resend)",
  cached: "Cached (build + reference)",
  interaction: "Interaction (system_instruction each turn)",
  interaction_inline: "Interaction (system prompt in 1st user turn)",
  interaction_stateless: "Interaction (client-side history, store:false)",
  nocontext: "Stateless — no context",
  cachebuild: "Building caches",
};

const ARM_COLORS = {
  stateless: "#ff6b6b",
  cached: "#4dd4ac",
  interaction: "#5b8def",
  interaction_inline: "#c58af9",
  interaction_stateless: "#f78c6b",
  nocontext: "#f6c453",
};
```

- [ ] **Step 2: Give it a dash and a marker**

Same file, inside the chart builder:

```javascript
  const ARM_DASH = { stateless: [], cached: [], interaction: [7, 4],
                     interaction_inline: [3, 3], interaction_stateless: [6, 2, 2, 2],
                     nocontext: [2, 3] };
  const ARM_POINT = { stateless: "circle", cached: "rect", interaction: "triangle",
                      interaction_inline: "star", interaction_stateless: "rectRot",
                      nocontext: "cross" };
```

`interaction_stateless` is expected to land on top of `stateless` on both axes — same payload, different endpoint — so the dash-dot and the diamond marker are what keep both readable where they coincide.

- [ ] **Step 3: Say what the arm is for**

In `templates/index.html`, replace the paragraph beginning "The two interaction arms differ only in who stores the system prompt":

```html
    <p class="sub">The three interaction arms differ in who stores what.
      <code>interaction</code> sends the system prompt as <code>system_instruction</code>,
      which is interaction-scoped and so goes back over the wire every single turn — a
      20 KB prompt is then essentially the whole request, every turn — while the server
      keeps the conversation. <code>interaction_inline</code> puts the prompt in the first
      user message instead, which makes it part of that server-side history, so later turns
      send only their question. <code>interaction_stateless</code> takes the server-side
      history away altogether (<code>store:false</code>, no
      <code>previous_interaction_id</code>): the client resends the whole conversation
      every turn, which is the bargain <code>stateless</code> makes on the other endpoint.
      Its gap to <code>interaction</code> is exactly what
      <code>previous_interaction_id</code> buys; its gap to <code>stateless</code> is what
      the interactions protocol costs. The model sees the same content in all three;
      whether it still <em>obeys</em> it as well is what the responses CSV is for.</p>
```

- [ ] **Step 4: See it**

Run: `cd gemini_token_test && GEMINI_MOCK=1 python3 app.py` and open the comparison page. Tick all five arms, run 3 turns, and confirm: the new checkbox is there, its series is drawn in its own color with the dash-dot pattern, and the legend reads "Interaction (client-side history, store:false)". Stop the server.

- [ ] **Step 5: Commit**

```bash
cd gemini_token_test && git add static/app.js templates/index.html
git commit -m "feat(ui): draw the fifth arm, and say what it isolates

interaction_stateless lands on top of stateless on both axes — same payload, a
different endpoint — so it needs its own dash and marker or one of the two reads
as a missing arm."
```

---

### Task 5: Verify against the live API

Mock runs prove the shape. Only a live run proves the history reaches the model and the bytes go where the spec says they go.

**Files:**
- Modify: `docs/README.md` (add the arm to whatever list of arms it carries; read it first and match its existing register)

- [ ] **Step 1: Run three turns live — billable, five arms**

Run:

```bash
cd gemini_token_test && curl -s -X POST localhost:8000/compare \
  -H 'Content-Type: application/json' \
  -d '{"turns": 3, "arms": ["stateless", "interaction", "interaction_stateless"], "pause_seconds": 20}' \
  | python3 -m json.tool > /tmp/verify.json
```

(Start the app first without `GEMINI_MOCK`, with `GEMINI_API_KEY` set.)

- [ ] **Step 2: Check the three things that can be quietly wrong**

Run:

```bash
cd gemini_token_test && python3 - <<'PY'
import json
recs = json.load(open("/tmp/verify.json"))["records"]
for arm in ("stateless", "interaction", "interaction_stateless"):
    rs = sorted((r for r in recs if r["arm"] == arm), key=lambda r: r["turn"])
    print(arm)
    for r in rs:
        print(f"  turn {r['turn']}: sent={r['wire_sent']:>7}  "
              f"in_tok={r['input_tokens']:>6}  ms={r['elapsed_ms']:>6}  "
              f"err={r['error'][:60]}")
PY
```

Three assertions, and each failure means something different:

1. **No errors on `interaction_stateless`.** An `http_400` here means the arm's body is being rejected — the Task 1 probe passed with a three-step history, so a failure at ten steps is a size or a shape the probe did not reach.
2. **`wire_sent` grows turn over turn.** Flat bytes mean the history is not being resent, and the arm is silently measuring `nocontext`.
3. **`input_tokens` tracks `stateless` within noise, and grows superlinearly.** If it is flat, the server accepted the steps and ignored them — the arm would then be measuring a model answering with no context at all, which is the one failure mode that produces plausible-looking numbers. This is the check that matters most.

If (3) fails, stop and report. Do not paper over it.

- [ ] **Step 3: Write the arm into the README**

Read `docs/README.md`, find where it enumerates the arms, and add `interaction_stateless` in the same voice as the others — one line, saying it calls `/interactions` with `store:false` and a client-kept history, and that its gap to `interaction` is what `previous_interaction_id` buys.

- [ ] **Step 4: Commit**

```bash
cd gemini_token_test && git add docs/README.md
git commit -m "docs: the fifth arm, and what its live run showed"
```

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: the Phase 0 probe and its Measured section → Task 1; `interaction_client.py` (`_step_user`/`_step_model`, `interaction_body(store, history)`, `run_interaction(client_history)`) → Task 2; `experiment.py` arm tuples and `_run_arm` dispatch → Task 3; `static/app.js` and `templates/index.html` → Task 4; the three-turn live verification and `docs/README.md` → Task 5. The spec's regression requirement ("existing arms unchanged, byte for byte") is pinned by three tests in Task 2 and by re-running `test_interaction_inline.py` and `test_interaction_devapi.py` in Task 2 Step 4. The spec's pcap-naming risk is covered by running `test_compare_capture.py` in Task 3 Step 4.

**Placeholders.** One deliberate fill-in remains: Task 1 Step 3's Measured section quotes `<status>`, `<verdict>`, and the model's answer, because those values do not exist until the probe runs. Every other step carries its literal content.

**Type consistency.** `_step_user` / `_step_model` are defined twice on purpose — once in `probe.py` (Task 1) and once in `interaction_client.py` (Task 2). `probe.py` does not import from `interaction_client`, and `interaction_client` imports from `gemini_client` and `experiment` only; wiring a new dependency between the probe and the arm to save four lines would be the wrong trade. The names and the returned dicts are identical in both. `client_history` is the parameter name in `run_interaction`, `_arm_interaction`, and both bodies-builders; `store` and `history` are the parameter names in `interaction_body`, `_call_interaction`, and `_mock_interaction`. The arm name is the string `"interaction_stateless"` everywhere — in the two arm tuples, the `_run_arm` branch, the `stage`/`arm` argument, `ARM_LABELS`, `ARM_COLORS`, `ARM_DASH`, `ARM_POINT`, and the tests.
