"""Each arm must put on the wire exactly what the experiment claims it does.

If responses_stateful quietly resent the history, the result would be a tautology;
if the mock billed the stateful arm for what it uploaded rather than for what the
server holds, the headline finding would disappear into the fixture. These tests
pin both.

Offline: mock mode, no key, no socket.
"""

from __future__ import annotations

import json

import pytest

from providers import openai as p


@pytest.fixture(autouse=True)
def mock(monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p.reset_mock()
    yield
    p.reset_mock()


SYSTEM = "You are a terse assistant. " * 40
STEPS = [f"Question number {k}, and some words to give it a body." for k in range(1, 7)]


def _run(arm, turns=6, measure="bytes"):
    return p.run_arm(arm, "gpt-4.1-nano", SYSTEM, STEPS[:turns], measure)


def _steady(records):
    return [r for r in records if r["phase"] == "steady"]


def _bodies(records):
    return [json.loads(r["request_raw"]) for r in _steady(records)]


def test_ready_without_a_key_in_mock_mode():
    ok, why = p.ready()
    assert ok and "mock" in why


def test_every_arm_produces_a_record_per_turn():
    for arm in p.HEADLINE_ARMS:
        p.reset_mock()
        assert len(_steady(_run(arm))) == len(STEPS)


def test_chat_stateless_resends_everything():
    bodies = _bodies(_run("chat_stateless"))
    for k, body in enumerate(bodies, start=1):
        msgs = body["messages"]
        assert msgs[0]["role"] == "system"
        # system + the k-1 completed turns (user + assistant) + the new question
        assert len(msgs) == 1 + 2 * (k - 1) + 1
        assert msgs[-1]["role"] == "user"
        assert body["stream"] is False


def test_responses_stateless_is_the_control_arm():
    bodies = _bodies(_run("responses_stateless"))
    for k, body in enumerate(bodies, start=1):
        assert body["input"][0]["role"] == "system"
        assert body["store"] is False, "store=false, or it is not a stateless arm"
        assert "conversation" not in body
        assert len(body["input"]) == 1 + 2 * (k - 1) + 1


def test_responses_stateful_sends_only_the_new_question():
    records = _run("responses_stateful")
    for body in _bodies(records):
        assert len(body["input"]) == 1
        assert body["input"][0]["role"] == "user"
        assert "system" not in json.dumps(body["input"]), \
            "the system prompt must not ride along on any turn"


def test_the_conversation_create_is_a_setup_record():
    records = _run("responses_stateful")
    setup = records[0]
    assert setup["phase"] == "setup", "prep is not traffic; metrics keeps it out of the totals"
    assert setup["turn"] == 0
    assert setup["req_payload_bytes"] > 0, "the setup upload is counted, not hidden"
    assert SYSTEM in setup["request_raw"], "setup is where the system prompt is uploaded"
    # 0 tokens here is measured, not assumed: the endpoint runs no inference and returns
    # no usage object. `kind` and `billed` are what say so, and keep core.metrics from
    # adding this row's zeros to a Gemini transcript call's real input tokens.
    assert setup["kind"] == "conversation_create"
    assert setup["billed"] is False
    assert setup["input_tokens"] == 0


# --------------------------------------------- the two arms that store it differently

def test_the_inline_arm_creates_an_empty_conversation():
    """`items` is optional on POST /v1/conversations, so the create is a bare container.

    The system prompt then rides turn 1, inside the measured window, instead of being
    uploaded out-of-band where it makes the arm look 21 KB cheaper than the identical one.
    """
    records = _run("responses_stateful_inline", turns=3)
    setup = records[0]
    assert setup["phase"] == "setup"
    assert SYSTEM not in setup["request_raw"], "the prompt does not go up here"
    assert json.loads(setup["request_raw"]) == {}

    seeded = _run("responses_stateful")[0]
    assert setup["req_payload_bytes"] < seeded["req_payload_bytes"] / 50


def test_the_inline_arm_uploads_the_system_prompt_once_on_turn_one():
    bodies = _bodies(_run("responses_stateful_inline", turns=3))
    first, rest = bodies[0], bodies[1:]

    roles = [it["role"] for it in first["input"]]
    assert roles == ["system", "user"], "turn 1 carries the prompt as a stored item"
    # An input item, never `instructions`: `instructions` is not stored, and this arm
    # would silently become the chained one.
    assert "instructions" not in first

    for body in rest:
        assert [it["role"] for it in body["input"]] == ["user"]
        assert SYSTEM not in json.dumps(body), "the server kept it; nobody resends it"


def test_the_chained_arm_resends_the_system_prompt_every_turn():
    """The OpenAI-side twin of gemini's `interaction`: the server holds the history and
    does not hold the system prompt. `instructions` is top-level and is not stored, so it
    must be resent with every request (OpenAI, migrate-to-responses)."""
    bodies = _bodies(_run("responses_chained", turns=3))
    for body in bodies:
        assert body["instructions"] == SYSTEM
        assert [it["role"] for it in body["input"]] == ["user"]
        assert body["store"] is True
        assert "conversation" not in body


def test_the_chain_is_linked_turn_to_turn():
    records = _steady(_run("responses_chained", turns=3))
    bodies = [json.loads(r["request_raw"]) for r in records]
    ids = [json.loads(r["response_raw"])["id"] for r in records]

    assert "previous_response_id" not in bodies[0], "turn 1 has nothing to chain onto"
    assert bodies[1]["previous_response_id"] == ids[0]
    assert bodies[2]["previous_response_id"] == ids[1]


def test_the_chained_arms_upload_is_flat_but_not_small():
    """It stops resending the *history* and keeps resending the *system prompt*, so its
    uplink is flat -- and flat at roughly the size of the prompt. That is the whole point
    of having it: `responses_stateful_inline` shows what the other half buys."""
    chained = [r["req_payload_bytes"] for r in _steady(_run("responses_chained", turns=4))]
    p.reset_mock()
    inline = [r["req_payload_bytes"]
              for r in _steady(_run("responses_stateful_inline", turns=4))]

    spread = max(chained) - min(chained)
    assert spread < 0.2 * min(chained), "flat: it is not accumulating a history"
    assert min(chained) > len(SYSTEM), "but it carries the system prompt every turn"
    # The gap between the two arms *is* the system prompt: same endpoint, same stored
    # history, same question -- the only thing chained still uploads and inline does not.
    # Asserted as a difference rather than a ratio, so the test does not quietly depend
    # on how big the fixture's prompt happens to be.
    assert chained[-1] - inline[-1] > 0.9 * len(SYSTEM)


def test_storing_the_prompt_server_side_does_not_stop_it_being_billed():
    """Three ways to stop uploading the history. None of them stops paying for it.

    Measured live, 2 turns: chat_stateless uploaded 21866 B on turn 1 and was billed 4338
    input tokens; responses_stateful uploaded 1176 B and was billed 4338. Identical. A
    mock that let any stateful arm look cheaper in *tokens* would erase the finding.
    """
    per_arm = {}
    for arm in ("chat_stateless", "responses_chained", "responses_stateful_inline",
                "responses_stateful"):
        p.reset_mock()
        per_arm[arm] = [r["input_tokens"] for r in _steady(_run(arm, turns=4))]

    for arm, tokens in per_arm.items():
        assert tokens == sorted(tokens), f"{arm}: input tokens must grow with the history"
        assert tokens[0] > len(SYSTEM) // 8, \
            f"{arm}: turn 1 is billed for the system prompt however it got there"


def test_the_conversation_id_is_carried_from_turn_one():
    records = _run("responses_stateful")
    conv = records[0]["conversation"]
    assert conv
    for body in _bodies(records):
        assert body["conversation"] == conv


def test_stateless_upload_grows_while_stateful_stays_flat():
    """The shape of the whole experiment, in one test."""
    chat = [r["req_payload_bytes"] for r in _steady(_run("chat_stateless"))]
    p.reset_mock()
    resp = [r["req_payload_bytes"] for r in _steady(_run("responses_stateless"))]
    p.reset_mock()
    stateful = [r["req_payload_bytes"] for r in _steady(_run("responses_stateful"))]

    for series in (chat, resp):
        assert series == sorted(series) and series[-1] > series[0], \
            "a client holding the history re-uploads more of it every turn"

    # The stateful turns carry a question and nothing else, so they track question
    # length rather than history length.
    assert max(stateful) < min(chat), \
        "every stateful turn uploads less than the smallest stateless turn"
    assert sum(stateful) * 5 < sum(chat), "the cumulative gap is not marginal"


def test_the_billing_gap_survives_the_collapse_in_bytes():
    """The headline finding: the bytes collapse and the billing does not.

    OpenAI bills all previous input tokens in the chain, so an arm that stopped
    uploading the history still pays for it. A mock that let input_tokens go flat
    here would be arguing the experiment's conclusion away.
    """
    records = _steady(_run("responses_stateful"))
    uploaded = [r["req_payload_bytes"] for r in records]
    billed = [r["input_tokens"] for r in records]

    p.reset_mock()
    resent = _steady(_run("responses_stateless"))
    resent_uploaded = [r["req_payload_bytes"] for r in resent]
    resent_billed = [r["input_tokens"] for r in resent]

    # Bytes: one arm's upload grows every turn, the other's does not move.
    assert max(uploaded) - min(uploaded) < 40, "the upload is flat: only a question goes up"
    assert resent_uploaded == sorted(resent_uploaded) and resent_uploaded[-1] > resent_uploaded[0], \
        "the arm that re-sends the history uploads more of it every turn"

    # Billing: both arms are charged for a history that grows every turn. The stateful
    # arm stopped putting that history on the wire and is billed for it all the same --
    # OpenAI bills every previous input token in the chain. The bytes can be saved; the
    # billing does not follow, and this is the run refusing to pretend otherwise.
    for series, who in ((billed, "stateful"), (resent_billed, "stateless")):
        assert series == sorted(series) and series[-1] > series[0], \
            f"{who}: input_tokens grow turn over turn"
    assert sum(billed) > 4 * (sum(uploaded) / 4), \
        "the stateful arm is billed for far more input than it ever uploaded"


def test_a_reasoning_item_is_echoed_back_verbatim(monkeypatch):
    """Rule 1: what the server sent goes back exactly as it came off the wire.

    Rebuilding the assistant turn from its answer text would drop the reasoning item
    and under-report what a real client uploads.
    """
    monkeypatch.setattr(p, "REASONING_EFFORT", "low")
    bodies = _bodies(_run("responses_stateless", turns=3))

    echoed = [it for it in bodies[-1]["input"] if it.get("type") == "reasoning"]
    assert len(echoed) == 2, "one reasoning item per completed turn, carried forward"
    assert all(it["encrypted_content"] for it in echoed), \
        "the opaque payload rides along; it is bytes a real client pays for"


def test_a_reasoning_summary_never_becomes_the_answer():
    """Rule 2: reasoning text is not the answer and must not start the TTFT clock."""
    assert p._responses_text_of(
        {"type": "response.reasoning_summary_text.delta", "delta": "hmm..."}) == ""
    assert p._responses_text_of(
        {"type": "response.output_text.delta", "delta": "the answer"}) == "the answer"


def test_latency_marks_only_exist_on_a_timed_pass():
    """A total is not a TTFT. A bytes pass reports zeros rather than a copied number."""
    byte_pass = _steady(_run("chat_stateless", turns=2, measure="bytes"))
    assert all(r["ttft_ms"] == 0 for r in byte_pass)

    p.reset_mock()
    timed = _steady(_run("chat_stateless", turns=2, measure="latency"))
    for r in timed:
        assert 0 < r["ttfb_ms"] <= r["ttft_ms"] <= r["ttlt_ms"] <= r["turn_end_ms"]


def _live_calls(monkeypatch, arm, measure):
    """(url, measure) per call the arm would have put on the wire.

    Mock mode short-circuits core.call, and the bug this covers lives in what core.call
    is asked to do -- so the mock has to be off and `send` is what gets spied on.
    """
    # Both switches: conftest pins TRAFFIC_MOCK *and* OPENAI_MOCK, and either one alone
    # keeps _send on the synthetic path. `send` is stubbed, so nothing reaches a socket.
    monkeypatch.delenv("TRAFFIC_MOCK", raising=False)
    monkeypatch.delenv("OPENAI_MOCK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = []

    def spy(url, headers, body, *, measure, text_of, stream_body=None, rebuild=None,
            timeout=180):
        calls.append((url, measure))
        return p.Exchange(status=200, response={"id": "conv_1", "output": [],
                                                "usage": {}})

    monkeypatch.setattr(p, "send", spy)
    p.run_arm(arm, "gpt-4.1-nano", SYSTEM, STEPS[:2], measure)
    return calls


@pytest.mark.parametrize("measure", ["bytes", "latency", "both"])
def test_the_conversation_create_is_never_streamed(monkeypatch, measure):
    """/v1/conversations has no stream to open, and it rejects the parameter outright:

        400 invalid_request_error: Unknown parameter: 'stream'.

    The create used to be handed the run's `measure`, so every `latency` and `both` run
    of this arm died on its prep call -- before its first question, with no conversation
    id to chain the turns onto. Prep is not a turn: it is blocking whatever the run is.
    """
    calls = _live_calls(monkeypatch, "responses_stateful", measure)
    create = [(url, m) for url, m in calls if url.endswith("/conversations")]
    assert create == [(create[0][0], "bytes")]
    # And the turns still get the pass the operator asked for.
    assert all(m == measure for url, m in calls if url.endswith("/responses"))


def test_the_prompt_cache_key_rotates_with_the_run():
    """The key is a routing hint, not a namespace -- prefix isolation is core.cachebust's
    job. But an un-rotated key routes this run's calls at the node holding *last* run's
    prefix, which is a node with nothing of ours on it."""
    from core import cachebust

    cachebust.begin("2026-07-14T09:52:30+00:00")
    first = p._cache_key("chat_stateless")
    assert first != p._cache_key("responses_stateless")

    cachebust.begin("2026-07-14T10:11:00+00:00")
    assert p._cache_key("chat_stateless") != first

    cachebust.begin("2026-07-14T10:11:00+00:00", enabled=False)
    assert p._cache_key("chat_stateless") == "tt-openai-chat_stateless"
