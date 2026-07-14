"""Three clocks per turn, because one of them hides the store.

Measured 2026-07-14 on /interactions: with `store:true` the last token of the answer
reaches the client at ~950 ms and the stream then stays open ~1.8 s more while the
server persists the interaction. A blocking client waits for all of it. A streaming
one does not. So every arm streams, and every turn reports:

    ttft_ms      -> first event carrying answer text
    ttlt_ms      -> last event carrying answer text
    turn_end_ms  -> stream closed

turn_end == ttlt on the generateContent arms (nothing happens after the last token).
On a stored interaction it does not, and the difference is the write.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment
import interaction_client as ic
import streaming


class FakeResp:
    """An SSE response whose lines arrive on a clock we control."""

    def __init__(self, lines, status=200):
        self.status_code = status
        self._lines = lines
        self.text = "\n".join(lines)

    def iter_lines(self):
        for line in self._lines:
            yield line.encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


GEN_LINES = [
    'data: {"candidates":[{"content":{"parts":[{"text":"Pa"}],"role":"model"}}]}',
    'data: {"candidates":[{"content":{"parts":[{"text":"ris."}],"role":"model"}}]}',
    'data: {"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"SIG"}],'
    '"role":"model"}}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":4,'
    '"totalTokenCount":20,"thoughtsTokenCount":7}}',
]

INT_LINES = [
    'data: {"event_type":"interaction.created","interaction":{"id":"int_1",'
    '"status":"in_progress"}}',
    'data: {"event_type":"step.start","index":0,"step":{"type":"thought"}}',
    'data: {"event_type":"step.delta","index":0,"delta":{"signature":"SIG"}}',
    'data: {"event_type":"step.stop","index":0}',
    'data: {"event_type":"step.start","index":1,"step":{"type":"model_output"}}',
    'data: {"event_type":"step.delta","index":1,"delta":{"text":"Paris.","type":"text"}}',
    'data: {"event_type":"step.stop","index":1}',
    'data: {"event_type":"interaction.completed","interaction":{"id":"int_1",'
    '"status":"completed","usage":{"total_input_tokens":9,"total_output_tokens":4,'
    '"total_tokens":20,"total_thought_tokens":7,"total_cached_tokens":0}}}',
    'data: [DONE]',
]


# --- the reader ------------------------------------------------------------

def test_the_answer_is_the_text_that_streamed():
    out = streaming.read_stream(FakeResp(GEN_LINES), streaming.gen_text, 0.0)
    assert out.text == "Paris."


def test_a_thought_part_does_not_start_the_ttft_clock():
    """A reasoning summary is not the answer. If it counted, TTFT would report the
    moment the model started thinking out loud."""
    lines = ['data: {"candidates":[{"content":{"parts":[{"text":"hmm","thought":true}],'
             '"role":"model"}}]}'] + GEN_LINES
    out = streaming.read_stream(FakeResp(lines), streaming.gen_text, 0.0)
    assert out.text == "Paris."


def test_the_timings_are_ordered():
    out = streaming.read_stream(FakeResp(INT_LINES), streaming.interaction_text, 0.0)
    assert out.ttft_ms <= out.ttlt_ms <= out.turn_end_ms


def test_an_answerless_stream_still_ends():
    """No text at all -- an empty answer, or an error stream. A zero TTFT would read
    as 'instant' rather than 'never'."""
    out = streaming.read_stream(FakeResp(['data: [DONE]']), streaming.gen_text, 0.0)
    assert out.ttft_ms == out.ttlt_ms == out.turn_end_ms


# --- rebuilding the non-streamed body --------------------------------------

def test_generatecontent_chunks_rebuild_into_a_response():
    out = streaming.read_stream(FakeResp(GEN_LINES), streaming.gen_text, 0.0)
    data = streaming.gen_response(out.events)
    parts = data["candidates"][0]["content"]["parts"]
    assert "".join(p.get("text", "") for p in parts) == "Paris."
    assert any(p.get("thoughtSignature") == "SIG" for p in parts)
    assert data["usageMetadata"]["promptTokenCount"] == 9


def test_interaction_events_rebuild_the_steps():
    """The completed event carries usage but *not* the steps (measured). If the deltas
    are not reassembled here, a client-side history has nothing to echo."""
    out = streaming.read_stream(FakeResp(INT_LINES), streaming.interaction_text, 0.0)
    data = streaming.interaction_response(out.events)
    assert data["id"] == "int_1"
    assert [s["type"] for s in data["steps"]] == ["thought", "model_output"]
    assert data["steps"][0]["signature"] == "SIG"
    assert data["steps"][1]["content"][0]["text"] == "Paris."
    assert data["usage"]["total_input_tokens"] == 9


# --- the arms carry the timings --------------------------------------------

def test_every_arm_reports_five_marks(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2,
                                    arms=list(experiment.COMPARE_ARMS))
    for r in out["records"]:
        assert {"req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms",
                "turn_end_ms"} <= set(r), r["arm"]
        assert (r["req_sent_ms"] <= r["ttfb_ms"] <= r["ttft_ms"]
                <= r["ttlt_ms"] <= r["turn_end_ms"]), r["arm"]


def test_a_bigger_upload_costs_more_to_send(monkeypatch):
    """req_sent is where a resent history is actually paid for. An arm whose payload
    grows every turn must show that growth here, or the mark measures nothing."""
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=4,
                                    arms=["stateless"])
    sent = [r["req_sent_ms"] for r in out["records"] if r["phase"] == "steady"]
    assert sent == sorted(sent) and sent[-1] > sent[0], sent


def test_the_client_uplink_is_reported_apart_from_the_download(monkeypatch):
    """What the client uploads is its own bandwidth bill, and the axis the arms
    differ on. Folded into one wire total it cannot be read off."""
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2,
                                    arms=["stateless", "nocontext"])
    import metrics
    summary = metrics.summarize_comparison(out)
    for arm in ("stateless", "nocontext"):
        t = summary["totals"][arm]
        assert t["steady_wire_sent"] + t["steady_wire_recv"] == t["steady_wire"]
        assert summary["series"][arm]["cum_wire_sent"]
    # The arm that resends its history uploads more than the one that sends nothing.
    assert (summary["totals"]["stateless"]["steady_wire_sent"]
            > summary["totals"]["nocontext"]["steady_wire_sent"])


def test_the_stateless_arms_end_when_the_answer_ends(monkeypatch):
    """Nothing happens after the last token on generateContent, so turn_end is ttlt."""
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2,
                                    arms=["stateless", "nocontext"])
    for r in [x for x in out["records"] if x["phase"] == "steady"]:
        assert r["turn_end_ms"] == r["ttlt_ms"], r["arm"]


def test_the_stored_interaction_pays_a_tail_after_the_answer(monkeypatch):
    """store:true keeps the stream open past the last token. An arm whose turn_end
    equals its ttlt would be hiding the write."""
    monkeypatch.setenv("GEMINI_MOCK", "1")
    recs = ic.run_interaction("gemini-3.1-flash-lite", turns=2)["interaction_records"]
    for r in recs:
        assert r["turn_end_ms"] - r["ttlt_ms"] == ic.MOCK_STORE_TAIL_MS


def test_the_unstored_interaction_pays_no_tail(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    recs = ic.run_interaction("gemini-3.1-flash-lite", turns=2,
                              client_history=True)["interaction_records"]
    for r in recs:
        assert r["turn_end_ms"] == r["ttlt_ms"]


def test_every_arm_streams(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    recs = ic.run_interaction("gemini-3.1-flash-lite", turns=1)["interaction_records"]
    assert json.loads(recs[0]["request_raw"])["stream"] is True


# --- the summary and the CSV -----------------------------------------------

def test_the_summary_averages_the_three_clocks(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    out = experiment.run_comparison("gemini-3.1-flash-lite", turns=2,
                                    arms=["stateless", "interaction"])
    import metrics
    totals = metrics.summarize_comparison(out)["totals"]
    for arm in ("stateless", "interaction"):
        for key in ("ttft", "ttlt", "turn_end", "store_tail_ms"):
            assert "mean" in totals[arm][key] and "median" in totals[arm][key]
    assert totals["stateless"]["store_tail_ms"]["mean"] == 0
    assert totals["interaction"]["store_tail_ms"]["mean"] == ic.MOCK_STORE_TAIL_MS


def test_the_csv_carries_the_timings():
    import app
    for col in ("ttft_ms", "ttlt_ms", "turn_end_ms", "store_tail_ms"):
        assert col in app.CASE_COLUMNS
    row = app._case_row({"arm": "interaction", "ttft_ms": 300, "ttlt_ms": 800,
                         "turn_end_ms": 2600, "elapsed_ms": 2600})
    assert row[app.CASE_COLUMNS.index("store_tail_ms")] == 1800
