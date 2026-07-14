"""The SSE reader's job is the marks, and the marks are where it can lie.

Three ways it could lie, one test each:

  - Let reasoning text start the TTFT clock. A model that thinks for 400 ms before it
    speaks would then report a TTFT of ~0 and look faster than a model that answers
    immediately. That is the exact number a reader uses to choose, and it would be
    backwards.
  - Report 0 for a stream that carried no answer. Zero reads as "instant"; the truth is
    "never". The turn still ended, and that is where the answer marks belong.
  - Drift the clocks out of order, which makes `store_tail` (turn_end - ttlt) negative
    and quietly meaningless.
"""

from __future__ import annotations

import json
import time

from core.streaming import StreamResult, read_stream, since


class _FakeResponse:
    """An SSE response that arrives over time, like a real one.

    The reader stamps its marks as lines arrive, so a fixture that hands over the whole
    body at once would let every mark land in the same microsecond and prove nothing
    about ordering. The delays are what make the assertions real.
    """

    def __init__(self, lines, status=200, gap=0.02):
        self._lines = lines
        self._gap = gap
        self.status_code = status

    def iter_lines(self):
        for line in self._lines:
            time.sleep(self._gap)
            yield line.encode("utf-8")


def _data(payload: dict) -> str:
    return "data: " + json.dumps(payload)


def _answer_text(event: dict) -> str:
    """A provider's text_of: the answer, and only the answer. A part flagged `thought`
    is the model reasoning aloud, not answering."""
    out = []
    for part in event.get("parts") or []:
        if part.get("thought"):
            continue
        out.append(part.get("text") or "")
    return "".join(out)


def test_marks_are_ordered_and_bracket_the_answer():
    lines = [
        _data({"parts": [{"text": "Hello "}]}),
        _data({"parts": [{"text": "world"}]}),
        _data({"usage": {"output_tokens": 2}}),     # trailing, carries no answer text
    ]
    t0 = time.monotonic()
    out = read_stream(_FakeResponse(lines), _answer_text, t0)

    assert out.status == 200
    assert out.text == "Hello world"
    assert len(out.events) == 3
    assert 0 < out.ttft_ms < out.ttlt_ms < out.turn_end_ms
    # The stream stayed open past the last token -- a usage frame here, a persisted
    # interaction in production. That gap is store_tail, and it must be positive.
    assert out.turn_end_ms - out.ttlt_ms > 0


def test_reasoning_text_does_not_start_the_ttft_clock():
    lines = [
        _data({"parts": [{"text": "let me think about this", "thought": True}]}),
        _data({"parts": [{"text": "Paris"}]}),
    ]
    t0 = time.monotonic()
    out = read_stream(_FakeResponse(lines, gap=0.03), _answer_text, t0)

    assert out.text == "Paris"                       # the reasoning is not the answer
    assert len(out.events) == 2                      # but it was still on the wire
    # TTFT is the second event, not the first: the thinking is time the user waits.
    assert out.ttft_ms >= 55


def test_answerless_stream_pins_the_answer_marks_to_the_end():
    lines = [
        _data({"parts": [{"text": "thinking", "thought": True}]}),
        _data({"usage": {"output_tokens": 0}}),
    ]
    t0 = time.monotonic()
    out = read_stream(_FakeResponse(lines), _answer_text, t0)

    assert out.text == ""
    # Never, not instantly. A zero here would chart as the fastest turn of the run.
    assert out.ttft_ms == out.turn_end_ms
    assert out.ttlt_ms == out.turn_end_ms
    assert out.turn_end_ms > 0
    # And the store tail computed from these is zero, not the whole turn.
    assert out.turn_end_ms - out.ttlt_ms == 0


def test_done_sentinel_and_junk_lines_are_not_events():
    lines = [
        ": keep-alive comment",
        _data({"parts": [{"text": "hi"}]}),
        "data: not-json",
        "data: [DONE]",
    ]
    out = read_stream(_FakeResponse(lines, gap=0), _answer_text, time.monotonic())

    assert out.text == "hi"
    assert len(out.events) == 1
    # The raw body keeps only what parsed; the bytes of the rest were already counted on
    # the socket, which is where bytes belong.
    assert "hi" in out.raw


def test_since_falls_back_rather_than_reporting_zero():
    t0 = time.monotonic()
    assert since(t0, None) == 0
    assert since(t0, None, fallback=1234) == 1234
    assert since(t0, t0 + 0.5) == 500


def test_stream_result_defaults_are_zero():
    # The two socket marks are filled in by core.call, not here; the reader must leave
    # room for them rather than inventing values.
    out = StreamResult()
    assert out.req_sent_ms == 0 and out.ttfb_ms == 0
