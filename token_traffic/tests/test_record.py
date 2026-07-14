"""One row, the same shape from every provider.

The record is the contract between the two adapters and every chart downstream. A
column that one provider fills and the other silently omits does not fail; it averages,
and the average is wrong in a way nothing in the pipeline can notice. So the field set
is asserted exactly, not loosely.

`store_tail_ms` is derived here rather than in the charts because it is the number the
stored-interaction arms are judged on, and a number computed in three places is a number
computed three ways.
"""

from __future__ import annotations

from core.call import Exchange
from core.record import SCHEMA_VERSION, turn_record

FIELDS = {
    "schema_version", "provider", "arm", "phase", "turn", "measure",
    "wire_sent", "wire_recv", "req_payload_bytes", "resp_payload_bytes",
    "req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms", "store_tail_ms",
    "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
    "question", "response_text", "request_raw", "response_raw", "error",
}

USAGE = {"input_tokens": 1200, "cached_tokens": 1024, "output_tokens": 64,
         "reasoning_tokens": 40, "total_tokens": 1304}


def _exchange(**kw) -> Exchange:
    base = dict(status=200, wire_sent=4096, wire_recv=2048,
                req_payload_bytes=3800, resp_payload_bytes=1900,
                req_sent_ms=30, ttfb_ms=210, ttft_ms=300, ttlt_ms=950,
                turn_end_ms=2750, elapsed_ms=2750, text="Paris.",
                request_json='{"q":1}', response_json='{"a":1}')
    base.update(kw)
    return Exchange(**base)


def _record(**kw) -> dict:
    args = dict(provider="gemini", arm="interaction", phase="steady", turn=3,
                question="What is the capital of France?", measure="both",
                exchange=_exchange(), usage=USAGE)
    args.update(kw)
    return turn_record(**args)


def test_the_record_carries_exactly_the_agreed_fields():
    rec = _record()
    assert set(rec) == FIELDS
    assert rec["schema_version"] == SCHEMA_VERSION
    assert rec["provider"] == "gemini" and rec["arm"] == "interaction"
    assert rec["phase"] == "steady" and rec["turn"] == 3
    assert rec["measure"] == "both"
    assert rec["error"] == ""
    # The audit trail: what was asked, what came back, and the raw bodies behind both.
    assert rec["question"].startswith("What is")
    assert rec["response_text"] == "Paris."
    assert rec["request_raw"] == '{"q":1}'
    assert rec["response_raw"] == '{"a":1}'


def test_store_tail_is_the_wait_after_the_last_token():
    rec = _record()
    # The answer finished at 950 ms; the server let go at 2750. A blocking client waits
    # out the difference and a streaming one does not -- which is the whole argument.
    assert rec["store_tail_ms"] == 1800


def test_store_tail_never_goes_negative():
    # A failed turn has its marks pinned to when it ended, and a pinned ttlt can land on
    # or past turn_end. A negative wait is not a thing that happens; report no tail.
    rec = _record(exchange=_exchange(ttlt_ms=2750, turn_end_ms=2750))
    assert rec["store_tail_ms"] == 0
    rec = _record(exchange=_exchange(ttlt_ms=3000, turn_end_ms=2750))
    assert rec["store_tail_ms"] == 0


def test_reasoning_tokens_are_provider_neutral():
    # Gemini bills them as thought tokens, OpenAI as reasoning tokens. The adapter does
    # the translation; the record only ever sees one name, or the two providers could
    # not be charted in one column.
    rec = _record()
    assert rec["reasoning_tokens"] == 40
    assert "thought_tokens" not in rec


def test_missing_usage_totals_are_derived_not_dropped():
    rec = _record(usage={"input_tokens": 10, "output_tokens": 5})
    assert rec["total_tokens"] == 15         # a provider that omits the total still adds up
    assert rec["cached_tokens"] == 0
    assert rec["reasoning_tokens"] == 0
    # And an empty usage block does not crash a run: a broken turn still produces a row.
    assert _record(usage={})["total_tokens"] == 0


def test_extra_fields_ride_along():
    rec = _record(extra={"cache_name": "cachedContents/abc", "mock": True})
    assert rec["cache_name"] == "cachedContents/abc"
    assert rec["mock"] is True
    assert FIELDS <= set(rec)


def test_a_failed_turn_still_produces_a_row():
    ex = _exchange(status=500, text="", response_json="", req_sent_ms=40,
                   ttfb_ms=40, ttft_ms=40, ttlt_ms=40, turn_end_ms=40)
    ex.error = "http_500: boom"
    rec = _record(exchange=ex)
    assert set(rec) == FIELDS
    assert rec["error"] == "http_500: boom"
    # Its marks are pinned, not zero -- a broken turn must not chart as the fastest one.
    assert rec["ttft_ms"] == 40
    assert rec["store_tail_ms"] == 0
