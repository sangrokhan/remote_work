"""The per-turn record every arm produces, and its schema version.

One record per (provider, arm, turn, pass); the CSV is one row per record. Both
providers build their rows here and nowhere else, because a chart that puts a Gemini
arm next to an OpenAI arm is only honest if the two rows mean the same thing in every
column. `reasoning_tokens` is the provider-neutral name for what Gemini bills as
thought tokens and OpenAI as reasoning tokens: the adapters translate, and the record
never sees a vendor word.

Two fields exist purely so a run can be doubted:

  `schema_version`  a run written under an older layout can be told apart instead of
                    being silently charted against a newer one, column by column, with
                    whatever each name happened to mean at the time.
  `measure`         bytes from a streamed pass and bytes from a blocking pass are not
                    the same measurement. Averaging them is the mistake this column is
                    here to make visible.

`store_tail_ms` is the number the stored-interaction arms live or die on: the gap
between the last token of the answer and the server finally letting go. A streaming
client is done at `ttlt`; a blocking client waits out the whole tail. It is derived
here, once, rather than in each chart -- and floored at zero, because a mark pinned by
a failed call can otherwise put `ttlt` past `turn_end` and produce a negative wait,
which is not a thing that happens.
"""

from __future__ import annotations

SCHEMA_VERSION = 1


def _store_tail(exchange) -> int:
    """The wait after the last answer token, or 0 when nothing measured that moment.

    A blocking pass has no `ttlt`: it never saw a last token, only a finished body. So
    `turn_end - ttlt` is not the tail there, it is the whole call -- and a `bytes` run
    reported plain generateContent as carrying a 2.2-second store tail, which is a
    property of stored interactions and not a thing that arm does at all. A mark nobody
    took must read as absent, not as zero-and-therefore-subtractable.
    """
    if exchange.ttlt_ms <= 0:
        return 0
    return max(0, exchange.turn_end_ms - exchange.ttlt_ms)


def turn_record(provider: str, arm: str, phase: str, turn: int, question: str,
                measure: str, exchange, usage: dict, extra: dict | None = None) -> dict:
    """One row.

    `usage` is already provider-neutral -- the adapter has translated its vendor's usage
    block into `input_tokens` / `cached_tokens` / `output_tokens` / `reasoning_tokens` /
    `total_tokens` before it gets here. `phase` is `steady` for the turns that count, or
    the name of a prep phase (`cachegen`, `setup`) whose cost is real but is setup, not
    traffic, and must never be folded into an arm's totals.
    """
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0) or (input_tokens + output_tokens)

    record = {
        "schema_version": SCHEMA_VERSION,
        "provider": provider,
        "arm": arm,
        "phase": phase,
        "turn": turn,
        "measure": measure,

        "wire_sent": exchange.wire_sent,
        "wire_recv": exchange.wire_recv,
        "req_payload_bytes": exchange.req_payload_bytes,
        "resp_payload_bytes": exchange.resp_payload_bytes,

        "req_sent_ms": exchange.req_sent_ms,
        "ttfb_ms": exchange.ttfb_ms,
        "ttft_ms": exchange.ttft_ms,
        "ttlt_ms": exchange.ttlt_ms,
        "turn_end_ms": exchange.turn_end_ms,
        "store_tail_ms": _store_tail(exchange),

        "input_tokens": input_tokens,
        "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        "output_tokens": output_tokens,
        "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
        "total_tokens": total_tokens,

        "question": question,
        "response_text": exchange.text,
        "request_raw": exchange.request_json,
        "response_raw": exchange.response_json,
        "error": exchange.error,
    }
    if extra:
        record.update(extra)
    return record
