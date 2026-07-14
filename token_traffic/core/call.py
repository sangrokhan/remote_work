"""One turn on the wire, in one or two passes.

Two of the measurements this suite exists to make want opposite things from the same
call, and no single request can serve both:

  Bytes want a blocking call. OpenAI's streamed deltas carry `include_obfuscation`
  padding, which inflates the SSE frames by an amount that has nothing to do with the
  conversation -- and the byte count is the whole point of the comparison.

  Latency wants a streamed call. Time-to-first-token cannot be read off a blocking
  response at all, and on a stored Gemini interaction the tail after the last token is
  ~1.8 s of server-side persistence that a streaming client never waits for. One
  "elapsed" number would hide which half of the turn a user actually feels.

So `measure` says what a turn pays for. `bytes` sends one blocking request; `latency`
sends one streamed request; `both` sends two and merges them, which doubles the API
bill and is therefore never a default. The pass a number came from is recorded, per
row, because bytes from a streamed pass and bytes from a blocking pass are different
quantities and averaging them together would produce a number that describes nothing.

Two things the caller supplies, because only the provider knows them:

  `text_of(event) -> str`   the answer text in one streamed event -- and, on a blocking
                            pass, in the whole response body, which is handed to it as
                            if it were a single event.
  `rebuild(events) -> dict` the body a blocking call would have returned, rebuilt from
                            the streamed events. Not a convenience: the Gemini
                            interactions endpoint streams the model's steps and its
                            completed event does not carry them, so the steps a
                            client-side history has to echo exist only as deltas that
                            went past.

`send` never raises. A turn that failed still has to produce a record -- a run with one
broken arm is still a run -- and its marks are pinned to the moment it ended, because a
zero mark reads as "instant" rather than "never".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from . import wire
from .streaming import read_stream, since

BYTES = "bytes"
LATENCY = "latency"
BOTH = "both"


@dataclass
class Exchange:
    """What one turn cost and how long each part of it took."""

    status: int = 0
    error: str = ""
    wire_sent: int = 0
    wire_recv: int = 0
    req_payload_bytes: int = 0
    resp_payload_bytes: int = 0
    req_sent_ms: int = 0
    ttfb_ms: int = 0
    ttft_ms: int = 0
    ttlt_ms: int = 0
    turn_end_ms: int = 0
    elapsed_ms: int = 0
    text: str = ""
    response: dict = field(default_factory=dict)   # the body a blocking call returns
    request_json: str = ""
    response_json: str = ""


def _pin(ex: Exchange, elapsed: int) -> None:
    """Pin every mark that never happened to when the turn ended.

    A failed call has no TTFT, and 0 is the wrong way to say so: it charts as the
    fastest turn in the run. The honest statement is "not before the end".
    """
    ex.elapsed_ms = elapsed
    ex.turn_end_ms = elapsed
    ex.ttfb_ms = ex.ttfb_ms or elapsed
    ex.ttft_ms = elapsed
    ex.ttlt_ms = elapsed
    ex.req_sent_ms = ex.req_sent_ms or elapsed


def _answer_of(text_of, body: dict) -> str:
    """The answer in a blocking response body.

    A blocking body has the same shape as the streamed events it would have been
    assembled from, near enough that the provider's own `text_of` can read it -- so it
    is handed the body as if it were one event. A provider whose two shapes really do
    differ makes `text_of` accept both; nothing else in core knows how to find an
    answer, and core must not start guessing.
    """
    try:
        return text_of(body) or ""
    except Exception:
        return ""


def _blocking(url: str, headers: dict, body: dict, text_of, timeout: int) -> Exchange:
    """One blocking POST. The bytes and the tokens are real; of the five marks only the
    two the socket can vouch for -- when the request left, when the turn ended -- are
    set. The rest stay 0, and `measure` on the record is what tells a reader they are
    absent rather than instantaneous."""
    payload = json.dumps(body)
    ex = Exchange(request_json=payload,
                  req_payload_bytes=len(payload.encode("utf-8")))
    t0 = time.monotonic()
    try:
        with wire.wire_counter() as w:
            resp = wire.session().post(url, data=payload, headers=headers,
                                       timeout=timeout)
            raw = resp.content
    except Exception as exc:
        ex.error = f"request_failed: {exc}"
        _pin(ex, int((time.monotonic() - t0) * 1000))
        return ex

    elapsed = int((time.monotonic() - t0) * 1000)
    ex.status = resp.status_code
    ex.wire_sent, ex.wire_recv = w.sent, w.recv
    ex.resp_payload_bytes = len(raw)
    ex.elapsed_ms = elapsed
    ex.turn_end_ms = elapsed
    ex.req_sent_ms = since(t0, w.last_send_at, elapsed)
    ex.response_json = resp.text

    if resp.status_code != 200:
        ex.error = f"http_{resp.status_code}: {resp.text[:200]}"
        _pin(ex, elapsed)
        return ex
    try:
        ex.response = resp.json()
    except ValueError as exc:
        ex.error = f"parse_failed: {exc}"
        _pin(ex, elapsed)
        return ex
    ex.text = _answer_of(text_of, ex.response)
    return ex


def _streamed(url: str, headers: dict, body: dict, text_of, rebuild,
              timeout: int) -> Exchange:
    """One streamed POST. The five marks are real. The bytes are the SSE framing --
    recorded, because they did cross the wire, but they are not the same quantity a
    blocking pass measures and the record's `measure` field is what says so."""
    payload = json.dumps(body)
    ex = Exchange(request_json=payload,
                  req_payload_bytes=len(payload.encode("utf-8")))
    t0 = time.monotonic()
    stream = None
    err_body = ""
    try:
        with wire.wire_counter() as w:
            with wire.session().post(url, data=payload, headers=headers,
                                     timeout=timeout, stream=True) as resp:
                status = resp.status_code
                if status != 200:
                    err_body = resp.text
                else:
                    # Read inside both the response and the counter: on a streamed
                    # response the bytes only cross the socket while it is consumed.
                    stream = read_stream(resp, text_of, t0)
    except Exception as exc:
        ex.error = f"request_failed: {exc}"
        _pin(ex, int((time.monotonic() - t0) * 1000))
        return ex

    elapsed = int((time.monotonic() - t0) * 1000)
    ex.status = status
    ex.wire_sent, ex.wire_recv = w.sent, w.recv
    ex.elapsed_ms = elapsed
    ex.req_sent_ms = since(t0, w.last_send_at, elapsed)

    if stream is None:
        ex.resp_payload_bytes = len(err_body.encode("utf-8"))
        ex.response_json = err_body
        ex.error = f"http_{status}: {err_body[:200]}"
        ex.ttfb_ms = since(t0, w.first_recv_at, elapsed)
        _pin(ex, elapsed)
        return ex

    ex.resp_payload_bytes = len(stream.raw.encode("utf-8"))
    ex.ttfb_ms = since(t0, w.first_recv_at, stream.ttft_ms)
    ex.ttft_ms = stream.ttft_ms
    ex.ttlt_ms = stream.ttlt_ms
    ex.turn_end_ms = stream.turn_end_ms
    ex.text = stream.text
    if rebuild is not None:
        try:
            ex.response = rebuild(stream.events)
            ex.response_json = json.dumps(ex.response)
        except Exception as exc:
            ex.error = f"rebuild_failed: {exc}"
            ex.response_json = stream.raw
    else:
        # Without a rebuild there is no blocking-shaped body to record, so the evidence
        # is the SSE body itself, exactly as it arrived.
        ex.response_json = stream.raw
    return ex


def send(url: str, headers: dict, body: dict, *, measure: str, text_of,
         stream_body: dict | None = None, stream_url: str | None = None,
         rebuild=None, timeout: int = 180) -> Exchange:
    """Run one turn and report what it cost.

    `body` is the blocking request. `stream_body` is the same request with whatever the
    provider needs to make it stream (`stream: true`), and `stream_url` is where that
    goes when streaming lives at a different endpoint -- Gemini's is
    `:streamGenerateContent`, not `:generateContent`, so the two passes of a `both` turn
    do not even share a URL.

    Never raises: a failed turn comes back with `error` set and its marks pinned.
    """
    if measure == BYTES:
        return _blocking(url, headers, body, text_of, timeout)

    if measure == LATENCY:
        return _streamed(stream_url or url, headers, stream_body or body,
                         text_of, rebuild, timeout)

    if measure != BOTH:
        return Exchange(error=f"bad_measure: {measure!r}")

    # Blocking first: it is the pass that produces the response body the caller may have
    # to echo back into the next turn, and the streamed pass is only there for its clock.
    blocking = _blocking(url, headers, body, text_of, timeout)
    streamed = _streamed(stream_url or url, headers, stream_body or body,
                         text_of, rebuild, timeout)
    return _merge(blocking, streamed)


def _merge(blocking: Exchange, streamed: Exchange) -> Exchange:
    """Bytes and body from the blocking pass, marks from the streamed one.

    Each half is taken from the only pass entitled to report it. The streamed pass's
    byte counts are dropped on the floor rather than added: they measure SSE framing of
    the same turn, and a sum of the two would be a number no client ever pays.
    """
    merged = Exchange(
        status=blocking.status or streamed.status,
        error=blocking.error or streamed.error,
        wire_sent=blocking.wire_sent,
        wire_recv=blocking.wire_recv,
        req_payload_bytes=blocking.req_payload_bytes,
        resp_payload_bytes=blocking.resp_payload_bytes,
        req_sent_ms=streamed.req_sent_ms,
        ttfb_ms=streamed.ttfb_ms,
        ttft_ms=streamed.ttft_ms,
        ttlt_ms=streamed.ttlt_ms,
        turn_end_ms=streamed.turn_end_ms,
        elapsed_ms=streamed.elapsed_ms,
        text=blocking.text or streamed.text,
        response=blocking.response or streamed.response,
        request_json=blocking.request_json,
        response_json=blocking.response_json,
    )
    return merged
