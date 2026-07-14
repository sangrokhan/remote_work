"""Read an SSE response and time it: TTFB, TTFT, TTLT, turn end.

A blocking call can report exactly one number -- when the whole thing came back --
and that number hides the thing the experiment is about. Measured on a Gemini
`/interactions` call with `store:true`: the answer's last token reaches the client at
~950 ms, and the stream then stays open ~1.8 s longer while the server persists the
interaction. A blocking client waits for all of it; a streaming client waits for the
token. Reporting only "elapsed" charges the stored-interaction arms for a write their
user never waits for; reporting only TTFT hides that write entirely. So both marks
are taken, and their difference (`store_tail_ms`) is a headline number, not a detail.

Five marks bracket one turn:

  req_sent_ms   the request's last byte went out          (from the socket, core.wire)
  ttfb_ms       the response's first byte came back       (from the socket, core.wire)
  ttft_ms       first event carrying ANSWER text
  ttlt_ms       last event carrying answer text
  turn_end_ms   the stream closed

`text_of` is the caller's, and it must return the answer and nothing else. Reasoning
text -- a Gemini `thought` part, an OpenAI reasoning summary -- is not the answer: a
turn that "thinks" for 400 ms before speaking has a real TTFT of 400 ms, and letting
a thought delta start the clock would report it as ~0 and make a reasoning model look
faster than a plain one. That is backwards, and it is the exact number a reader would
use to pick a model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class StreamResult:
    """One streamed call: what came back, and when."""

    status: int = 0
    req_sent_ms: int = 0        # filled in by core.call, from the wire marks
    ttfb_ms: int = 0            # filled in by core.call, from the wire marks
    ttft_ms: int = 0
    ttlt_ms: int = 0
    turn_end_ms: int = 0
    text: str = ""
    events: list = field(default_factory=list)   # parsed `data:` payloads, in order
    raw: str = ""                                # the SSE body, for the audit trail
    error: str = ""


def since(t0: float, mark, fallback: int = 0) -> int:
    """Milliseconds from the request going out to a monotonic mark on the socket.

    `mark` is None when the socket never got that far -- a connection that failed, a
    pool entry that was closed under us. Fall back rather than report a 0, which a
    reader would take as "instant" when it means "never".
    """
    return int((mark - t0) * 1000) if mark else fallback


def _iter_data(resp):
    """Yield (parsed_json, raw_line) for each SSE `data:` line.

    `[DONE]` sentinels and unparseable lines are skipped as events but still count as
    bytes -- the socket counter has already seen them, which is as it should be: they
    crossed the wire.
    """
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            yield json.loads(payload), line
        except ValueError:
            continue


def read_stream(resp, text_of, t0: float) -> StreamResult:
    """Consume `resp` as SSE, timing the first and last event that carries answer text.

    `text_of(event) -> str` pulls the answer text out of one event -- the only thing
    that differs between providers. `t0` is the monotonic reading taken just before the
    request went out, so every mark includes the request itself and the marks from the
    socket and the marks from the stream sit on one clock.
    """
    out = StreamResult(status=resp.status_code)
    lines, first, last = [], None, None
    for event, line in _iter_data(resp):
        now = (time.monotonic() - t0) * 1000
        out.events.append(event)
        lines.append(line)
        chunk = text_of(event)
        if chunk:
            if first is None:
                first = now
            last = now
            out.text += chunk
    out.turn_end_ms = int((time.monotonic() - t0) * 1000)
    # No answer text at all -- an empty completion, a refusal, an error stream. The turn
    # still ended, and it ended when the stream closed. Pinning the two answer marks to
    # turn_end says "never"; leaving them at 0 would say "instantly", which is the one
    # reading that is certainly false.
    out.ttft_ms = int(first if first is not None else out.turn_end_ms)
    out.ttlt_ms = int(last if last is not None else out.turn_end_ms)
    out.raw = "\n".join(lines)
    return out
