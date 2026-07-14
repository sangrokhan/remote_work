"""Read an SSE response and time it: TTFT, TTLT, turn end.

A blocking call can only ever report one number -- when the whole thing came back --
and that number hides the thing this experiment cares about. Measured 2026-07-14 on
`/interactions` with `store:true`: the answer's last token reaches the client at
~950 ms, and the stream then stays open ~1.8 s more while the server persists the
interaction. A blocking client waits for all of it; a streaming client waits for the
token. Reporting only "elapsed" would charge the interaction arms for a write their
user never waits for -- and reporting only TTFT would hide the write entirely.

So every arm streams, and every turn carries three timings:

  ttft_ms      request sent -> first event carrying answer text
  ttlt_ms      request sent -> last event carrying answer text
  turn_end_ms  request sent -> stream closed (server done, connection released)

For the stateless arms turn_end_ms lands on ttlt_ms (nothing happens after the last
token). For the stored interaction arms it does not, and the gap is the write.

Two wire vocabularies, one reader:

  generateContent (`:streamGenerateContent?alt=sse`)
      data: {"candidates":[{"content":{"parts":[{"text":"Paris"}],"role":"model"}}],
             "usageMetadata":{...}}
      The thoughtSignature arrives in its own later chunk, as a part whose text is
      empty. Both parts are kept: they are the model turn a client echoes back.

  interactions (`stream:true`)
      data: {"event_type":"step.start","index":0,"step":{"type":"thought"}}
      data: {"event_type":"step.delta","index":0,"delta":{"signature":"..."}}
      data: {"event_type":"step.delta","index":1,"delta":{"text":"Paris","type":"text"}}
      data: {"event_type":"interaction.completed","interaction":{"usage":{...}}}
      The completed event does NOT carry the steps (measured) -- they exist only as
      the deltas that streamed past, so the steps a client-side history echoes have
      to be rebuilt here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field


@dataclass
class StreamResult:
    """One streamed call: what came back, and when."""
    status: int = 0
    ttft_ms: int = 0
    ttlt_ms: int = 0
    turn_end_ms: int = 0
    text: str = ""
    events: list = field(default_factory=list)   # parsed `data:` payloads, in order
    raw: str = ""                                # the SSE body, for the audit trail
    error: str = ""


def _iter_data(resp):
    """Yield (elapsed_seconds_marker_is_caller_side, parsed_json) for each `data:`
    line. `[DONE]` and unparseable lines are skipped, but they still count as bytes
    the caller already read."""
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
    that differs between the two endpoints. `t0` is the monotonic clock reading taken
    just before the request went out, so the timings include the request itself.
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
    # No text at all (an empty answer, or an error stream): the turn still ended, and
    # a zero TTFT would read as "instant" rather than "never". Pin both to the end.
    out.ttft_ms = int(first if first is not None else out.turn_end_ms)
    out.ttlt_ms = int(last if last is not None else out.turn_end_ms)
    out.raw = "\n".join(lines)
    return out


# --- generateContent -------------------------------------------------------

def gen_text(event: dict) -> str:
    """Answer text in one generateContent chunk. A part flagged `thought` is the
    model's reasoning summary, not its answer -- it must not start the TTFT clock."""
    out = []
    for cand in event.get("candidates") or []:
        for part in (cand.get("content") or {}).get("parts") or []:
            if part.get("thought"):
                continue
            out.append(part.get("text") or "")
    return "".join(out)


def gen_response(events: list) -> dict:
    """Rebuild the non-streamed response body from the chunks.

    The parts are kept as they arrived -- the empty-text part carrying the
    thoughtSignature included -- because that list *is* the model turn a
    client-side history echoes back. usageMetadata is cumulative; the last one wins.
    """
    parts, usage, role = [], {}, "model"
    for ev in events:
        for cand in ev.get("candidates") or []:
            content = cand.get("content") or {}
            role = content.get("role") or role
            parts.extend(content.get("parts") or [])
        if ev.get("usageMetadata"):
            usage = ev["usageMetadata"]
    return {"candidates": [{"content": {"role": role, "parts": parts}}],
            "usageMetadata": usage}


# --- interactions ----------------------------------------------------------

def interaction_text(event: dict) -> str:
    if event.get("event_type") != "step.delta":
        return ""
    return (event.get("delta") or {}).get("text") or ""


def interaction_response(events: list) -> dict:
    """Rebuild the non-streamed interaction body from the events.

    Steps are reassembled by `index`: `step.start` declares the type, `step.delta`
    appends a signature (thought) or a text block (model_output). The completed event
    carries the id and usage but not the steps, so this is the only place they exist.
    """
    steps: dict[int, dict] = {}
    order: list[int] = []
    iid, usage, status = "", {}, ""
    for ev in events:
        kind = ev.get("event_type")
        idx = ev.get("index")
        if kind == "step.start":
            steps[idx] = {"type": (ev.get("step") or {}).get("type", "")}
            order.append(idx)
        elif kind == "step.delta":
            step = steps.setdefault(idx, {"type": ""})
            delta = ev.get("delta") or {}
            if delta.get("signature"):
                step["signature"] = step.get("signature", "") + delta["signature"]
            if delta.get("text") is not None:
                content = step.setdefault("content", [{"type": "text", "text": ""}])
                content[0]["text"] += delta.get("text") or ""
        elif kind in ("interaction.created", "interaction.completed"):
            it = ev.get("interaction") or {}
            iid = it.get("id") or iid
            usage = it.get("usage") or usage
            status = it.get("status") or status
    return {"id": iid, "status": status, "usage": usage,
            "steps": [steps[i] for i in order if i in steps]}
