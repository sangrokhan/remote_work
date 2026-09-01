"""aipt.backends.mock.server: HTTP/1.1 inference-mock server (stdlib only).

Migrated from ``tcp_congestion/tcp_congestion/server.py`` (DESIGN.md 5, A3),
extended per DESIGN.md 5 B1 to also serve real scenario-record answer text, not
just N-byte dummy padding.

Endpoints (unchanged from the original, still stdlib-only, still
HTTP/1.1 keep-alive so a client can reuse one TCP connection across turns
-- the idle-gap cwnd experiment needs that):

  GET  /ping                       -> {"ts": <float>}
  GET  /health                     -> {"status": "ok"}
  GET|POST /inference-mock?delay=<ms>&response_bytes=<n>&turn=<i>
                                    -> JSON body (see below)
  GET  /admin/idle-reset           -> {"ok", "enabled", "reason"}
                                       (aipt.core.idle_reset.status())
  POST /admin/idle-reset?enabled=0|1
                                    -> same shape, after attempting the
                                       write (aipt.core.idle_reset.write()).
                                       This toggles
                                       net.ipv4.tcp_slow_start_after_idle in
                                       *this container's* netns -- the
                                       responding side of the connection,
                                       which is where the LLM-turn response
                                       (the much larger of the two
                                       directions) actually re-enters slow
                                       start. Needs CAP_NET_ADMIN (already
                                       granted to this service in
                                       docker-compose.yml for the L3 routing
                                       fix-up); a write attempt without it
                                       reports ok=false with a reason
                                       instead of a raw 500 (aipt.core.
                                       idle_reset's never-raises contract).

``/inference-mock`` behaviour:

  * No ``record`` bound to the server (``Server(record=None)``, the
    default): exactly the original dummy-byte behaviour --
    ``{"tokens": 100, "ts": ..., ["prompt_bytes": <n> on POST]}``, padded
    with a "pad" filler field to ``response_bytes`` bytes on the wire via
    ``_pad_json_to_size`` (unchanged; this is the "pure byte-size sweep"
    option DESIGN.md 5 keeps around).
  * A ``record`` bound and a valid ``turn=<i>`` query param: the response
    additionally carries ``"answer": record.turns[i].answer`` -- the real
    (or replay-placeholder, see ``aipt.backends.mock.replay``) answer text
    for that turn -- and, unless the caller passed an explicit
    ``response_bytes``, the padding target defaults to that answer's own
    byte length, so the wire response size matches the fixture's answer
    size without the caller having to compute and pass it separately.
    ``turn`` out of range, non-numeric, or no record bound: falls back to
    the plain dummy response (never a 5xx -- a malformed replay request
    should degrade to dummy bytes, not break the run).

``delay`` still simulates inference latency by sleeping before responding
-- this is the *only* place latency is controlled by (server side, or the
client side's own ``inference_delay_ms`` sleep in conversation.py); real
captured latency is never replayed (DESIGN.md 4.5 "Mock 재생 충실도").
"""

from __future__ import annotations

import json
import socketserver
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler
from typing import TYPE_CHECKING

from aipt.core import idle_reset

if TYPE_CHECKING:
    from aipt.backends.mock.records import ScenarioRecord


def _pad_json_to_size(base: dict, target_bytes: int) -> bytes:
    """Serialize *base* as JSON, padded with a "pad" field of literal
    filler characters so the encoded body is exactly *target_bytes* long.

    Unchanged from the original ``tcp_congestion`` implementation -- see
    the historical note there: without this, a caller's requested response
    size never actually reached the wire, and only the *simulated* history
    size used it. Padding the real response body to size closes that gap.

    target_bytes <= the base (no-pad) JSON's encoded size returns the base
    JSON unpadded -- this floor is unavoidable and reported implicitly by
    the caller comparing actual vs. requested size.
    """
    encoded = json.dumps(base).encode()
    if target_bytes <= 0 or target_bytes <= len(encoded):
        return encoded
    padded = dict(base)
    padded["pad"] = ""
    deficit = target_bytes - len(json.dumps(padded).encode())
    padded["pad"] = "x" * max(deficit, 0)
    return json.dumps(padded).encode()


def _record_answer(record: "ScenarioRecord | None", query: dict) -> tuple[str | None, int]:
    """(answer_text_or_None, byte_len_to_pad_to_if_answer_found).

    Returns (None, 0) whenever there is no record, no/invalid ``turn``
    param, or the index is out of range -- the caller falls back to plain
    dummy behaviour in every one of those cases.
    """
    if record is None:
        return None, 0
    raw_turn = query.get("turn", [None])[0]
    if raw_turn is None:
        return None, 0
    try:
        idx = int(raw_turn)
    except ValueError:
        return None, 0
    if idx < 0 or idx >= len(record.turns):
        return None, 0
    answer = record.turns[idx].answer
    return answer, len(answer.encode())


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access log
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/ping":
            self._json(200, {"ts": time.time()})
        elif path == "/health":
            self._json(200, {"status": "ok"})
        elif path == "/inference-mock":
            self._handle_inference(query, prompt_bytes=None)
        elif path == "/admin/idle-reset":
            self._json(200, idle_reset.status())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/inference-mock":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            self._handle_inference(query, prompt_bytes=len(body))
        elif path == "/admin/idle-reset":
            raw = query.get("enabled", [None])[0]
            if raw not in ("0", "1"):
                self._json(400, {"error": "enabled must be 0 or 1"})
                return
            ok, reason = idle_reset.write(raw == "1")
            body = idle_reset.status()
            body["write_ok"] = ok
            body["write_reason"] = reason
            self._json(200 if ok else 500, body)
        else:
            self._json(404, {"error": "not found"})

    def _handle_inference(self, query: dict, prompt_bytes: int | None) -> None:
        delay_ms = int(query.get("delay", ["0"])[0])
        response_bytes = int(query.get("response_bytes", ["0"])[0])
        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

        record = getattr(self.server, "record", None)
        answer, answer_bytes = _record_answer(record, query)

        body: dict = {"tokens": 100, "ts": time.time()}
        if prompt_bytes is not None:
            body["prompt_bytes"] = prompt_bytes
        target_bytes = response_bytes
        if answer is not None:
            body["answer"] = answer
            if not response_bytes:
                target_bytes = answer_bytes

        self._json(200, body, target_bytes=target_bytes)

    def _json(self, code: int, body: dict, target_bytes: int = 0) -> None:
        data = (_pad_json_to_size(body, target_bytes)
                if target_bytes else json.dumps(body).encode())
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(data)


class Server(socketserver.ThreadingTCPServer):
    """HTTP/1.1 keep-alive mock inference server.

    ``record`` (optional): an ``aipt.backends.mock.records.ScenarioRecord``
    (Q&A-loaded, byte-size-swept, or replay-built) whose answers
    ``/inference-mock?turn=<i>`` will serve. ``None`` (default) preserves
    the exact original tcp_congestion behaviour: pure dummy-byte
    responses, no record lookup at all.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, host: str = "0.0.0.0", port: int = 8888,
                 record: "ScenarioRecord | None" = None):
        super().__init__((host, port), _Handler)
        self.host, self.port = self.server_address
        self.record = record
