"""A tiny in-process stand-in for the OpenAI REST API.

Lets the tests exercise the real request-building and byte-counting paths without
a key, quota, or network. It records every request body it saw, so a test can
assert what each arm actually put on the wire.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_REPLY = "ack"
# The fake stalls this long before the first token, so TTFT and TTLT are
# distinguishable in a test instead of both landing on ~0 ms.
FIRST_TOKEN_DELAY = 0.12
TOKEN_GAP = 0.02


def _chars(items) -> int:
    """Crude token proxy source: the characters of the message contents."""
    total = 0
    for it in items:
        c = it.get("content", "")
        total += len(c) if isinstance(c, str) else len(json.dumps(c))
    return total


class FakeOpenAI:
    def __init__(self):
        self.requests: list[dict] = []   # {"path", "body", "content_length"}
        self._conv_seq = 0
        self._resp_seq = 0
        self._conversations: dict[str, int] = {}   # conv_id -> chars held server-side
        self._server = None
        self._thread = None

    # --- server lifecycle -------------------------------------------------
    def start(self) -> str:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                body = json.loads(raw) if raw else {}
                outer.requests.append(
                    {"path": self.path, "body": body, "content_length": length}
                )
                if body.get("stream"):
                    self._stream(body)
                    return
                payload = json.dumps(outer._respond(self.path, body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _stream(self, body):
                """SSE, with a deliberate gap before the first token so a test can
                tell TTFT apart from TTLT instead of both rounding to zero.

                Connection: close, because an HTTP/1.1 response with neither a
                Content-Length nor chunked framing has no other way to say where
                the body ends — the client would block on the socket forever.
                """
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                for delay, chunk in outer._stream_events(self.path, body):
                    if delay:
                        time.sleep(delay)
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    # --- canned responses -------------------------------------------------
    def _respond(self, path: str, body: dict) -> dict:
        if path.endswith("/conversations"):
            self._conv_seq += 1
            conv_id = f"conv_{self._conv_seq}"
            # the server now holds the seeded items, and will bill for them on
            # every subsequent turn — same as the real thing
            self._conversations[conv_id] = _chars(body.get("items", []))
            return {"id": conv_id, "object": "conversation"}

        self._resp_seq += 1
        uploaded = body.get("input") or body.get("messages") or []
        chars = _chars(uploaded)

        conv_id = body.get("conversation")
        if conv_id:
            # Real behaviour, and the whole point of the experiment: the client
            # uploaded only the new message, but OpenAI bills every prior input
            # token in the chain anyway. So input_tokens must reflect the FULL
            # server-side history, not just what came up the wire.
            self._conversations[conv_id] += chars
            chars = self._conversations[conv_id]
            # server appends the reply to the conversation too
            self._conversations[conv_id] += len(_REPLY)

        in_tokens = max(chars // 4, 1)

        if path.endswith("/chat/completions"):
            return {
                "id": f"chatcmpl_{self._resp_seq}",
                "choices": [{"message": {"role": "assistant", "content": _REPLY}}],
                "usage": {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }

        return {
            "id": f"resp_{self._resp_seq}",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": _REPLY}],
                }
            ],
            "usage": {
                "input_tokens": in_tokens,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    def _stream_events(self, path: str, body: dict):
        """(delay_before, sse_chunk) pairs, mirroring the real event sequences.

        Chat: a role chunk with no content, then content deltas, then a
        finish_reason chunk, then a usage-only chunk (choices: []), then [DONE].
        The usage chunk arrives AFTER finish_reason, which is why TTLT must be
        timed off finish_reason and not off usage.

        Responses: response.created / output_text.delta xN / response.completed,
        with usage on response.completed.
        """
        full = self._respond(path, body)
        usage = full["usage"]
        tokens = [_REPLY[i:i + 1] for i in range(len(_REPLY))] or ["ack"]

        if path.endswith("/chat/completions"):
            yield 0, _sse_data({"choices": [{"delta": {"role": "assistant", "content": ""},
                                             "finish_reason": None}], "usage": None})
            for i, t in enumerate(tokens):
                yield (FIRST_TOKEN_DELAY if i == 0 else TOKEN_GAP), _sse_data(
                    {"choices": [{"delta": {"content": t}, "finish_reason": None}],
                     "usage": None})
            yield 0, _sse_data({"choices": [{"delta": {}, "finish_reason": "stop"}],
                                "usage": None})
            yield 0, _sse_data({"choices": [], "usage": usage})
            yield 0, "data: [DONE]\n\n"
            return

        rid = full["id"]
        yield 0, _sse_event("response.created", {"response": {"id": rid}})
        yield 0, _sse_event("response.output_item.added", {"output_index": 0})
        for i, t in enumerate(tokens):
            yield (FIRST_TOKEN_DELAY if i == 0 else TOKEN_GAP), _sse_event(
                "response.output_text.delta", {"delta": t})
        yield 0, _sse_event("response.output_text.done", {"text": _REPLY})
        yield 0, _sse_event("response.completed", {"response": {**full, "usage": usage}})
        yield 0, "data: [DONE]\n\n"

    def bodies_for(self, suffix: str) -> list[dict]:
        return [r["body"] for r in self.requests if r["path"].endswith(suffix)]


def _sse_data(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _sse_event(event: str, obj: dict) -> str:
    return f"event: {event}\ndata: {json.dumps({'type': event, **obj})}\n\n"
