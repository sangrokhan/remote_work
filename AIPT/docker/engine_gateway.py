#!/usr/bin/env python3
"""docker/engine_gateway.py -- L7 HTTP reverse-proxy sidecar for the
local-llm container ("engine Gateway", DESIGN.md 4.5 B4 / 4.7 -- distinct
from the Network Gateway container in aipt/gateway/, which does pure L3/L4
`tc netem` shaping in front of this one).

Composed topology this sidecar completes:

    web -> Network Gateway (L3/L4, tc netem delay)
         -> engine Gateway (L7, THIS FILE)
         -> llama-server (upstream, 127.0.0.1:40080, same container)

Unlike ``aipt/backends/local_llm/gateway.py``'s ``Gateway`` class (an
in-process hook layer a *caller* goes through -- no socket listener, see
that module's docstring), this is the real network listener that class's
docstring anticipated ("a later iteration that actually wants to expose
this to a child process ... can wrap this class in a real listener").
It binds its own port (``ENGINE_GATEWAY_PORT``, default 40079) and forwards
every request to the real engine on ``ENGINE_GATEWAY_UPSTREAM_HOST:PORT``
(default 127.0.0.1:40080, i.e. llama-server in the same container/network
namespace -- same pattern as docker/idle_reset_admin.py's sidecar).

Streaming vs caching decision (2026-09 Slack design discussion, see AGENTS
notes): the request body's ``"stream"`` field is read ONCE, before the
upstream call, and that single decision is carried through the entire
request/response lifecycle -- it is never re-derived from response chunk
contents (an individual SSE chunk is itself a complete, self-contained
JSON object, so "does this look like complete JSON" is NOT a safe signal
for "is this the whole response" -- see module-level NOTE below):

  * ``stream`` is falsy (or body isn't JSON)  -> **cacheable path**: the
    full request body is already in hand (it arrived as one HTTP message),
    the full response is buffered from upstream, then
    ``on_cacheable_request``/``on_cacheable_response`` hooks run around
    it. THIS PASS lands the hook points wired up but empty (no-op passthrough)
    -- the actual cache key/store/TTL logic is intentionally left for a
    follow-up change (scope decided in chat: "실제 네트워크 리스너로 승격 +
    통과 골격만, 캐싱 로직은 다음 단계").
  * ``stream`` is truthy -> **passthrough path**: hooks are never called.
    Bytes are relayed to the client using Python's own HTTP client
    (``http.client``), which already understands ``Transfer-Encoding:
    chunked`` framing (the actual HTTP-level end-of-message signal --
    *not* SSE's ``data: [DONE]`` convention, which is an
    application-level courtesy some engines send but not a protocol
    guarantee) and relays each engine chunk to the client as soon as it
    arrives, without buffering the whole reply in memory.

Why the branch decision is made once, from the request, and never
revisited mid-stream: a TCP connection carries exactly one request/response
at a time (HTTP/1.1 keep-alive still serializes messages on one
connection -- see http.client's own chunked-decoding, which already
tracks message boundaries for us), so there is no ambiguity about which
bytes belong to which reply; the risk this design avoids is a naive
"looks like complete JSON -> try to cache it" check firing on every
individual SSE chunk mid-stream, which is exactly the failure mode
flagged in the design discussion.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# aipt/ is importable here because docker/Dockerfile.local_llm copies a
# minimal aipt/{__init__,core/__init__,core/idle_reset,core/cache_protocol}
# .py slice to /app/aipt/ (this project deliberately does not reimplement
# inference, so the full aipt package with its web/requests dependencies
# has no reason to live in this image -- see docker/idle_reset_admin.py's
# identical import fix-up for the same reason). Try both /app (the
# container layout) and the repo root two levels up (running this file
# directly out of a checkout for local smoke-testing) so this module works
# in either context.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.dirname(_HERE)):
    if os.path.isdir(os.path.join(_candidate, "aipt")):
        sys.path.insert(0, _candidate)
        break

from aipt.core import cache_protocol  # noqa: E402

LISTEN_HOST = os.environ.get("ENGINE_GATEWAY_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("ENGINE_GATEWAY_PORT", "40079"))
UPSTREAM_HOST = os.environ.get("ENGINE_GATEWAY_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("ENGINE_GATEWAY_UPSTREAM_PORT", "40080"))
UPSTREAM_TIMEOUT = float(os.environ.get("ENGINE_GATEWAY_UPSTREAM_TIMEOUT", "120"))
CACHE_THRESHOLD_BYTES = int(
    os.environ.get("ENGINE_GATEWAY_CACHE_THRESHOLD_BYTES", "")
    or cache_protocol.DEFAULT_THRESHOLD_BYTES
)

# Headers that must never be blindly copied from one hop to the next --
# they describe THIS connection/message framing, not the payload, and
# stale/duplicate values here corrupt the proxied message.
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
    # Content-Length is recomputed per hop below (buffered path) or
    # dropped in favour of chunked framing (streaming path).
    "content-length",
}


# --------------------------------------------------------------------------
# Cache hook points -- deliberately empty in this pass. A follow-up change
# fills these in; this pass only guarantees WHEN they fire (see module
# docstring): never on the streaming path, always on the non-streaming path,
# after the full request/response body is available.
# --------------------------------------------------------------------------

def on_cacheable_request(method: str, path: str, headers: dict, body: bytes) -> "tuple[int, dict, bytes] | None":
    """Runs after the full (non-streaming) request body has arrived, before
    the upstream call. Return ``(status, headers_dict, body_bytes)`` to
    short-circuit with a cached response instead of calling upstream, or
    ``None`` to continue to upstream unchanged. No-op today."""
    if os.environ.get("ENGINE_GATEWAY_DEBUG_HOOKS"):
        print(f"[engine_gateway] on_cacheable_request fired: {method} {path}")
    return None  # type: tuple[int, dict, bytes] | None


def on_cacheable_response(method: str, path: str, req_body: bytes,
                           status: int, resp_headers: dict, resp_body: bytes) -> None:
    """Runs after the full (non-streaming) upstream response has been read,
    just before it is written back to the client -- the place a cache
    write would happen. No-op today (return value ignored)."""
    if os.environ.get("ENGINE_GATEWAY_DEBUG_HOOKS"):
        print(f"[engine_gateway] on_cacheable_response fired: {method} {path} status={status}")
    return None


def _is_stream_request(headers, body: bytes) -> bool:
    """The single, request-time decision this whole file hinges on (see
    module docstring). Body's JSON ``"stream"`` field is authoritative;
    an ``Accept: text/event-stream`` header is honoured too, in case a
    caller signals streaming that way without a matching body field."""
    accept = (headers.get("Accept") or "").lower()
    if "text/event-stream" in accept:
        return True
    ctype = (headers.get("Content-Type") or "").lower()
    if "application/json" not in ctype or not body:
        return False
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return False
    return bool(isinstance(parsed, dict) and parsed.get("stream"))


def _decode_cache_body(body: bytes, cache: "cache_protocol.SessionCache") -> "bytes | None":
    """Parses ``body`` as JSON, restores any ``$aipt_cache_map``-listed
    leaves via :func:`cache_protocol.decode_body`, and re-serializes.
    Returns ``None`` (meaning: forward ``body`` unchanged) when the body is
    not JSON at all -- the caching header being set does not mean every
    request necessarily carries a JSON body worth touching. Re-raises
    :class:`cache_protocol.CacheMiss` so the caller can turn it into a 409.

    Uses ``CACHE_THRESHOLD_BYTES`` (module-level, from
    ``ENGINE_GATEWAY_CACHE_THRESHOLD_BYTES``) for the symmetric-learning
    pass so this side's threshold matches the client's -- a mismatch here
    means a leaf the client considers a dedup candidate (and later sends
    hashed) never got learned server-side on its first plaintext
    appearance, guaranteeing a spurious 409 on every subsequent turn. Both
    sides default to ``cache_protocol.DEFAULT_THRESHOLD_BYTES`` (200) so
    this only matters if either side overrides its threshold.
    """
    if not body:
        return None
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    decoded = cache_protocol.decode_body(parsed, cache, CACHE_THRESHOLD_BYTES)
    return json.dumps(decoded).encode("utf-8")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # silence default access log
        pass

    def setup(self) -> None:
        super().setup()
        # Session-scoped cache store (docs/engine_gateway_caching_seed.md
        # §6/§8.6): BaseHTTPRequestHandler is instantiated once per
        # accepted TCP connection and handle() loops over every keep-alive
        # request on it, so a cache built here in setup() (which runs once,
        # before that loop) lives exactly as long as the connection does --
        # matching the design's "session = the TCP connection" decision
        # with no extra bookkeeping needed.
        self._cache = cache_protocol.SessionCache()
        if os.environ.get("ENGINE_GATEWAY_DEBUG_HOOKS"):
            print(f"[engine_gateway] NEW CONNECTION setup(), handler id={id(self)}, cache id={id(self._cache)}")

    # -- helpers ----------------------------------------------------------

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length else b""

    def _forward_headers(self) -> dict:
        return {
            k: v for k, v in self.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "host"
        }

    def _open_upstream(self) -> http.client.HTTPConnection:
        conn = http.client.HTTPConnection(
            UPSTREAM_HOST, UPSTREAM_PORT, timeout=UPSTREAM_TIMEOUT,
        )
        return conn

    def _write_response(self, status: int, headers: dict, body: bytes) -> None:
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() in _HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _relay_streaming(self, method: str, path: str, headers: dict, body: bytes) -> None:
        """Passthrough path: cache hooks are never invoked. Relays upstream
        bytes to the client as they arrive, using HTTP chunked framing on
        our own side too (we don't know the total length up front)."""
        conn = self._open_upstream()
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            self._write_response(
                502, {"Content-Type": "application/json"},
                json.dumps({"error": f"upstream_unreachable: {exc}"}).encode(),
            )
            conn.close()
            return

        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in _HOP_BY_HOP:
                continue
            self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n" % len(chunk))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-stream -- nothing more to do
        finally:
            conn.close()

    def _relay_cacheable(self, method: str, path: str, headers: dict, body: bytes,
                          cache_on: bool) -> None:
        """Non-streaming path: full request already in hand, so the cache
        hooks run around a single buffered upstream call.

        ``cache_on`` gates the request-body dedup protocol
        (docs/engine_gateway_caching_seed.md) separately from
        ``on_cacheable_request``/``on_cacheable_response`` (a different,
        currently no-op, response-side hook pair -- see module docstring).
        When ``cache_on`` is False this behaves exactly as before this
        feature existed.
        """
        if cache_on:
            try:
                decoded = _decode_cache_body(body, self._cache)
            except cache_protocol.CacheMiss as exc:
                resp_body = json.dumps(
                    {"error": "cache_miss", "missing_paths": exc.missing_paths}
                ).encode()
                self._write_response(
                    409, {"Content-Type": "application/json"}, resp_body,
                )
                return
            if decoded is not None:
                body = decoded

        hit = on_cacheable_request(method, path, headers, body)
        if hit is not None:
            status, resp_headers, resp_body = hit
            self._write_response(status, resp_headers, resp_body)
            return

        conn = self._open_upstream()
        try:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            resp_body = resp.read()
        except (OSError, http.client.HTTPException) as exc:
            self._write_response(
                502, {"Content-Type": "application/json"},
                json.dumps({"error": f"upstream_unreachable: {exc}"}).encode(),
            )
            return
        finally:
            conn.close()

        resp_headers = dict(resp.getheaders())
        on_cacheable_response(method, path, body, resp.status, resp_headers, resp_body)
        self._write_response(resp.status, resp_headers, resp_body)

    def _handle(self, method: str) -> None:
        body = self._read_body()
        headers = self._forward_headers()
        cache_on = (
            headers.get(cache_protocol.CACHE_HEADER, "").strip().lower()
            == cache_protocol.CACHE_HEADER_VALUE
        )
        if _is_stream_request(self.headers, body):
            self._relay_streaming(method, self.path, headers, body)
        else:
            self._relay_cacheable(method, self.path, headers, body, cache_on)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PUT(self):
        self._handle("PUT")

    def do_DELETE(self):
        self._handle("DELETE")


def serve() -> None:
    try:
        server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), _Handler)
    except OSError as exc:
        print(f"[engine_gateway] could not bind {LISTEN_HOST}:{LISTEN_PORT}: {exc}", file=sys.stderr)
        raise
    print(
        f"[engine_gateway] listening on {LISTEN_HOST}:{LISTEN_PORT}, "
        f"forwarding to {UPSTREAM_HOST}:{UPSTREAM_PORT}"
    )
    server.serve_forever()


if __name__ == "__main__":
    serve()
