#!/usr/bin/env python3
"""docker/idle_reset_admin.py -- tiny sidecar admin server for the
local-llm container's idle-reset toggle (DESIGN.md idle-reset TTFT
experiment).

``local-llm`` wraps the upstream ``ghcr.io/ggml-org/llama-server`` binary
(a real llama.cpp server this project never reimplements -- see
docker/entrypoint_local_llm.py's module docstring), so unlike
``mock-server`` (this project's own stdlib HTTP server, which can import
``aipt.core.idle_reset`` directly, see aipt/backends/mock/server.py) there
is no in-process hook to attach a ``/admin/idle-reset`` route to.

Instead this is a second, tiny stdlib HTTP server bound to
``IDLE_RESET_ADMIN_PORT`` (default 40081, next to llama-server's own 40080)
that runs in a background thread inside the *same container* -- same
network namespace, so toggling ``net.ipv4.tcp_slow_start_after_idle`` here
affects exactly the socket llama-server's responses go out on, the same as
the mock-server case. ``entrypoint_local_llm.py`` starts this thread before
``exec``-ing into ``llama-server`` (which then becomes PID 1, preserving
the "signals/healthcheck behave exactly like the upstream image" property
that module already documents).

Endpoints (same shape as mock-server's, so the web proxy route in
aipt/web/routes_gateway.py can treat both identically):
  GET  /admin/idle-reset            -> {"ok", "enabled", "reason"}
  POST /admin/idle-reset?enabled=0|1 -> same + write_ok/write_reason
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# aipt/ is importable here because docker/Dockerfile.local_llm copies a
# minimal aipt/{__init__,core/__init__,core/idle_reset}.py slice to
# /app/aipt/ (this project deliberately does not reimplement inference, so
# the full aipt package with its web/requests dependencies has no reason
# to live in this image -- see that Dockerfile's comment). Try both /app
# (the container layout) and the repo root two levels up (running this
# file directly out of a checkout, e.g. `python3 docker/idle_reset_admin.py`
# for local smoke-testing) so this module works in either context.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.dirname(_HERE)):
    if os.path.isdir(os.path.join(_candidate, "aipt")):
        sys.path.insert(0, _candidate)
        break

from aipt.core import idle_reset  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access log
        pass

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urllib.parse.urlparse(self.path).path == "/admin/idle-reset":
            self._json(200, idle_reset.status())
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/admin/idle-reset":
            self._json(404, {"error": "not found"})
            return
        query = urllib.parse.parse_qs(parsed.query)
        raw = query.get("enabled", [None])[0]
        if raw not in ("0", "1"):
            self._json(400, {"error": "enabled must be 0 or 1"})
            return
        ok, reason = idle_reset.write(raw == "1")
        body = idle_reset.status()
        body["write_ok"] = ok
        body["write_reason"] = reason
        self._json(200 if ok else 500, body)


def start_background(port: int | None = None) -> threading.Thread:
    """Starts the admin server on a daemon thread and returns it. Never
    raises -- a bind failure (port already in use, no permission) is
    printed and the thread simply never starts serving; the main
    llama-server process must not be blocked by this sidecar's health."""
    port = port or int(os.environ.get("IDLE_RESET_ADMIN_PORT", "40081"))
    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    except OSError as exc:
        print(f"[idle_reset_admin] could not bind :{port}: {exc} -- idle-reset toggle unavailable")
        return threading.Thread(target=lambda: None)
    thread = threading.Thread(target=server.serve_forever, daemon=True,
                               name="idle-reset-admin")
    thread.start()
    print(f"[idle_reset_admin] listening on :{port}")
    return thread


if __name__ == "__main__":
    # Standalone smoke-test entrypoint (not used in the container image,
    # which imports start_background() from entrypoint_local_llm.py).
    t = start_background()
    t.join()
