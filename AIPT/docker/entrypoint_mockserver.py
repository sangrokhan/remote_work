#!/usr/bin/env python3
"""docker/entrypoint_mockserver.py -- container entrypoint for the AIPT
mock-server image (docker/Dockerfile.mockserver, DESIGN.md 4.7 B10).

``aipt.backends.mock.server.Server`` (see that module) is a plain
``socketserver.ThreadingTCPServer`` subclass with no ``if __name__ ==
"__main__":`` block of its own -- it's designed to be started in-process by
``aipt.backends.mock.conversation.MockBackend`` for the default (non-Docker)
run path. This script is the standalone-process equivalent for the
Docker topology: it just binds ``Server(host, port)`` and calls
``serve_forever()`` so the container can run this backend out-of-process,
reachable over the network from the ``gateway``/``web`` services (DESIGN.md
4.7's "MockBackend must only be reachable via Gateway" topology decision).

Deliberately NOT applying tc netem/offload here -- DESIGN.md 4.7 moved that
responsibility to the dedicated ``gateway`` container (``aipt/gateway/``,
``docker/Dockerfile.gateway``); this container only serves fixed/dummy
inference-mock responses.

Env vars:
  MOCK_HOST (default "0.0.0.0")
  MOCK_PORT (default "8888")
"""
import os
import sys

sys.path.insert(0, "/app")

from aipt.backends.mock.server import Server  # noqa: E402

host = os.environ.get("MOCK_HOST", "0.0.0.0")
port = int(os.environ.get("MOCK_PORT", "8888"))
print(f"[mock-server] listening on {host}:{port}")

srv = Server(host=host, port=port)
try:
    srv.serve_forever()
except KeyboardInterrupt:
    pass
finally:
    srv.shutdown()
