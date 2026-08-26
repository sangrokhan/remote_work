"""aipt.gateway -- the Network Gateway container (DESIGN.md 4.7, B9).

A separate container/process sitting between the client (measurement code)
and ``MockBackend``/``LocalLLMBackend``, injecting pure L3/L4 traffic
characteristics (delay/jitter/loss/reorder/duplicate) via ``tc qdisc
netem``. ``PublicAIBackend`` never goes through this -- it already crosses
the real internet, which supplies those characteristics on its own
(DESIGN.md 4.7's opening paragraph).

**Not** the "engine gateway" in ``aipt.backends.local_llm.gateway`` -- that
is an application-level (L7) HTTP proxy layer for a *different* concern
(request/response experiment hooks). This package does no HTTP parsing at
all; it only shells out to ``tc`` against a network interface. See that
module's docstring for the full disambiguation DESIGN.md 4.8 asks for.

Submodules:
  * ``profiles`` -- named netem parameter presets (clean/broadband/3g/
    satellite/lossy/custom) plus env-var parsing for the container's
    startup preset (``GATEWAY_*``, with ``CLIENT_NETEM_DELAY_MS``/
    ``SERVER_NETEM_DELAY_MS`` kept as deprecated aliases).
  * ``netem_control`` -- the control loop itself: turns a ``Profile`` into
    ``tc`` commands and runs them, following the ``aipt.core.offload``/
    ``aipt.core.capture`` "no exception, report {ok, reason}" pattern when
    the process lacks NET_ADMIN (the common case outside a real container).
  * ``app`` -- the standalone FastAPI mini-app (``GET /health``,
    ``GET /gateway/profile``, ``POST /gateway/profile``) meant to run as
    its own process/container (``docker/Dockerfile.gateway``), independent
    of ``aipt/web``.
"""

from __future__ import annotations

from aipt.gateway import netem_control, profiles

__all__ = ["profiles", "netem_control"]
