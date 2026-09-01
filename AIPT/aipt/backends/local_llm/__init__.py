"""aipt.backends.local_llm -- standard serving engine + gateway (DESIGN.md 4.5).

Real implementation (DESIGN.md 5, B4): a standard serving framework
(llama.cpp/vLLM, reached only as an OpenAI-compatible HTTP API --
``engine_adapter.py``) behind an in-repo application-level proxy
(``gateway.py``) that owns the transport/HTTP-feature experiment surface
today's ``transport`` slot (``aipt.backends.base``) reserves for a later
QUIC gateway.

``LocalLLMBackend`` is the ``aipt.backends.base.Backend`` protocol
implementation client code actually drives: ``connect`` builds one
``aipt.core.wire``-instrumented HTTP connection to the configured engine
(``LOCAL_LLM_ENGINE_URL``) via a ``gateway.Gateway``, ``send_turn`` sends
one turn's message history through it and returns an
``aipt.backends.record.Exchange``, and ``close`` tears the cwnd monitor
down. This mirrors ``MockBackend``'s connect/send_turn/close shape (one
``aipt.core.cwnd.Monitor`` for the connection's whole lifetime) rather than
``PublicAIBackend``'s per-arm-state-machine shape, because
``LocalLLMBackend`` has exactly one calling convention today (DESIGN.md
4.7's premise: this backend "traverses the Network Gateway to reach the
serving engine", so its connection -- unlike public_ai's real-internet
path -- is exactly the thing under measurement).

DESIGN.md 4.7's Network Gateway container (separate component, `tc netem`,
out of scope for this change) is assumed to already sit between this
backend and the engine when one is deployed; this backend does not know or
care whether it is there -- it just connects to whatever
``LOCAL_LLM_ENGINE_URL`` resolves to, the same way ``MockBackend`` connects
to whatever ``Server`` it started without knowing if a Network Gateway sits
in front of it.
"""

from __future__ import annotations

import os
import socket
import time

from aipt.backends.base import Transport
from aipt.backends.local_llm import engine_adapter as engine_adapter_mod
from aipt.backends.local_llm.engine_adapter import EngineAdapter
from aipt.backends.local_llm.gateway import Gateway
from aipt.backends.record import Exchange
from aipt.core import cwnd, cache_protocol
from aipt.core import wire

NAME = "local_llm"

#: One calling convention today: a plain multi-turn chat history sent as
#: one OpenAI-compatible /v1/chat/completions call per turn, growing the
#: message list each time (the local-engine equivalent of public_ai's
#: "stateless" arm -- a local engine has no server-side session/cache
#: concept standardized across llama.cpp/vLLM the way Gemini's cached
#: content / OpenAI's stored responses do, so there is nothing else to
#: arm-select on yet).
ARMS = ("chat",)
HEADLINE_ARMS = ("chat",)


def _host_port_from_url(url: str) -> tuple[str, int]:
    """(host, port) aipt.core.cwnd.Monitor needs to watch this engine's
    connection, parsed out of the engine's base URL without pulling in a
    full URL-parsing dependency."""
    rest = url.split("://", 1)[-1]
    hostport = rest.split("/", 1)[0]
    if ":" in hostport:
        host, port_s = hostport.rsplit(":", 1)
        try:
            return host, int(port_s)
        except ValueError:
            return host, 443 if url.startswith("https") else 80
    return hostport, 443 if url.startswith("https") else 80


class LocalLLMBackend:
    """``aipt.backends.base.Backend`` over a standard serving engine
    (llama.cpp/vLLM) reached through this package's ``Gateway`` proxy.

    ``engine_kind``/``engine_url``/``model`` default from the
    ``LOCAL_LLM_ENGINE_KIND``/``LOCAL_LLM_ENGINE_URL``/``LOCAL_LLM_MODEL``
    env vars (see ``engine_adapter.py``) when not given explicitly, mirroring
    how ``public_ai``'s adapters read their env vars at call time rather
    than import time.
    """

    NAME = NAME
    DEFAULT_MODEL = engine_adapter_mod.DEFAULT_MODEL
    ARMS = ARMS
    HEADLINE_ARMS = HEADLINE_ARMS

    def __init__(
        self,
        *,
        engine_url: str | None = None,
        engine_kind: str | None = None,
        model: str | None = None,
        transport: Transport = "http1",
        label: str = "local-llm",
        timeout: int = 120,
        cache_enabled: bool | None = None,
        cache_threshold_bytes: int | None = None,
    ) -> None:
        self._engine_url = engine_url or engine_adapter_mod.engine_url()
        self._engine_kind = engine_kind or engine_adapter_mod.engine_kind()
        self._model = model or engine_adapter_mod.default_model()
        self.transport = transport
        self.label = label
        self.timeout = timeout
        # Request-body leaf-hash dedup (docs/engine_gateway_caching_seed.md),
        # opt-in and defaulting from env so the web UI/API layer can toggle
        # it per run without a code change -- mirrors how engine_url/model
        # already default from env at call time (see docstring above).
        # LOCAL_LLM_CACHE_ENABLED: "1"/"true"/"yes" (case-insensitive) to
        # default on; unset/anything else defaults off.
        if cache_enabled is None:
            cache_enabled = os.environ.get(
                "LOCAL_LLM_CACHE_ENABLED", ""
            ).strip().lower() in ("1", "true", "yes")
        self.cache_enabled = cache_enabled
        if cache_threshold_bytes is None:
            cache_threshold_bytes = int(
                os.environ.get("LOCAL_LLM_CACHE_THRESHOLD_BYTES", "")
                or cache_protocol.DEFAULT_THRESHOLD_BYTES
            )
        self.cache_threshold_bytes = cache_threshold_bytes

        self._adapter: EngineAdapter | None = None
        self._gateway: Gateway | None = None
        self._system: str = ""
        self._messages: list[dict] = []
        self._monitor: cwnd.Monitor | None = None
        self._cwnd_result: dict = {}
        self._conn: socket.socket | None = None
        self._unsubscribe_watch = None

    def ready(self) -> tuple[bool, str]:
        """Whether this instance is configured to try talking to an engine.

        Reports against ``self._engine_url`` (this instance's actual
        configured target -- from the ``engine_url=`` constructor arg, or
        ``engine_adapter_mod.engine_url()``'s env/default fallback when not
        given), not a fresh env-var lookup -- otherwise a caller who
        explicitly picked a different ``engine_url=`` would get a
        ``ready()`` reason describing some other engine entirely. Bug
        caught when DEFAULT_ENGINE_URL moved off 8080 (previously
        the default and an explicit ``engine_url="...:8080"`` happened to
        collide, masking that this ignored the instance's own value).
        """
        if not self._engine_url:
            return False, "LOCAL_LLM_ENGINE_URL is empty"
        return True, f"targeting {self._engine_url} ({self._engine_kind})"

    def api_host(self) -> str:
        return self._engine_url

    def connect(self, arm: str, model: str, system: str) -> None:
        """Configure the adapter/gateway for this run and start watching
        the connection this run's requests will open.

        Unlike ``MockBackend`` (which owns/starts its own server socket),
        the socket here is opened lazily by ``aipt.core.wire``'s pooled
        session on the first request -- so the monitor is armed via
        ``aipt.core.wire.watch_connections`` (the same hook
        ``public_ai``'s adapters rely on implicitly through the shared
        session) rather than an explicit ``announce()`` call, and
        ``aipt.core.wire.reset_session()`` is called first so a run gets a
        fresh connection instead of reusing one from a previous run/test
        that would already be past its slow start.
        """
        if arm not in ARMS:
            raise ValueError(f"unknown local_llm arm: {arm!r} (known: {', '.join(ARMS)})")
        self._system = system or ""
        self._messages = [{"role": "system", "content": self._system}] if self._system else []

        self._adapter = EngineAdapter(
            base_url=self._engine_url,
            kind=self._engine_kind,
            model=model or self._model,
            timeout=self.timeout,
        )
        self._gateway = Gateway(
            self._adapter, transport=self.transport, timeout=self.timeout,
            cache_enabled=self.cache_enabled,
            cache_threshold_bytes=self.cache_threshold_bytes,
        )

        host, port = _host_port_from_url(self._adapter.base_url)
        wire.reset_session()
        self._monitor = cwnd.Monitor(self.label, host, port=port)
        self._monitor.__enter__()

        def _watch(sock: socket.socket) -> None:
            self._conn = sock
            if self._monitor is not None:
                self._monitor.announce(sock)

        self._unsubscribe_watch = wire.watch_connections(_watch)

    def send_turn(
        self, turn: int, question: str, measure: str, on_progress=None
    ) -> Exchange:
        if self._gateway is None:
            raise RuntimeError("send_turn called before connect()")
        from aipt.backends.base import progress
        progress(on_progress, backend=self.NAME, arm="chat", phase="steady",
                  turn=turn, turns=turn)

        self._messages.append({"role": "user", "content": question})
        t_start = time.monotonic()
        result = self._gateway.send(self._messages)
        turn_end_ms = int((time.monotonic() - t_start) * 1000)

        usage = engine_adapter_mod.EngineAdapter.usage_of(result.response_body)
        if result.text:
            self._messages.append({"role": "assistant", "content": result.text})

        return Exchange(
            wire_sent=result.wire_sent,
            wire_recv=result.wire_recv,
            req_payload_bytes=result.req_payload_bytes,
            resp_payload_bytes=result.resp_payload_bytes,
            req_sent_ms=result.req_sent_ms,
            ttfb_ms=result.ttfb_ms,
            # A blocking chat/completions call has no separate TTFT/TTLT
            # marks (matching public_ai's "bytes" pass, aipt.backends
            # .public_ai._call._blocking): both are pinned to when the
            # whole body finished, which is also turn_end here.
            ttft_ms=result.elapsed_ms or turn_end_ms,
            ttlt_ms=result.elapsed_ms or turn_end_ms,
            turn_end_ms=result.elapsed_ms or turn_end_ms,
            text=result.text,
            request_json=result.request_body,
            response_json=result.response_body,
            error=result.error or None,
            cache_bytes_saved=result.cache_bytes_saved,
        )

    def close(self) -> None:
        if self._unsubscribe_watch is not None:
            self._unsubscribe_watch()
            self._unsubscribe_watch = None
        if self._monitor is not None:
            self._monitor.stop()
            self._cwnd_result = self._monitor.result()
            self._monitor = None
        self._adapter = None
        self._gateway = None
        self._conn = None

    def cwnd_result(self) -> dict:
        """The continuous cwnd trace for this connection's lifetime, same
        contract as ``MockBackend.cwnd_result()``."""
        if self._monitor is not None:
            return self._monitor.result()
        return self._cwnd_result


__all__ = ["NAME", "ARMS", "HEADLINE_ARMS", "LocalLLMBackend"]
