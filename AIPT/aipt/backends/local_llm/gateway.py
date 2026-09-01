"""aipt.backends.local_llm.gateway -- the in-repo "engine gateway" proxy
layer between a caller and a standard serving engine (DESIGN.md 4.5 B4,
4.7's "미해결 세부사항 #3", 4.8).

**Not** the Network Gateway container (DESIGN.md 4.7, `aipt/gateway/`,
B9) -- that is a separate, later component doing pure L3/L4 traffic
shaping (`tc netem`) in its own container, sitting *in front of* both
``MockBackend`` and this backend's connection to its engine. This module is
the *other* thing DESIGN.md 4.8 warns reads as confusingly similar: an
application-level HTTP layer that owns the experiment surface for
request/response mutation (new headers, new params, and later a transport
switch) on the path between the client code and the engine's
OpenAI-compatible API. DESIGN.md 4.7 §3's conclusion, restated in code
terms: the two compose as ``client -> Network Gateway (L3/L4, separate
component, out of scope here) -> this engine Gateway (L7) ->
engine_adapter -> serving engine``.

This lands only the extensible skeleton (DESIGN.md B4's own scoping: "실제
신기능 실험은 이번 범위 밖 - 확장 가능한 골격만"): a request/response hook
point (``on_request``/``on_response`` callables, registered rather than
subclassed so a caller can add an experiment without touching this file)
and the ``transport`` slot reflected onto the request as a header, which is
the only concrete "HTTP feature experiment" this module actually performs
today -- everything else is future hook-driven work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from aipt.backends.base import DEFAULT_TRANSPORT, Transport
from aipt.backends.local_llm.engine_adapter import EngineAdapter
from aipt.core import cache_protocol, wire

#: Header the gateway sets from the ``Backend.transport`` slot on every
#: request it forwards, so an engine/proxy hop downstream (or a pcap
#: reader) can tell which transport a request was nominally sent under --
#: today always "http1" in practice, but the header exists so wiring a
#: later "http3" value through requires no protocol change here (DESIGN.md
#: 4.5 B5: the slot is a plain field, not a branch).
TRANSPORT_HEADER = "X-AIPT-Transport"

#: A request hook receives the outgoing request dict (mutates it in place
#: or returns a replacement) before it is sent; a response hook receives
#: the parsed response dict (or the raw exchange metadata on error) after
#: it comes back. Both are plain callables, not a base class to subclass --
#: matching this file's own "hook point, not a framework" scope.
RequestHook = Callable[[dict], "dict | None"]
ResponseHook = Callable[[dict], "dict | None"]


@dataclass
class GatewayResult:
    """What one proxied call produced -- the pieces a Backend's
    ``send_turn`` needs to build an ``aipt.backends.record.Exchange``
    without this module needing to know about that dataclass."""

    status: int = 0
    error: str = ""
    request_body: dict = field(default_factory=dict)
    response_body: dict = field(default_factory=dict)
    text: str = ""
    wire_sent: int = 0
    wire_recv: int = 0
    req_payload_bytes: int = 0
    resp_payload_bytes: int = 0
    req_sent_ms: int = 0
    ttfb_ms: int = 0
    elapsed_ms: int = 0
    # Request-body leaf-hash dedup measurement (docs/
    # engine_gateway_caching_seed.md): how many bytes THIS turn's actual
    # wire payload is smaller than the same request would have been with
    # caching off, computed from the SAME single call (no separate
    # baseline run needed) -- see Gateway.send()'s cache_bytes_saved
    # computation. 0 when cache_enabled is False.
    cache_bytes_saved: int = 0


class Gateway:
    """Sits between a caller (``LocalLLMBackend``) and one
    :class:`~aipt.backends.local_llm.engine_adapter.EngineAdapter`,
    forwarding chat/completions requests while giving experiments a place
    to hook in.

    Deliberately not itself a network proxy process (no socket listener) --
    "gateway" here means an in-process layer a caller goes through, mirroring
    how ``aipt.backends.public_ai``'s adapters already funnel every call
    through ``aipt.core.wire.session()`` rather than calling ``requests``
    directly. A later iteration that actually wants to expose this to a
    child process (an experimental HTTP/3 proxy in front of the engine, per
    DESIGN.md B5) can wrap this class in a real listener without changing
    the hook contract callers already depend on.
    """

    def __init__(
        self,
        adapter: EngineAdapter,
        *,
        transport: Transport = DEFAULT_TRANSPORT,
        timeout: int | None = None,
        cache_enabled: bool = False,
        cache_threshold_bytes: int = cache_protocol.DEFAULT_THRESHOLD_BYTES,
    ) -> None:
        self.adapter = adapter
        self.transport = transport
        self.timeout = timeout or adapter.timeout
        self._request_hooks: list[RequestHook] = []
        self._response_hooks: list[ResponseHook] = []
        # Request-body leaf-hash dedup (docs/engine_gateway_caching_seed.md).
        # Session = this Gateway instance's lifetime, which in turn tracks
        # the underlying wire.session() keep-alive connection's lifetime
        # (LocalLLMBackend.connect() builds a fresh Gateway per run) -- see
        # the Seed doc's section 8.6 for why this is the right scope.
        self.cache_enabled = cache_enabled
        self.cache_threshold_bytes = cache_threshold_bytes
        self._cache = cache_protocol.SessionCache() if cache_enabled else None

    # -- hook registration ---------------------------------------------

    def on_request(self, hook: RequestHook) -> Callable[[], None]:
        """Register a hook run on every outgoing request just before it is
        sent. Returns an unsubscribe callable, mirroring
        ``aipt.core.wire.watch_connections``'s pattern so a caller
        (a test, a scoped experiment) can clean up after itself."""
        self._request_hooks.append(hook)

        def unsubscribe() -> None:
            try:
                self._request_hooks.remove(hook)
            except ValueError:
                pass

        return unsubscribe

    def on_response(self, hook: ResponseHook) -> Callable[[], None]:
        """Register a hook run on every response just after it is parsed."""
        self._response_hooks.append(hook)

        def unsubscribe() -> None:
            try:
                self._response_hooks.remove(hook)
            except ValueError:
                pass

        return unsubscribe

    def _run_request_hooks(self, req: dict) -> dict:
        for hook in list(self._request_hooks):
            try:
                out = hook(req)
                if out is not None:
                    req = out
            except Exception:
                # A broken experiment hook must not take the whole run
                # down -- the same "best-effort instrumentation" contract
                # aipt.core.wire.watch_connections and cwnd.announce hold.
                pass
        return req

    def _run_response_hooks(self, resp: dict) -> dict:
        for hook in list(self._response_hooks):
            try:
                out = hook(resp)
                if out is not None:
                    resp = out
            except Exception:
                pass
        return resp

    # -- the actual proxied call -----------------------------------------

    def _build_request(self, messages: list[dict], **kwargs) -> dict:
        body = self.adapter.build_body(messages, **kwargs)
        headers = self.adapter.headers()
        headers[TRANSPORT_HEADER] = self.transport
        return {
            "url": self.adapter.chat_completions_url(),
            "headers": headers,
            "body": body,
        }

    def send(self, messages: list[dict], **kwargs) -> GatewayResult:
        """POST one chat/completions call through the gateway.

        Not streamed (``stream=False`` is forced regardless of ``kwargs``)
        -- ``LocalLLMBackend`` uses this for the blocking byte-accurate
        path; a streamed variant would reuse ``aipt.core.streaming`` the
        same way ``aipt.backends.public_ai._call`` does, and is left for
        whichever future change actually needs TTFT off a local engine
        (out of this change's scope, matching the task's "얇은 클라이언트"
        framing).
        """
        import json as _json

        kwargs = dict(kwargs)
        kwargs["stream"] = False
        req = self._build_request(messages, **kwargs)
        req = self._run_request_hooks(req)

        uncached_payload_bytes = 0
        if self.cache_enabled:
            assert self._cache is not None
            # Snapshot what this exact request would have cost WITHOUT
            # caching -- the body at this point is still the real,
            # unhashed dict (encode_body runs next), so this is the
            # honest "baseline" for this one turn, computed from the same
            # call rather than a separate uncached run. Cheap: one more
            # json.dumps of a dict already fully built.
            uncached_payload_bytes = len(_json.dumps(req["body"]).encode("utf-8"))
            req["headers"][cache_protocol.CACHE_HEADER] = cache_protocol.CACHE_HEADER_VALUE
            req["body"] = cache_protocol.encode_body(
                req["body"], self._cache, self.cache_threshold_bytes,
            )

        result = self._send_once(req)

        # Cache-miss recovery (docs/engine_gateway_caching_seed.md §8.2):
        # the server's session-side store lost one or more hashes this
        # connection's client-side store still has. Restore exactly those
        # paths to their plaintext (from this Gateway's own cache -- it is
        # authoritative regardless of what the server forgot) and resend
        # ONCE. A second miss on the retry is treated as a real error
        # (falls through with whatever _send_once returned) rather than
        # looping -- a server that keeps missing after being handed the
        # plaintext back has a bug this layer cannot paper over.
        if (
            self.cache_enabled
            and result.status == 409
            and cache_protocol.CACHE_MAP_FIELD in req["body"]
        ):
            missing = _parse_cache_miss_paths(result)
            if missing:
                req["body"] = _revert_missing_paths(req["body"], missing, self._cache)
                result = self._send_once(req)

        if self.cache_enabled and result.status and result.status != 409:
            # Only credit a saving on a request that actually went out
            # under the cached body (a 409 that never recovered leaves
            # req_payload_bytes reflecting the failed cached attempt, not
            # a real successful uncached-equivalent comparison).
            result.cache_bytes_saved = max(0, uncached_payload_bytes - result.req_payload_bytes)

        return result

    def _send_once(self, req: dict) -> GatewayResult:
        import json as _json

        payload = _json.dumps(req["body"])
        result = GatewayResult(request_body=req["body"])
        t0 = time.monotonic()
        try:
            with wire.wire_counter() as w:
                resp = wire.session().post(
                    req["url"], data=payload, headers=req["headers"],
                    timeout=self.timeout,
                )
                raw = resp.content
        except Exception as exc:
            result.error = f"request_failed: {exc}"
            result.elapsed_ms = int((time.monotonic() - t0) * 1000)
            result.req_sent_ms = result.elapsed_ms
            result.ttfb_ms = result.elapsed_ms
            return result

        elapsed = int((time.monotonic() - t0) * 1000)
        result.status = resp.status_code
        result.wire_sent, result.wire_recv = w.sent, w.recv
        result.req_payload_bytes = len(payload.encode("utf-8"))
        result.resp_payload_bytes = len(raw)
        result.elapsed_ms = elapsed
        result.req_sent_ms = _since(t0, w.last_send_at, elapsed)
        result.ttfb_ms = _since(t0, w.first_recv_at, elapsed)

        if resp.status_code != 200:
            result.error = f"http_{resp.status_code}: {resp.text[:200]}"
            result.response_body = _safe_json(resp)
            return result
        try:
            body = resp.json()
        except ValueError as exc:
            result.error = f"parse_failed: {exc}"
            return result

        body = self._run_response_hooks(body)
        result.response_body = body
        # text_of() expects a full OpenAI-compatible response/chunk dict
        # (it reads body["choices"][0] itself) -- see engine_adapter.py.
        result.text = self.adapter.text_of(body)
        return result


def _safe_json(resp) -> dict:
    try:
        return resp.json()
    except ValueError:
        return {}


def _parse_cache_miss_paths(result: GatewayResult) -> list[str]:
    body = result.response_body or {}
    if not isinstance(body, dict) or body.get("error") != "cache_miss":
        return []
    return list(body.get("missing_paths") or [])


def _revert_missing_paths(body: dict, missing_labels: list[str],
                           cache: "cache_protocol.SessionCache") -> dict:
    """Restore exactly the paths the server reported missing to their
    plaintext value (read back from this Gateway's own cache, which still
    has value<->hash for anything it has ever sent), and drop them from
    ``$aipt_cache_map`` so the resend carries them as plain leaves again."""
    import copy

    new_body = copy.deepcopy(body)
    cache_map = dict(new_body.get(cache_protocol.CACHE_MAP_FIELD) or {})
    reverse_map = {v: k for k, v in cache_map.items()}
    for label in missing_labels:
        path = cache_protocol.parse_label(label)
        h = cache_protocol.get_at_path(new_body, path)
        original = cache.value_for(h)
        if original is None:
            # Nothing this Gateway can do -- it never actually held this
            # value itself, which should not happen (it is the one that
            # substituted the hash in the first place), but don't crash a
            # whole run over it: leave the hash in place, the server will
            # 409 again and _send_once's plain error path takes over.
            continue
        cache_protocol.set_at_path(new_body, path, original)
        map_key = reverse_map.get(label)
        if map_key is not None:
            cache_map.pop(map_key, None)
    if cache_map:
        new_body[cache_protocol.CACHE_MAP_FIELD] = cache_map
    else:
        new_body.pop(cache_protocol.CACHE_MAP_FIELD, None)
    return new_body


def _since(t0: float, mark, fallback: int) -> int:
    return int((mark - t0) * 1000) if mark else fallback


__all__ = ["Gateway", "GatewayResult", "TRANSPORT_HEADER", "RequestHook", "ResponseHook"]
