"""Single generateContent call with real wire-byte counting.

Targets the **Gemini Developer API** (generativelanguage.googleapis.com) with an
API key. That is the only host serving plain-model Interactions, and every arm has
to sit on one host, one auth, one network path or the latency numbers compare
nothing -- so Vertex/ADC is gone from this project entirely.

Measures, per call:
  - wire_sent / wire_recv : bytes of the HTTP request/response as they cross the
    TLS stream — i.e. headers + content-encoded (often gzip) body, the real
    transferred size. This is post-decryption HTTP framing, NOT the raw TLS
    ciphertext; for true packet/ciphertext sizes use the optional pcap capture.
  - req_payload_bytes / resp_payload_bytes : decoded JSON body sizes (app layer)
  - prompt_tokens / resp_tokens / total_tokens : from response usageMetadata

No tcpdump / NET_ADMIN needed: we wrap the socket (send + makefile read paths) so
every byte of the HTTP exchange is tallied.

Mock mode (GEMINI_MOCK=1) returns synthetic data so the whole flow and charts
work locally without GCP creds or quota.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import requests
from urllib3.connection import HTTPSConnection, HTTPConnection

import streaming

# --- Developer API (generativelanguage) -----------------------------------
# The comparison runs entirely on the Developer API: it is the only host that
# serves plain-model Interactions, and keeping every arm on one host/auth/route
# is what makes the latency numbers comparable. Auth is an API key, not ADC.
# Env is read at call time (not import) so tests can monkeypatch without reload.

def api_host() -> str:
    """Host[:port]. GEMINI_API_HOST lets tests point at a local server."""
    return os.environ.get("GEMINI_API_HOST", "generativelanguage.googleapis.com")


def api_base() -> str:
    # Scheme is overridable so tests can drive a local (TLS-less) server.
    scheme = os.environ.get("GEMINI_API_SCHEME", "https")
    return f"{scheme}://{api_host()}/v1beta"


def api_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "")


def auth_headers() -> dict:
    return {"x-goog-api-key": api_key()}


def generate_url(model: str) -> str:
    return f"{api_base()}/models/{model}:generateContent"


def stream_generate_url(model: str) -> str:
    """Every arm streams, because TTFT cannot be measured any other way -- and
    without TTFT the interaction arms are charged for a write their user never waits
    for (see streaming.py)."""
    return f"{api_base()}/models/{model}:streamGenerateContent?alt=sse"


def cache_base_url() -> str:
    return f"{api_base()}/cachedContents"


def cache_create_body(model: str, contents: list, system_instruction: str,
                      ttl_seconds: int) -> dict:
    """CachedContent create body. The cache endpoint wants `models/{id}`, unlike
    generateContent's path which carries the bare id; systemInstruction is a
    Content object, not a bare string."""
    body: dict = {
        "model": f"models/{model}",
        "contents": contents,
        "ttl": f"{ttl_seconds}s",
    }
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    return body


ENDPOINT = f"{api_host()}:443"

# Rough public estimate (USD per token). Clearly an estimate; override via env.
PRICE_PER_TOKEN = float(os.environ.get("GEMINI_PRICE_PER_TOKEN", "0.0000001"))


@dataclass
class CallResult:
    mode: str
    turn: int
    prompt_tokens: int = 0
    resp_tokens: int = 0
    total_tokens: int = 0
    wire_sent: int = 0
    wire_recv: int = 0
    req_payload_bytes: int = 0
    resp_payload_bytes: int = 0
    cached_tokens: int = 0
    thought_tokens: int = 0
    elapsed_ms: int = 0          # request start -> response fully read
    # The three timings every streamed turn carries. ttft/ttlt bracket the answer;
    # turn_end_ms is when the server finally let go. On generateContent they coincide
    # -- nothing happens after the last token -- which is exactly what makes them
    # worth reporting next to the interaction arms, where they do not.
    req_sent_ms: int = 0         # -> request fully written to the socket (upload done)
    ttfb_ms: int = 0             # -> first response byte back (server started talking)
    ttft_ms: int = 0             # -> first event carrying answer text
    ttlt_ms: int = 0             # -> last event carrying answer text
    turn_end_ms: int = 0         # -> stream closed
    response_text: str = ""
    # Raw JSON bodies exactly as sent to / received from the server (no headers,
    # so no bearer token). Lets the log show the full wire payload per step.
    request_json: str = ""
    response_json: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# Module-global byte tally. Every counting socket and reader adds to it, so the
# count survives connection pooling: a keep-alive socket keeps feeding the tally
# whether or not connect() fired for this request. wire_counter() reads it by
# difference. The experiment loop is single-threaded, so one request is ever in
# flight and the difference is exact.
_wire_tally = {"sent": 0, "recv": 0}

# When the socket last wrote, and when it first read back, for the request in flight.
# The byte counts alone cannot say how long the upload took: a turn that ships a
# 90 KB history spends real time putting it on the wire before the model has seen a
# word of it, and that time belongs to the client, not to the model. Stamped on the
# socket, so it measures the write itself and not the library around it.
_wire_marks: dict = {"last_send": None, "first_recv": None}


class _WireDelta:
    """Bytes sent/received during one wire_counter() block, and when."""
    __slots__ = ("sent", "recv", "last_send_at", "first_recv_at")

    def __init__(self):
        self.sent = 0
        self.recv = 0
        self.last_send_at = None      # monotonic: request fully written
        self.first_recv_at = None     # monotonic: first response byte back


@contextmanager
def wire_counter():
    """Count HTTP bytes on the socket for the enclosed request(s), headers and
    content-encoding included, regardless of keep-alive reuse. Also stamp when the
    request finished going out and when the first byte came back.

    Yields a _WireDelta whose fields are populated when the block exits.
    """
    before_sent = _wire_tally["sent"]
    before_recv = _wire_tally["recv"]
    _wire_marks["last_send"] = None
    _wire_marks["first_recv"] = None
    delta = _WireDelta()
    try:
        yield delta
    finally:
        delta.sent = _wire_tally["sent"] - before_sent
        delta.recv = _wire_tally["recv"] - before_recv
        delta.last_send_at = _wire_marks["last_send"]
        delta.first_recv_at = _wire_marks["first_recv"]


class _CountingReader:
    """Wraps the file object returned by socket.makefile(), counting bytes read.

    http.client / urllib3 read the response through sock.makefile() rather than
    sock.recv(), so the read path must be counted here or wire_recv stays 0.
    """

    def __init__(self, fp, counter):
        self._fp = fp
        self._c = counter

    def _count(self, n: int) -> None:
        if n and _wire_marks["first_recv"] is None:
            _wire_marks["first_recv"] = time.monotonic()
        self._c.recv += n
        _wire_tally["recv"] += n

    def read(self, *a, **k):
        b = self._fp.read(*a, **k)
        self._count(len(b))
        return b

    def read1(self, *a, **k):
        b = self._fp.read1(*a, **k)
        self._count(len(b))
        return b

    def readline(self, *a, **k):
        b = self._fp.readline(*a, **k)
        self._count(len(b))
        return b

    def readinto(self, buf):
        n = self._fp.readinto(buf)
        self._count(n or 0)
        return n

    def __getattr__(self, name):
        return getattr(self._fp, name)


class _CountingSocket:
    """Wraps a socket, tallying every byte sent and received."""

    def __init__(self, sock):
        self._sock = sock
        self.sent = 0
        self.recv = 0

    def _mark_send(self) -> None:
        # Every write moves the mark, so when the request is done the mark sits on
        # its last byte -- which is exactly when the upload finished.
        _wire_marks["last_send"] = time.monotonic()

    def _mark_recv(self, n: int) -> None:
        if n and _wire_marks["first_recv"] is None:
            _wire_marks["first_recv"] = time.monotonic()

    def sendall(self, data, *args, **kwargs):
        self.sent += len(data)
        _wire_tally["sent"] += len(data)
        out = self._sock.sendall(data, *args, **kwargs)
        self._mark_send()
        return out

    def send(self, data, *args, **kwargs):
        n = self._sock.send(data, *args, **kwargs)
        self.sent += n
        _wire_tally["sent"] += n
        self._mark_send()
        return n

    def recv(self, bufsize, *args, **kwargs):
        chunk = self._sock.recv(bufsize, *args, **kwargs)
        self._mark_recv(len(chunk))
        self.recv += len(chunk)
        _wire_tally["recv"] += len(chunk)
        return chunk

    def recv_into(self, buf, *args, **kwargs):
        n = self._sock.recv_into(buf, *args, **kwargs)
        self._mark_recv(n or 0)
        self.recv += n
        _wire_tally["recv"] += n
        return n

    def makefile(self, mode="r", *args, **kwargs):
        fp = self._sock.makefile(mode, *args, **kwargs)
        # Only the readable binary path carries response bytes worth counting.
        if "b" in mode and "w" not in mode and "+" not in mode:
            return _CountingReader(fp, self)
        return fp

    def __getattr__(self, name):
        return getattr(self._sock, name)


# Active counter for the in-flight request (single-threaded experiment loop).
_active_counter: dict = {"counter": None}


class _CountingConnection:
    """Swaps in a counting socket once the connection is up.

    Mixed into both connection classes: HTTPS is what the API is called over, and
    plain HTTP exists so a local (TLS-less) test server is counted the same way.
    """

    def connect(self):
        super().connect()
        counter = _CountingSocket(self.sock)
        self.sock = counter
        _active_counter["counter"] = counter


class _CountingHTTPSConnection(_CountingConnection, HTTPSConnection):
    pass


class _CountingHTTPConnection(_CountingConnection, HTTPConnection):
    pass


def _build_session() -> requests.Session:
    """Session whose http(s) pools use the counting connection classes."""
    from requests.adapters import HTTPAdapter
    from urllib3.poolmanager import PoolManager
    from urllib3.connectionpool import HTTPSConnectionPool, HTTPConnectionPool

    class _CountingHTTPSPool(HTTPSConnectionPool):
        ConnectionCls = _CountingHTTPSConnection

    class _CountingHTTPPool(HTTPConnectionPool):
        ConnectionCls = _CountingHTTPConnection

    class _CountingPoolManager(PoolManager):
        def _new_pool(self, scheme, host, port, request_context=None):
            kw = self.connection_pool_kw.copy()
            kw.pop("scheme", None)
            if scheme == "https":
                return _CountingHTTPSPool(host, port, **kw)
            if scheme == "http":
                return _CountingHTTPPool(host, port, **kw)
            return super()._new_pool(scheme, host, port, request_context)

    class _CountingAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **kw):
            self.poolmanager = _CountingPoolManager(
                num_pools=connections, maxsize=maxsize, block=block, **kw
            )

    sess = requests.Session()
    adapter = _CountingAdapter()
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


_SESSION = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _build_session()
    return _SESSION


def reset_session() -> None:
    """Close pooled connections and drop the session so the next call opens a
    fresh TCP socket (new 3-way handshake).

    The global session pools TLS connections and reuses one socket across turns
    and stages, so per-stage pcaps miss the SYN/SYN-ACK/ACK and show "ACK for
    unseen segment" warnings. Calling this between capture stages guarantees each
    stage's pcap starts from a clean handshake and ends with the socket teardown.
    """
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()  # closes all pooled sockets (sends FIN)
        except Exception:
            pass
        _SESSION = None


def is_mock() -> bool:
    return os.environ.get("GEMINI_MOCK") == "1"


def ready() -> tuple[bool, str]:
    """Whether a real call can be made. Returns (ok, reason_if_not)."""
    if is_mock():
        return True, ""
    if not api_key():
        return False, "GEMINI_API_KEY not set (or run with GEMINI_MOCK=1)."
    return True, ""


def _text_tokens(contents: list) -> int:
    text_len = sum(len(p.get("text", "")) for c in contents for p in c.get("parts", []))
    return max(1, text_len // 4)


# Mock timings. Fixed, so a mock run's latency chart is stable; shaped like the real
# thing, so the arms that pay a tail can be told apart from the arms that do not.
# The upload mark scales with the payload, because that is the one part of the delay
# a bigger history really does buy: MOCK_UPLOAD_MS_PER_KB per KB on the wire.
MOCK_REQ_SENT_BASE_MS = 20
MOCK_UPLOAD_MS_PER_KB = 2
MOCK_TTFB_MS = 200
MOCK_TTFT_MS = 300
MOCK_TTLT_MS = 800


def _mock_req_sent_ms(req_bytes: int) -> int:
    return MOCK_REQ_SENT_BASE_MS + (req_bytes // 1024) * MOCK_UPLOAD_MS_PER_KB


def _mock_signature(turn: int) -> str:
    """Stand-in for the opaque base64 thought signature a real response carries.
    Roughly the real length: most of what echoing a model turn costs is this blob."""
    return f"MOCKSIG{turn:03d}" + ("A" * 60)


def _mock_call(mode: str, turn: int, contents: list, cached_tokens: int) -> CallResult:
    body = json.dumps({"contents": contents})
    req_bytes = len(body.encode("utf-8"))
    prompt_tokens = _text_tokens(contents)  # only what the client actually sends
    resp_tokens = 64
    # Deterministic synthetic answer referencing the last question (for scenarios).
    last_q = ""
    for c in reversed(contents):
        if c.get("role") == "user":
            last_q = "".join(p.get("text", "") for p in c.get("parts", []))[:40]
            break
    resp_text = f"(mock answer to: {last_q}) " + ("lorem ipsum " * 20)
    resp_bytes = len(resp_text.encode("utf-8")) + 120
    # Synthetic response body mirroring the real generateContent shape -- including
    # the thoughtSignature every Gemini 3 part comes back with, so a client-side
    # history that echoes the model's turn is exercised (and billed for its bytes)
    # in mock mode too.
    resp_json = json.dumps({
        "candidates": [{"content": {"role": "model",
                                    "parts": [{"text": resp_text,
                                               "thoughtSignature": _mock_signature(turn)}]}}],
        "usageMetadata": {"promptTokenCount": prompt_tokens,
                          "candidatesTokenCount": resp_tokens,
                          "cachedContentTokenCount": cached_tokens,
                          "totalTokenCount": prompt_tokens + resp_tokens + cached_tokens},
    })
    # Synthetic timings with the shape the real thing has: the answer streams, and on
    # generateContent nothing happens after its last token -- turn_end lands on ttlt.
    return CallResult(
        mode=mode, turn=turn,
        prompt_tokens=prompt_tokens, resp_tokens=resp_tokens,
        total_tokens=prompt_tokens + resp_tokens + cached_tokens,
        cached_tokens=cached_tokens, response_text=resp_text,
        wire_sent=req_bytes + 200, wire_recv=resp_bytes + 200,
        req_payload_bytes=req_bytes, resp_payload_bytes=resp_bytes,
        req_sent_ms=_mock_req_sent_ms(req_bytes), ttfb_ms=MOCK_TTFB_MS,
        ttft_ms=MOCK_TTFT_MS, ttlt_ms=MOCK_TTLT_MS,
        turn_end_ms=MOCK_TTLT_MS, elapsed_ms=MOCK_TTLT_MS,
        request_json=body, response_json=resp_json,
    )


def call_gemini(model: str, contents: list, mode: str, turn: int,
                cached_content: str | None = None,
                cached_tokens_hint: int = 0) -> CallResult:
    """One streamed generateContent call, optionally with a cachedContent ref.

    Never raises; errors land in .error. cached_tokens_hint is used only in mock
    mode to simulate the cached prefix size.
    """
    if is_mock():
        return _mock_call(mode, turn, contents, cached_tokens_hint if cached_content else 0)

    payload = {"contents": contents}
    if cached_content:
        payload["cachedContent"] = cached_content
    body = json.dumps(payload)
    req_payload_bytes = len(body.encode("utf-8"))

    headers = {"Content-Type": "application/json", **auth_headers()}
    t0 = time.monotonic()
    try:
        with wire_counter() as w:
            with _session().post(stream_generate_url(model), data=body,
                                 headers=headers, timeout=120, stream=True) as resp:
                if resp.status_code != 200:
                    err_body = resp.text
                    stream = None
                else:
                    # Read inside both the response and the counter: the bytes are
                    # only on the socket while the stream is open.
                    stream = streaming.read_stream(resp, streaming.gen_text, t0)
    except Exception as exc:
        return CallResult(mode=mode, turn=turn, req_payload_bytes=req_payload_bytes,
                          request_json=body, error=f"request_failed: {exc}",
                          elapsed_ms=int((time.monotonic() - t0) * 1000))

    elapsed = int((time.monotonic() - t0) * 1000)
    if stream is None:
        result = CallResult(mode=mode, turn=turn, wire_sent=w.sent, wire_recv=w.recv,
                            req_payload_bytes=req_payload_bytes,
                            resp_payload_bytes=len(err_body),
                            elapsed_ms=elapsed, turn_end_ms=elapsed,
                            req_sent_ms=streaming.since(t0, w.last_send_at),
                            ttfb_ms=streaming.since(t0, w.first_recv_at, elapsed),
                            ttft_ms=elapsed, ttlt_ms=elapsed,
                            request_json=body, response_json=err_body)
        result.error = f"http_{resp.status_code}: {err_body[:200]}"
        # A restricted-VIP refusal is a 403 that reads like a rejected key. Name it,
        # or the operator spends the afternoon rotating a key that works fine.
        if resp.status_code == 403:
            import netdiag
            if netdiag.is_vip_block(err_body):
                result.error += (" — NOT a key problem: this VPC routes "
                                 "googleapis.com to a restricted VIP that does not "
                                 "carry the Gemini Developer API. See /diagnose.")
        return result

    # The chunks, reassembled into the body a non-streaming call would have returned.
    # That is what the history echo and the audit trail both want; the SSE framing
    # itself is not information about the conversation.
    data = streaming.gen_response(stream.events)
    result = CallResult(
        mode=mode, turn=turn,
        wire_sent=w.sent, wire_recv=w.recv,
        req_payload_bytes=req_payload_bytes,
        resp_payload_bytes=len(stream.raw),
        elapsed_ms=elapsed,
        req_sent_ms=streaming.since(t0, w.last_send_at),
        ttfb_ms=streaming.since(t0, w.first_recv_at, stream.ttft_ms),
        ttft_ms=stream.ttft_ms, ttlt_ms=stream.ttlt_ms,
        turn_end_ms=stream.turn_end_ms,
        request_json=body,
        response_json=json.dumps(data),
    )

    try:
        usage = data.get("usageMetadata", {})
        result.prompt_tokens = int(usage.get("promptTokenCount", 0))
        result.resp_tokens = int(usage.get("candidatesTokenCount", 0))
        result.cached_tokens = int(usage.get("cachedContentTokenCount", 0))
        result.thought_tokens = int(usage.get("thoughtsTokenCount", 0))
        result.total_tokens = int(
            usage.get("totalTokenCount", result.prompt_tokens + result.resp_tokens)
        )
        # The answer as it streamed: thought parts excluded, so a reasoning summary
        # never ends up in the transcript (or in a cache built from it).
        result.response_text = stream.text
    except Exception as exc:
        result.error = f"parse_failed: {exc}"

    return result


def _min_cache_tokens() -> int:
    # Read at call time so tests can vary it. Gemini 3.x floor is 4096; the value
    # for flash-lite is unpublished, so this is a guard, not an authority — a live
    # create is what actually settles it.
    return int(os.environ.get("MIN_CACHE_TOKENS", "2048"))


def create_cache(model: str, contents: list, ttl_seconds: int = 1800,
                 system_instruction: str = "") -> dict:
    """Create a Developer API cachedContent holding `contents` (and optionally the
    system prompt). Returns {name, cached_tokens, error}. Skips (name=None) below
    the min token size. Mock mode returns a synthetic cache.

    `name` comes back as `cachedContents/{id}` and is what generateContent's
    `cachedContent` field references.
    """
    approx = _text_tokens(contents) + (len(system_instruction) // 4)
    floor = _min_cache_tokens()
    body = json.dumps(cache_create_body(model, contents, system_instruction,
                                        ttl_seconds))
    zero_wire = {"wire_sent": 0, "wire_recv": 0, "elapsed_ms": 0,
                 "request_raw": body, "response_raw": ""}
    if approx < floor:
        return {"name": None, "cached_tokens": 0,
                "error": f"below_min ({approx} < {floor} tokens)", **zero_wire}
    if is_mock():
        return {"name": f"cachedContents/mock_{secrets.token_hex(4)}",
                "cached_tokens": approx, "error": "",
                "wire_sent": len(body) + 200, "wire_recv": 200, "elapsed_ms": 0,
                "request_raw": body, "response_raw": ""}
    t0 = time.monotonic()
    try:
        with wire_counter() as w:
            resp = _session().post(cache_base_url(), data=body,
                                   headers={"Content-Type": "application/json",
                                            **auth_headers()}, timeout=120)
            _ = resp.content
    except Exception as exc:
        return {"name": None, "cached_tokens": 0, "error": f"create_failed: {exc}",
                "wire_sent": 0, "wire_recv": 0,
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
                "request_raw": body, "response_raw": ""}
    elapsed = int((time.monotonic() - t0) * 1000)
    wire = {"wire_sent": w.sent, "wire_recv": w.recv, "elapsed_ms": elapsed,
            "request_raw": body, "response_raw": resp.text}
    if resp.status_code not in (200, 201):
        return {"name": None, "cached_tokens": 0,
                "error": f"http_{resp.status_code}: {resp.text[:200]}", **wire}
    data = resp.json()
    tok = int(data.get("usageMetadata", {}).get("totalTokenCount", approx))
    return {"name": data.get("name"), "cached_tokens": tok, "error": "", **wire}


def delete_cache(name: str) -> None:
    """Best-effort delete of a cachedContent. No-op in mock / on error.

    `name` is `cachedContents/{id}`; the delete endpoint is /v1beta/{name}."""
    if not name or is_mock() or "mock_" in name:
        return
    try:
        _session().delete(f"{api_base()}/{name}", headers=auth_headers(), timeout=60)
    except Exception:
        pass


DEFAULT_MODEL = "gemini-3.1-flash-lite"

# The catalog reports supportedGenerationMethods. Two of the three arms are
# decidable from it; the third is not.
#   stateless / nocontext -> generateContent
#   cached                -> createCachedContent   (no method, no cache to build)
#   interaction           -> not advertised at all. No interactions method ever
#                            appears in the catalog, so the only way to know is to
#                            call it -- that is what the probe is for.
_GENERATE = "generateContent"
_CACHE = "createCachedContent"


def _model_entry(mid: str, methods: list) -> dict:
    can_cache = _CACHE in methods
    note = "all arms (cache supported)" if can_cache else "no cache arm — createCachedContent unsupported"
    return {
        "id": mid,
        "label": f"{mid} — {note}",
        "methods": sorted(methods),
        "can_generate": _GENERATE in methods,
        "can_cache": can_cache,
        # Interaction support is unknowable here, so "ready" means the two arms the
        # catalog *can* answer for. The probe covers the third.
        "comparison_ready": _GENERATE in methods and can_cache,
    }


# Fallback when the catalog can't be reached (verified against the live Developer
# API catalog on 2026-07-13). gemini-2.5-* is retired and 404s for new users, so it
# is not offered: handing back a model that cannot run at all is worse than a short
# list.
STATIC_MODELS = [
    _model_entry("gemini-3.1-flash-lite", [_GENERATE, "countTokens", _CACHE]),
    _model_entry("gemini-3.1-pro-preview", [_GENERATE, "countTokens", _CACHE]),
    _model_entry("gemini-3.5-flash", [_GENERATE, "countTokens", _CACHE]),
    _model_entry("gemini-3-pro-preview", [_GENERATE, "countTokens", _CACHE]),
]


def list_models() -> dict:
    """Models the Developer API catalog offers, tagged with the arms each can serve.

    Every arm runs on generativelanguage with an API key, so this asks that catalog
    -- a Vertex publisher list would describe a host the experiment never calls, and
    would happily offer a model that cannot build a cache, silently gutting the
    `cached` arm. Models that cannot generateContent at all (embeddings, bidi-only
    live models) are dropped.

    Returns {source, default, models:[{id,label,methods,can_cache,comparison_ready}]}.
    """
    fallback = {"source": "static", "default": DEFAULT_MODEL, "models": STATIC_MODELS}
    if is_mock() or not api_key():
        return fallback
    try:
        models, token = [], None
        while True:
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            resp = _session().get(f"{api_base()}/models", headers=auth_headers(),
                                  params=params, timeout=30)
            if resp.status_code != 200:
                return {**fallback, "source": f"static-fallback (http_{resp.status_code})"}
            body = resp.json()
            for m in body.get("models", []):
                mid = m.get("name", "").split("/")[-1]
                methods = m.get("supportedGenerationMethods", [])
                if mid.startswith("gemini") and _GENERATE in methods:
                    models.append(_model_entry(mid, methods))
            token = body.get("nextPageToken")
            if not token:
                break
        if not models:
            return {**fallback, "source": "static-fallback (empty)"}
        # Models that can run the whole comparison come first; the rest stay
        # selectable, but the label says which arm they'd break.
        models.sort(key=lambda m: (not m["comparison_ready"], m["id"]))
        default = DEFAULT_MODEL if any(m["id"] == DEFAULT_MODEL for m in models) else models[0]["id"]
        return {"source": "devapi", "default": default, "models": models}
    except Exception as exc:
        return {**fallback, "source": f"static-fallback ({type(exc).__name__})"}
