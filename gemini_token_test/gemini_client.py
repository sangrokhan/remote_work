"""Single Vertex AI generateContent call with real wire-byte counting.

Targets **Vertex AI** (aiplatform.googleapis.com), NOT the Developer API.
Auth = OAuth bearer via Application Default Credentials (google-auth). On Cloud
Run the token comes from the metadata server / service account automatically.

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

# Vertex config (overridable via env; project/creds come from ADC on Cloud Run).
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")


def _vertex_host() -> str:
    return "aiplatform.googleapis.com" if LOCATION == "global" else f"{LOCATION}-aiplatform.googleapis.com"


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
    response_text: str = ""
    # Raw JSON bodies exactly as sent to / received from the server (no headers,
    # so no bearer token). Lets the log show the full wire payload per step.
    request_json: str = ""
    response_json: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# Cached input tokens are billed at 10% of normal for Gemini 2.5+.
CACHED_DISCOUNT = 0.10
# Vertex context caching needs >= ~2048 tokens; below this, skip caching.
MIN_CACHE_TOKENS = int(os.environ.get("MIN_CACHE_TOKENS", "2048"))


# Module-global byte tally. Every counting socket and reader adds to it, so the
# count survives connection pooling: a keep-alive socket keeps feeding the tally
# whether or not connect() fired for this request. wire_counter() reads it by
# difference. The experiment loop is single-threaded, so one request is ever in
# flight and the difference is exact.
_wire_tally = {"sent": 0, "recv": 0}


class _WireDelta:
    """Bytes sent/received during one wire_counter() block."""
    __slots__ = ("sent", "recv")

    def __init__(self):
        self.sent = 0
        self.recv = 0


@contextmanager
def wire_counter():
    """Count HTTP bytes on the socket for the enclosed request(s), headers and
    content-encoding included, regardless of keep-alive reuse.

    Yields a _WireDelta whose .sent/.recv are populated when the block exits.
    """
    before_sent = _wire_tally["sent"]
    before_recv = _wire_tally["recv"]
    delta = _WireDelta()
    try:
        yield delta
    finally:
        delta.sent = _wire_tally["sent"] - before_sent
        delta.recv = _wire_tally["recv"] - before_recv


class _CountingReader:
    """Wraps the file object returned by socket.makefile(), counting bytes read.

    http.client / urllib3 read the response through sock.makefile() rather than
    sock.recv(), so the read path must be counted here or wire_recv stays 0.
    """

    def __init__(self, fp, counter):
        self._fp = fp
        self._c = counter

    def read(self, *a, **k):
        b = self._fp.read(*a, **k)
        self._c.recv += len(b)
        _wire_tally["recv"] += len(b)
        return b

    def read1(self, *a, **k):
        b = self._fp.read1(*a, **k)
        self._c.recv += len(b)
        _wire_tally["recv"] += len(b)
        return b

    def readline(self, *a, **k):
        b = self._fp.readline(*a, **k)
        self._c.recv += len(b)
        _wire_tally["recv"] += len(b)
        return b

    def readinto(self, buf):
        n = self._fp.readinto(buf)
        self._c.recv += n or 0
        _wire_tally["recv"] += n or 0
        return n

    def __getattr__(self, name):
        return getattr(self._fp, name)


class _CountingSocket:
    """Wraps a socket, tallying every byte sent and received."""

    def __init__(self, sock):
        self._sock = sock
        self.sent = 0
        self.recv = 0

    def sendall(self, data, *args, **kwargs):
        self.sent += len(data)
        _wire_tally["sent"] += len(data)
        return self._sock.sendall(data, *args, **kwargs)

    def send(self, data, *args, **kwargs):
        n = self._sock.send(data, *args, **kwargs)
        self.sent += n
        _wire_tally["sent"] += n
        return n

    def recv(self, bufsize, *args, **kwargs):
        chunk = self._sock.recv(bufsize, *args, **kwargs)
        self.recv += len(chunk)
        _wire_tally["recv"] += len(chunk)
        return chunk

    def recv_into(self, buf, *args, **kwargs):
        n = self._sock.recv_into(buf, *args, **kwargs)
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


class _CountingHTTPSConnection(HTTPSConnection):
    """HTTPS connection that swaps in a counting socket after connect."""

    def connect(self):
        super().connect()
        counter = _CountingSocket(self.sock)
        self.sock = counter
        _active_counter["counter"] = counter


class _CountingHTTPConnection(HTTPConnection):
    """Plain-HTTP counterpart, so a local (TLS-less) test server is counted too."""

    def connect(self):
        super().connect()
        counter = _CountingSocket(self.sock)
        self.sock = counter
        _active_counter["counter"] = counter


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
_CREDS = None


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


def _bearer_token() -> str:
    """ADC OAuth token (service account on Cloud Run, gcloud creds locally)."""
    global _CREDS
    import google.auth
    from google.auth.transport.requests import Request as GAuthRequest

    if _CREDS is None:
        _CREDS, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    if not _CREDS.valid:
        _CREDS.refresh(GAuthRequest())
    return _CREDS.token


def vertex_url(model: str) -> str:
    return (
        f"https://{_vertex_host()}/v1/projects/{PROJECT}/locations/{LOCATION}"
        f"/publishers/google/models/{model}:generateContent"
    )


def ready() -> tuple[bool, str]:
    """Whether a real call can be made. Returns (ok, reason_if_not)."""
    if is_mock():
        return True, ""
    if not api_key():
        return False, "GEMINI_API_KEY not set (or run with GEMINI_MOCK=1)."
    return True, ""


def _ready_vertex_unused() -> tuple[bool, str]:
    """Retained ADC readiness check, unused now that the comparison is on the
    Developer API. Kept so the Vertex path can be revived without re-deriving it."""
    if not PROJECT:
        return False, "GOOGLE_CLOUD_PROJECT not set (or run with GEMINI_MOCK=1)."
    try:
        _bearer_token()
    except Exception as exc:
        return False, f"No ADC credentials: {exc}"
    return True, ""


def _text_tokens(contents: list) -> int:
    text_len = sum(len(p.get("text", "")) for c in contents for p in c.get("parts", []))
    return max(1, text_len // 4)


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
    # Synthetic response body mirroring the real generateContent shape.
    resp_json = json.dumps({
        "candidates": [{"content": {"role": "model",
                                    "parts": [{"text": resp_text}]}}],
        "usageMetadata": {"promptTokenCount": prompt_tokens,
                          "candidatesTokenCount": resp_tokens,
                          "cachedContentTokenCount": cached_tokens,
                          "totalTokenCount": prompt_tokens + resp_tokens + cached_tokens},
    })
    return CallResult(
        mode=mode, turn=turn,
        prompt_tokens=prompt_tokens, resp_tokens=resp_tokens,
        total_tokens=prompt_tokens + resp_tokens + cached_tokens,
        cached_tokens=cached_tokens, response_text=resp_text,
        wire_sent=req_bytes + 200, wire_recv=resp_bytes + 200,
        req_payload_bytes=req_bytes, resp_payload_bytes=resp_bytes,
        request_json=body, response_json=resp_json,
    )


def call_gemini(model: str, contents: list, mode: str, turn: int,
                cached_content: str | None = None,
                cached_tokens_hint: int = 0) -> CallResult:
    """One Vertex generateContent call, optionally with a cachedContent ref.

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
            resp = _session().post(generate_url(model), data=body,
                                   headers=headers, timeout=120)
            _ = resp.content            # force the body to be read inside the counter
    except Exception as exc:
        return CallResult(mode=mode, turn=turn, req_payload_bytes=req_payload_bytes,
                          request_json=body, error=f"request_failed: {exc}",
                          elapsed_ms=int((time.monotonic() - t0) * 1000))

    result = CallResult(
        mode=mode, turn=turn,
        wire_sent=w.sent, wire_recv=w.recv,
        req_payload_bytes=req_payload_bytes,
        resp_payload_bytes=len(resp.content),
        elapsed_ms=int((time.monotonic() - t0) * 1000),
        request_json=body,
        response_json=resp.text,  # raw response body exactly as received
    )

    if resp.status_code != 200:
        result.error = f"http_{resp.status_code}: {resp.text[:200]}"
        return result

    try:
        data = resp.json()
        usage = data.get("usageMetadata", {})
        result.prompt_tokens = int(usage.get("promptTokenCount", 0))
        result.resp_tokens = int(usage.get("candidatesTokenCount", 0))
        result.cached_tokens = int(usage.get("cachedContentTokenCount", 0))
        result.thought_tokens = int(usage.get("thoughtsTokenCount", 0))
        result.total_tokens = int(
            usage.get("totalTokenCount", result.prompt_tokens + result.resp_tokens)
        )
        cands = data.get("candidates", [])
        if cands:
            parts = cands[0].get("content", {}).get("parts", [])
            result.response_text = "".join(p.get("text", "") for p in parts)
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


# Curated fallback list (2026-06). gemini-2.0-* retired 2026-06-01; 2.5 GA until
# 2026-10-16. Cheapest GA model first -> used as the default.
DEFAULT_MODEL = "gemini-2.5-flash-lite"
STATIC_MODELS = [
    {"id": "gemini-2.5-flash-lite", "label": "gemini-2.5-flash-lite (GA · cheapest)", "status": "GA"},
    {"id": "gemini-2.5-flash", "label": "gemini-2.5-flash (GA)", "status": "GA"},
    {"id": "gemini-2.5-pro", "label": "gemini-2.5-pro (GA)", "status": "GA"},
    {"id": "gemini-3.1-flash-lite", "label": "gemini-3.1-flash-lite (preview)", "status": "preview"},
    {"id": "gemini-3.1-pro", "label": "gemini-3.1-pro (preview)", "status": "preview"},
]


def list_models() -> dict:
    """Available Gemini models for the dropdown.

    Live (real project + creds): query Vertex publisher models. Mock / no creds /
    any failure: the curated STATIC_MODELS fallback. Always returns a usable list.
    Returns {source, default, models:[{id,label,status}]}.
    """
    fallback = {"source": "static", "default": DEFAULT_MODEL, "models": STATIC_MODELS}
    if is_mock() or not PROJECT:
        return fallback
    try:
        token = _bearer_token()
        url = f"https://{_vertex_host()}/v1beta1/publishers/google/models"
        resp = _session().get(
            url, headers={"Authorization": f"Bearer {token}"},
            params={"pageSize": 200}, timeout=30,
        )
        if resp.status_code != 200:
            return {**fallback, "source": f"static-fallback (http_{resp.status_code})"}
        models = []
        for m in resp.json().get("publisherModels", []):
            mid = m.get("name", "").split("/")[-1]
            if mid.startswith("gemini"):
                stage = m.get("launchStage", "").replace("GA", "GA").replace("_", " ").lower()
                models.append({"id": mid, "label": mid, "status": stage})
        if not models:
            return {**fallback, "source": "static-fallback (empty)"}
        models.sort(key=lambda x: x["id"])
        default = DEFAULT_MODEL if any(m["id"] == DEFAULT_MODEL for m in models) else models[0]["id"]
        return {"source": "vertex", "default": default, "models": models}
    except Exception as exc:
        return {**fallback, "source": f"static-fallback ({type(exc).__name__})"}
