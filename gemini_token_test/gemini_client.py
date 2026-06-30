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
from dataclasses import dataclass, asdict

import requests
from urllib3.connection import HTTPSConnection

# Vertex config (overridable via env; project/creds come from ADC on Cloud Run).
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_PROJECT", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")


def _vertex_host() -> str:
    return "aiplatform.googleapis.com" if LOCATION == "global" else f"{LOCATION}-aiplatform.googleapis.com"


ENDPOINT = f"{_vertex_host()}:443"

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
    response_text: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# Cached input tokens are billed at 10% of normal for Gemini 2.5+.
CACHED_DISCOUNT = 0.10
# Vertex context caching needs >= ~2048 tokens; below this, skip caching.
MIN_CACHE_TOKENS = int(os.environ.get("MIN_CACHE_TOKENS", "2048"))


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
        return b

    def read1(self, *a, **k):
        b = self._fp.read1(*a, **k)
        self._c.recv += len(b)
        return b

    def readline(self, *a, **k):
        b = self._fp.readline(*a, **k)
        self._c.recv += len(b)
        return b

    def readinto(self, buf):
        n = self._fp.readinto(buf)
        self._c.recv += n or 0
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
        return self._sock.sendall(data, *args, **kwargs)

    def send(self, data, *args, **kwargs):
        n = self._sock.send(data, *args, **kwargs)
        self.sent += n
        return n

    def recv(self, bufsize, *args, **kwargs):
        chunk = self._sock.recv(bufsize, *args, **kwargs)
        self.recv += len(chunk)
        return chunk

    def recv_into(self, buf, *args, **kwargs):
        n = self._sock.recv_into(buf, *args, **kwargs)
        self.recv += n
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


def _build_session() -> requests.Session:
    """Session whose https pool uses the counting connection class."""
    from requests.adapters import HTTPAdapter
    from urllib3.poolmanager import PoolManager
    from urllib3.connectionpool import HTTPSConnectionPool

    class _CountingPool(HTTPSConnectionPool):
        ConnectionCls = _CountingHTTPSConnection

    class _CountingPoolManager(PoolManager):
        def _new_pool(self, scheme, host, port, request_context=None):
            if scheme == "https":
                kw = self.connection_pool_kw.copy()
                kw.pop("scheme", None)
                return _CountingPool(host, port, **kw)
            return super()._new_pool(scheme, host, port, request_context)

    class _CountingAdapter(HTTPAdapter):
        def init_poolmanager(self, connections, maxsize, block=False, **kw):
            self.poolmanager = _CountingPoolManager(
                num_pools=connections, maxsize=maxsize, block=block, **kw
            )

    sess = requests.Session()
    sess.mount("https://", _CountingAdapter())
    return sess


_SESSION = None
_CREDS = None


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _build_session()
    return _SESSION


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
    return CallResult(
        mode=mode, turn=turn,
        prompt_tokens=prompt_tokens, resp_tokens=resp_tokens,
        total_tokens=prompt_tokens + resp_tokens + cached_tokens,
        cached_tokens=cached_tokens, response_text=resp_text,
        wire_sent=req_bytes + 200, wire_recv=resp_bytes + 200,
        req_payload_bytes=req_bytes, resp_payload_bytes=resp_bytes,
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

    try:
        token = _bearer_token()
    except Exception as exc:
        return CallResult(mode=mode, turn=turn, req_payload_bytes=req_payload_bytes,
                          error=f"auth_failed: {exc}")

    _active_counter["counter"] = None
    try:
        resp = _session().post(
            vertex_url(model),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=120,
        )
    except Exception as exc:
        return CallResult(mode=mode, turn=turn, req_payload_bytes=req_payload_bytes,
                          error=f"request_failed: {exc}")

    counter = _active_counter["counter"]
    wire_sent = counter.sent if counter else req_payload_bytes
    wire_recv = counter.recv if counter else len(resp.content)

    result = CallResult(
        mode=mode, turn=turn,
        wire_sent=wire_sent, wire_recv=wire_recv,
        req_payload_bytes=req_payload_bytes,
        resp_payload_bytes=len(resp.content),
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


def _cache_base_url() -> str:
    return (f"https://{_vertex_host()}/v1beta1/projects/{PROJECT}"
            f"/locations/{LOCATION}/cachedContents")


def create_cache(model: str, contents: list, ttl_seconds: int = 1800) -> dict:
    """Create a Vertex cachedContent of `contents`. Returns
    {name, cached_tokens, error}. Skips (name=None) if below the min token size.
    Mock mode returns a synthetic cache.
    """
    approx = _text_tokens(contents)
    if approx < MIN_CACHE_TOKENS:
        return {"name": None, "cached_tokens": 0,
                "error": f"below_min ({approx} < {MIN_CACHE_TOKENS} tokens)"}
    if is_mock():
        return {"name": f"projects/mock/locations/{LOCATION}/cachedContents/"
                        f"mock_{secrets.token_hex(4)}",
                "cached_tokens": approx, "error": ""}
    try:
        token = _bearer_token()
        full_model = (f"projects/{PROJECT}/locations/{LOCATION}"
                      f"/publishers/google/models/{model}")
        body = json.dumps({"model": full_model, "contents": contents,
                           "ttl": f"{ttl_seconds}s"})
        resp = _session().post(_cache_base_url(), data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            timeout=120)
        if resp.status_code not in (200, 201):
            return {"name": None, "cached_tokens": 0,
                    "error": f"http_{resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        tok = int(data.get("usageMetadata", {}).get("totalTokenCount", approx))
        return {"name": data.get("name"), "cached_tokens": tok, "error": ""}
    except Exception as exc:
        return {"name": None, "cached_tokens": 0, "error": f"create_failed: {exc}"}


def delete_cache(name: str) -> None:
    """Best-effort delete of a cachedContent. No-op in mock / on error."""
    if not name or is_mock() or name.startswith("projects/mock/"):
        return
    try:
        token = _bearer_token()
        _session().delete(f"https://{_vertex_host()}/v1beta1/{name}",
                          headers={"Authorization": f"Bearer {token}"}, timeout=60)
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
