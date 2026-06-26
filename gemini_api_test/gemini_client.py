"""Single Gemini generateContent call with real wire-byte counting.

Measures, per call:
  - wire_sent / wire_recv : raw bytes through the TLS socket (real on-wire bytes)
  - req_payload_bytes / resp_payload_bytes : JSON body sizes (application layer)
  - prompt_tokens / resp_tokens / total_tokens : from response usageMetadata

No tcpdump / NET_ADMIN needed: we wrap the socket so every send()/recv() byte is
counted (this is ciphertext on the wire, ~= plaintext + small TLS overhead).

Mock mode (GEMINI_MOCK=1 or no key) returns synthetic data so the whole flow and
charts work without burning quota.
"""

from __future__ import annotations

import json
import os
import socket as _socket
from dataclasses import dataclass, asdict

import requests
from urllib3.connection import HTTPSConnection

ENDPOINT_HOST = "generativelanguage.googleapis.com"
ENDPOINT = f"{ENDPOINT_HOST}:443"

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
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


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


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _build_session()
    return _SESSION


def _is_mock(api_key: str) -> bool:
    return os.environ.get("GEMINI_MOCK") == "1" or not api_key


def _mock_call(mode: str, turn: int, contents: list) -> CallResult:
    body = json.dumps({"contents": contents})
    req_bytes = len(body.encode("utf-8"))
    # ~4 chars/token heuristic on the request text.
    text_len = sum(
        len(p.get("text", ""))
        for c in contents
        for p in c.get("parts", [])
    )
    prompt_tokens = max(1, text_len // 4)
    resp_tokens = 64  # fixed synthetic answer size
    resp_text = "x" * (resp_tokens * 4)
    resp_body = json.dumps(
        {
            "candidates": [{"content": {"parts": [{"text": resp_text}]}}],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": resp_tokens,
                "totalTokenCount": prompt_tokens + resp_tokens,
            },
        }
    )
    resp_bytes = len(resp_body.encode("utf-8"))
    # Synthetic wire bytes ~ payload + small TLS/HTTP overhead.
    return CallResult(
        mode=mode,
        turn=turn,
        prompt_tokens=prompt_tokens,
        resp_tokens=resp_tokens,
        total_tokens=prompt_tokens + resp_tokens,
        wire_sent=req_bytes + 200,
        wire_recv=resp_bytes + 200,
        req_payload_bytes=req_bytes,
        resp_payload_bytes=resp_bytes,
    )


def call_gemini(model: str, contents: list, api_key: str, mode: str, turn: int) -> CallResult:
    """One generateContent call. Never raises; errors land in CallResult.error."""
    if _is_mock(api_key):
        return _mock_call(mode, turn, contents)

    url = (
        f"https://{ENDPOINT_HOST}/v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )
    payload = {"contents": contents}
    body = json.dumps(payload)
    req_payload_bytes = len(body.encode("utf-8"))

    _active_counter["counter"] = None
    try:
        resp = _session().post(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
    except Exception as exc:  # network failure
        return CallResult(
            mode=mode, turn=turn, req_payload_bytes=req_payload_bytes,
            error=f"request_failed: {exc}",
        )

    counter = _active_counter["counter"]
    wire_sent = counter.sent if counter else req_payload_bytes
    wire_recv = counter.recv if counter else len(resp.content)
    resp_payload_bytes = len(resp.content)

    result = CallResult(
        mode=mode, turn=turn,
        wire_sent=wire_sent, wire_recv=wire_recv,
        req_payload_bytes=req_payload_bytes,
        resp_payload_bytes=resp_payload_bytes,
    )

    if resp.status_code != 200:
        result.error = f"http_{resp.status_code}: {resp.text[:200]}"
        return result

    try:
        data = resp.json()
        usage = data.get("usageMetadata", {})
        result.prompt_tokens = int(usage.get("promptTokenCount", 0))
        result.resp_tokens = int(usage.get("candidatesTokenCount", 0))
        result.total_tokens = int(
            usage.get("totalTokenCount", result.prompt_tokens + result.resp_tokens)
        )
    except Exception as exc:
        result.error = f"parse_failed: {exc}"

    return result
