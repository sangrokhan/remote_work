"""Real wire-byte counting for HTTP(S) requests made through `requests`.

Ported from gemini_token_test/gemini_client.py, stripped of Gemini specifics.

We wrap the socket (send path *and* the makefile() read path) so every byte of
the HTTP exchange is tallied — headers plus the content-encoded body, i.e. the
size actually transferred. This is post-decryption HTTP framing, not raw TLS
ciphertext.

Why not the official `openai` SDK: it rides on httpx, which does not use
http.client / urllib3 connection classes, so this counter cannot attach to it.
We speak the REST API directly through `requests` instead.

Usage:

    sess = session()
    with wire_counter() as w:
        resp = sess.post(url, json=body, headers=headers)
    w.sent, w.recv   # bytes on the wire for that exchange
"""

from __future__ import annotations

from contextlib import contextmanager

import requests
from urllib3.connection import HTTPSConnection, HTTPConnection

# Module-global byte tally. Every counting socket and reader adds to it, so the
# count survives connection pooling: a keep-alive socket keeps feeding the tally
# whether or not connect() fired for this request. wire_counter() reads it by
# difference. The experiment loop is single-threaded, so only one request is ever
# in flight and the difference is exact.
_wire_tally = {"sent": 0, "recv": 0}


class WireDelta:
    """Bytes sent/received during one wire_counter() block."""

    __slots__ = ("sent", "recv")

    def __init__(self):
        self.sent = 0
        self.recv = 0


@contextmanager
def wire_counter():
    """Count HTTP bytes on the socket for the enclosed request(s), headers and
    content-encoding included, regardless of keep-alive reuse.

    Yields a WireDelta whose .sent/.recv are populated when the block exits.
    """
    before_sent = _wire_tally["sent"]
    before_recv = _wire_tally["recv"]
    delta = WireDelta()
    try:
        yield delta
    finally:
        delta.sent = _wire_tally["sent"] - before_sent
        delta.recv = _wire_tally["recv"] - before_recv


class _CountingReader:
    """Wraps the file object returned by socket.makefile(), counting bytes read.

    http.client / urllib3 read the response through sock.makefile() rather than
    sock.recv(), so the read path must be counted here or recv stays 0.
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


class _CountingHTTPSConnection(HTTPSConnection):
    """HTTPS connection that swaps in a counting socket after connect."""

    def connect(self):
        super().connect()
        self.sock = _CountingSocket(self.sock)


class _CountingHTTPConnection(HTTPConnection):
    """Plain-HTTP counterpart, so a local (TLS-less) test server is counted too."""

    def connect(self):
        super().connect()
        self.sock = _CountingSocket(self.sock)


def build_session() -> requests.Session:
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


_SESSION: requests.Session | None = None


def session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = build_session()
    return _SESSION


def reset_session() -> None:
    """Close pooled connections so the next call opens a fresh TCP socket."""
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()
        except Exception:
            pass
        _SESSION = None
