"""Count the bytes an HTTP exchange actually puts on the wire, and time the write.

The number this whole experiment turns on is uplink bytes: how much a client has to
re-upload per turn under each strategy. Nothing above the socket can answer that.
`len(json.dumps(body))` misses the request line, the headers, and the
content-encoding -- a gzipped 90 KB history is not 90 KB on the wire -- so a
payload-size "measurement" would compare the arms on a quantity none of them pays.
We therefore wrap the socket itself: every byte written and every byte read is
tallied, headers and compression included. What is counted is the HTTP framing after
TLS decryption, not the ciphertext; for the true packet sizes there is `core.capture`.

The count lives in a module-global tally rather than on the connection object,
because `requests` pools connections. A keep-alive socket is created once and reused
for every turn afterwards, so anything installed in `connect()` and read back per
call would be `None` from turn 2 onward -- silently. Counting on the socket and
reading the tally by difference survives reuse.

Two marks ride along with the bytes, because the bytes alone cannot say how long the
upload took. A turn that ships a 90 KB history spends real time putting it on the
wire before the model has seen a word of it, and that time belongs to the client, not
to the model:

  last_send_at    the request's final byte went out
  first_recv_at   the response's first byte came back

Both connection classes are wrapped, HTTPS and plain HTTP: the providers are called
over TLS, and a localhost test server is called without it, and both must be counted
by the same code or the tests prove nothing about production.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import requests
from urllib3.connection import HTTPSConnection, HTTPConnection

# Every counting socket and reader adds to this tally, so the count survives
# connection pooling. wire_counter() reads it by difference. The experiment loop is
# single-threaded, so only one request is ever in flight and the difference is exact.
_wire_tally = {"sent": 0, "recv": 0}

# When the socket last wrote, and when it first read back, for the request in flight.
# Stamped on the socket, so they measure the write itself and not the library around it.
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

    Yields a _WireDelta whose fields are populated when the block exits. The response
    body must be read *inside* the block: on a streamed response the bytes only cross
    the socket while the stream is being consumed.
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


def _mark_recv(n: int) -> None:
    # Only the first byte of the response matters: after that the server is already
    # talking, and every later mark would just overwrite the answer we want.
    if n and _wire_marks["first_recv"] is None:
        _wire_marks["first_recv"] = time.monotonic()


class _CountingReader:
    """Wraps the file object returned by socket.makefile(), counting bytes read.

    http.client and urllib3 read the response through sock.makefile() rather than
    sock.recv(), so the read path has to be counted here as well or recv stays 0 on
    every real request.
    """

    def __init__(self, fp, counter):
        self._fp = fp
        self._c = counter

    def _count(self, n: int) -> None:
        _mark_recv(n)
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
    """Wraps a socket, tallying every byte sent and received and stamping the marks."""

    def __init__(self, sock):
        self._sock = sock
        self.sent = 0
        self.recv = 0

    def _mark_send(self) -> None:
        # Every write moves the mark, so once the request is out the mark sits on its
        # last byte -- which is exactly when the upload finished.
        _wire_marks["last_send"] = time.monotonic()

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
        _mark_recv(len(chunk))
        self.recv += len(chunk)
        _wire_tally["recv"] += len(chunk)
        return chunk

    def recv_into(self, buf, *args, **kwargs):
        n = self._sock.recv_into(buf, *args, **kwargs)
        _mark_recv(n or 0)
        self.recv += n or 0
        _wire_tally["recv"] += n or 0
        return n

    def makefile(self, mode="r", *args, **kwargs):
        fp = self._sock.makefile(mode, *args, **kwargs)
        # Only the readable binary path carries response bytes worth counting.
        if "b" in mode and "w" not in mode and "+" not in mode:
            return _CountingReader(fp, self)
        return fp

    def __getattr__(self, name):
        return getattr(self._sock, name)


class _CountingConnection:
    """Swaps in a counting socket once the connection is up.

    Mixed into both connection classes: HTTPS is what the providers are called over,
    and plain HTTP exists so a local TLS-less test server is counted by exactly the
    same code path.
    """

    def connect(self):
        super().connect()
        self.sock = _CountingSocket(self.sock)


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


_SESSION: requests.Session | None = None


def session() -> requests.Session:
    """The one session every call goes through. Its pools count."""
    global _SESSION
    if _SESSION is None:
        _SESSION = _build_session()
    return _SESSION


def reset_session() -> None:
    """Close pooled connections and drop the session, so the next call opens a fresh
    TCP socket with a fresh handshake.

    An arm that captures packets must call this before its capture window opens.
    Otherwise the pooled socket is already established, tcpdump starts recording
    mid-conversation, and the pcap shows acknowledgements for segments it never saw --
    a capture that cannot be read is worse than no capture.
    """
    global _SESSION
    if _SESSION is not None:
        try:
            _SESSION.close()   # closes all pooled sockets (sends FIN)
        except Exception:
            pass
        _SESSION = None
