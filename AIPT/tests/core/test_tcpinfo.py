"""tcpinfo.py: one-shot getsockopt(TCP_INFO) snapshot.

Migrated unchanged from tcp_congestion/tests/test_tcpinfo.py (if present) --
token_traffic has no equivalent module. Uses a real localhost TCP connection
so the parsing round-trips against actual kernel-reported struct tcp_info
bytes rather than a hand-built fixture.
"""

from __future__ import annotations

import socket
import sys

import pytest

from aipt.core import tcpinfo


def _connected_pair():
    """A real, connected TCP socket pair on localhost."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    server, _ = listener.accept()
    listener.close()
    return client, server


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                     reason="TCP_INFO snapshot is Linux-specific")
def test_snapshot_returns_expected_keys_on_a_real_socket():
    client, server = _connected_pair()
    try:
        snap = tcpinfo.snapshot(client)
    finally:
        client.close()
        server.close()
    assert set(snap) == {"cwnd", "rtt_ms", "rto_ms", "delivery_rate"}
    assert isinstance(snap["cwnd"], int)
    assert snap["cwnd"] >= 0
    assert isinstance(snap["rtt_ms"], float)
    assert isinstance(snap["rto_ms"], float)
    assert isinstance(snap["delivery_rate"], int)


def test_snapshot_on_a_udp_socket_returns_zeros_not_an_exception():
    """TCP_INFO is not a valid getsockopt on a UDP socket -- OSError, and the
    module must swallow it rather than crash the experiment."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        snap = tcpinfo.snapshot(sock)
    finally:
        sock.close()
    assert snap == {"cwnd": 0, "rtt_ms": 0.0, "rto_ms": 0.0, "delivery_rate": 0}


def test_snapshot_returns_zeros_on_non_linux(monkeypatch):
    monkeypatch.setattr(tcpinfo.sys, "platform", "darwin")
    client, server = _connected_pair() if sys.platform.startswith("linux") else (None, None)
    if client is None:
        pytest.skip("needs a real socket to construct, even though platform is faked")
    try:
        snap = tcpinfo.snapshot(client)
    finally:
        client.close()
        server.close()
    assert snap == {"cwnd": 0, "rtt_ms": 0.0, "rto_ms": 0.0, "delivery_rate": 0}


def test_zeros_helper_shape():
    assert tcpinfo._zeros() == {"cwnd": 0, "rtt_ms": 0.0, "rto_ms": 0.0, "delivery_rate": 0}
