"""The monitor against a real socket: real netlink, real congestion state, no API.

Everything else in the suite about congestion is arithmetic on made-up samples. This is
where the helper is pointed at a socket that actually exists, on 127.0.0.1, and its
numbers are checked against `ss -ti` -- the tool it is imitating. If these agree, the
netlink parsing, the destination filter, the field layout and the sample rate are all
right, and the synthetic tests are testing something real.

What is deliberately NOT asserted here is the idle reset itself. Loopback RTT is tens
of microseconds, so a window that gets restarted after an idle gap climbs all the way
back inside a single 10 ms sample -- measured: over five trials the drop was visible
twice. That is not a flaky test to be retried, it is the sampling limit stated plainly
in native/cwnd_monitor.c, and a path with a real RTT does not have it: against an API
34 ms away, slow start takes hundreds of milliseconds to climb and every step of it
lands in its own sample. `native/idle_reset_demo.py` is the reproduction, and it says
what it needs to be believed.

No API, no key, no external network -- a server on 127.0.0.1 and a client talking to
it, the same shape tests/test_wire.py uses to count bytes without paying a vendor.

Skipped, not failed, where the helper cannot run: a box with no compiler, or a sandbox
with no AF_NETLINK. What is never done is passing without checking.
"""

import re
import shutil
import socket
import subprocess
import threading
import time

import pytest

from core import cwnd

BLOB = b"x" * (1 << 20)
BURST = 12

_available, _reason = cwnd.available()
pytestmark = pytest.mark.skipif(
    not _available, reason=f"cwnd monitor unavailable: {_reason}")


@pytest.fixture
def sink():
    """A loopback server that reads and discards. Yields its port.

    It has to actually drain: a server that stopped reading would fill the receive
    window, and a window-limited sender's cwnd says nothing about congestion control.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    stop = threading.Event()

    def serve():
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        with conn:
            while not stop.is_set():
                try:
                    if not conn.recv(1 << 20):
                        return
                except OSError:
                    return

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield srv.getsockname()[1]
    finally:
        stop.set()
        srv.close()
        thread.join(timeout=2)


def _monitored(port, body):
    """Run `body` with a monitor watching 127.0.0.1:port, and return its result."""
    client = socket.create_connection(("127.0.0.1", port))
    monitor = cwnd.Monitor("test", "loopback", "127.0.0.1", port=port)
    monitor.__enter__()
    try:
        time.sleep(0.2)          # let the monitor tick before any data moves
        body(client)
    finally:
        monitor.stop()
        client.close()
    return monitor.result()


def test_a_live_socket_produces_real_congestion_state(sink):
    """The end-to-end claim: bytes move, and the kernel's view of that connection --
    the window it is willing to fill, the threshold it will stop growing at, the round
    trip it measured -- comes back through netlink for a socket this process does not
    own."""
    def burst(client):
        for _ in range(BURST):
            client.sendall(BLOB)
        time.sleep(0.2)

    result = _monitored(sink, burst)

    assert result["error"] == ""
    assert result["sample_count"] > 0, "netlink returned nothing for a live socket"
    assert result["sockets"], "no socket matched the loopback destination"

    # The window grew past the initial 10 segments -- so the monitor is watching the
    # sending side of a connection that actually did congestion control, not an idle
    # socket sitting at its initial value.
    assert result["peak_cwnd"] > cwnd.INIT_CWND, (
        f"window never grew past {cwnd.INIT_CWND} (peak {result['peak_cwnd']})")

    sample = result["samples"][0]
    assert sample["remote"].endswith(f":{sink}")
    assert sample["state"] == "ESTABLISHED"
    assert sample["rtt_us"] > 0
    assert sample["snd_mss"] > 0


def test_only_the_matching_destination_is_reported(sink):
    """The filter is the difference between a monitor and a firehose. A run watches one
    API host; every other socket on the box is somebody else's traffic and must not end
    up in the arm's CSV."""
    other = socket.socket()
    other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    other.bind(("127.0.0.1", 0))
    other.listen(1)
    noise = socket.create_connection(("127.0.0.1", other.getsockname()[1]))
    try:
        result = _monitored(sink, lambda c: (c.sendall(b"ping"), time.sleep(0.2)))
    finally:
        noise.close()
        other.close()

    remotes = {s["remote"] for s in result["samples"]}
    assert remotes == {f"127.0.0.1:{sink}"}, f"filter leaked: {remotes}"


@pytest.mark.skipif(not shutil.which("ss"), reason="ss not installed")
def test_the_numbers_agree_with_ss(sink):
    """Cross-checked against the tool it imitates, on the same socket at nearly the
    same moment. `ss` and this helper read the same struct through the same interface;
    if they disagree, the parsing is wrong somewhere and every number above is fiction.
    """
    captured = {}

    def burst_then_ask(client):
        for _ in range(BURST):
            client.sendall(BLOB)
        time.sleep(0.1)
        local_port = client.getsockname()[1]
        out = subprocess.run(["ss", "-tin", "state", "established",
                              f"( dport = :{sink} )"],
                             capture_output=True, text=True, timeout=10).stdout
        captured["port"] = local_port
        captured["ss"] = out
        time.sleep(0.1)

    result = _monitored(sink, burst_then_ask)

    # Find the paragraph ss printed for our socket, and the cwnd in it.
    block = ""
    lines = captured["ss"].splitlines()
    for i, line in enumerate(lines):
        if f"127.0.0.1:{captured['port']}" in line and i + 1 < len(lines):
            block = lines[i + 1]
            break
    if not block:
        pytest.skip("ss did not report the socket (it closed too quickly)")

    m = re.search(r"\bcwnd:(\d+)", block)
    assert m, f"no cwnd in ss output: {block!r}"
    ss_cwnd = int(m.group(1))

    ours = [s["snd_cwnd"] for s in result["samples"]]
    assert ss_cwnd in ours, (
        f"ss reported cwnd {ss_cwnd}; the monitor never saw it. Saw: "
        f"{sorted(set(ours))}")

    m = re.search(r"\bmss:(\d+)", block)
    if m:
        assert int(m.group(1)) in {s["snd_mss"] for s in result["samples"]}


def test_the_sample_rate_is_the_one_that_was_asked_for(sink):
    """10 ms, near enough. A monitor that quietly sampled at 100 ms would still find a
    multi-second idle reset and would still be wrong about when it happened."""
    result = _monitored(sink, lambda c: (c.sendall(b"ping"), time.sleep(0.5)))

    assert result["interval_ms"] == 10
    # Ticks, not samples: samples are per (tick, socket). Slack is generous because a
    # loaded box is allowed to miss ticks -- the claim is "about a hundred a second",
    # not a real-time guarantee this program does not make.
    assert result["ticks"] >= 30, f"only {result['ticks']} ticks in ~0.7s"


def test_the_helper_emits_every_field_the_csv_promises(sink):
    """The third of the three lists that have to agree. tests/test_cwnd.py checks that
    SAMPLE_FIELDS and the CSV header match each other; only a real sample can show that
    the C helper emits them."""
    result = _monitored(sink, lambda c: (c.sendall(b"ping"), time.sleep(0.2)))

    assert result["samples"], "no samples from a live loopback socket"
    missing = [f for f in cwnd.SAMPLE_FIELDS if f not in result["samples"][0]]
    assert not missing, f"helper does not emit: {missing}"
