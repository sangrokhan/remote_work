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
from core import wire

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
    """2 ms, and actually delivered.

    This is the assertion that would have caught the old behaviour. Dumping the whole
    socket table every tick cost about 5 ms, so asking for 2 ms silently produced 5 --
    the sampler reported the period it wanted in its metadata while running at half
    the rate, and events landed at the wrong times with nothing to say so.
    """
    result = _monitored(sink, lambda c: (c.sendall(b"ping"), time.sleep(0.5)))

    assert result["interval_ms"] == 2
    # Ticks, not samples: samples are per (tick, socket). The window is ~0.7s, so 2 ms
    # should give ~350. Demand at least half of that: a loaded box may miss ticks, but
    # falling to a 5 ms period -- which is what a table walk per tick produced -- would
    # leave under 150 and fail here.
    assert result["ticks"] >= 175, (
        f"asked for 2ms and got {result['ticks']} ticks in ~0.7s "
        f"(~{700 / max(result['ticks'], 1):.1f}ms per tick)")


def test_it_stops_dumping_once_it_knows_which_socket_to_watch(sink):
    """The change that made 2 ms affordable, asserted on the numbers the helper reports.

    A dump makes the kernel walk the entire established hash table -- measured at
    2410us, and the same whether it returns eleven sockets or none, because the walk is
    the cost. An exact query is a hash lookup at 3us. If this ever regresses to dumping
    per tick, the period silently stretches and the reset lands at the wrong time, so
    the ratio is worth pinning rather than trusting.
    """
    result = _monitored(sink, lambda c: (c.sendall(b"ping"), time.sleep(0.5)))
    dumps, exacts = result["dumps"], result["exact_queries"]

    assert result["ticks"] > 0
    assert exacts > 0, "never switched to exact queries; every tick was a full dump"
    # Rediscovery runs on a 100ms timer, so ~0.7s of monitoring allows a handful of
    # dumps. Anything approaching one per tick means the fast path is not being taken.
    assert dumps <= result["ticks"] // 10, (
        f"{dumps} dumps in {result['ticks']} ticks -- expected a dump only every "
        f"100ms, not per tick")
    assert result["tracked"] >= 1


def test_a_connection_announced_by_the_client_is_sampled_from_its_first_window(sink):
    """The initial congestion window, recorded -- which discovery-by-dump could not do.

    A connection opens at cwnd=10 and, once data flows, is past 60 within a few round
    trips. Waiting for the next rediscovery dump to notice it meant arriving 4-22ms
    late with the window already at 48-75: measured, five connections in a row, initial
    window missed on every one. That is a pcap with a three-way handshake in it beside
    a cwnd log that starts mid-flight.

    core.wire announces each socket the instant connect() returns, so the helper
    queries it on the next tick instead.
    """
    # The monitor starts with nothing to watch, and the connection is opened after it
    # is already running -- the case discovery-by-dump arrived too late for. Only one
    # connection, because the sink fixture accepts exactly one: a second would never be
    # drained and sendall would block forever.
    monitor = cwnd.Monitor("test", "announced", "127.0.0.1", port=sink)
    monitor.__enter__()
    client = None
    try:
        time.sleep(0.2)
        connected_at = time.time()
        client = socket.create_connection(("127.0.0.1", sink))
        wire._announce(client)          # what _CountingConnection.connect() does
        local = f"127.0.0.1:{client.getsockname()[1]}"
        for _ in range(BURST):
            client.sendall(BLOB)
        time.sleep(0.2)
    finally:
        monitor.stop()
        if client is not None:
            client.close()

    result = monitor.result()
    assert result["announced"] >= 1, "the connection was never announced"

    mine = sorted((s for s in result["samples"] if s["local"] == local),
                  key=lambda s: s["t_ms"])
    assert mine, f"{local} was announced but never sampled"

    # The claim, and the only one loopback can make honestly: the socket was picked up
    # because the client said so, not because a dump happened to come round. The
    # rediscovery timer is 100ms, so anything under a few milliseconds could not have
    # come from it.
    #
    # What is NOT asserted is that cwnd=10 was recorded. On loopback the RTT is tens of
    # microseconds and the window is past 10 before the next 2ms tick, so that check
    # would be a coin toss -- the same sampling limit this file explains at the top. On
    # a real path there are milliseconds of slow start to land in, and the lag asserted
    # here is what buys them.
    lag_ms = (mine[0]["wall"] - connected_at) * 1000
    assert lag_ms < 25, (
        f"first sample of {local} arrived {lag_ms:.1f}ms after connect -- that is "
        f"discovery-by-dump timing, not an announcement")

    # And it is a real transfer, so the samples describe congestion control rather than
    # an idle socket parked at its initial window.
    assert max(s["snd_cwnd"] for s in mine) > cwnd.INIT_CWND


def test_the_helper_emits_every_field_the_csv_promises(sink):
    """The third of the three lists that have to agree. tests/test_cwnd.py checks that
    SAMPLE_FIELDS and the CSV header match each other; only a real sample can show that
    the C helper emits them."""
    result = _monitored(sink, lambda c: (c.sendall(b"ping"), time.sleep(0.2)))

    assert result["samples"], "no samples from a live loopback socket"
    missing = [f for f in cwnd.SAMPLE_FIELDS if f not in result["samples"][0]]
    assert not missing, f"helper does not emit: {missing}"
