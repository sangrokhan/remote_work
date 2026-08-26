"""probe.py: idle-period HTTP PING probe.

During an LLM inference idle gap the client keeps the TCP connection alive and
sends periodic lightweight HTTP PINGs to measure RTT.  delivery_rate is NOT
updated by probes: the last value sampled right after the previous request
completes is preserved and returned alongside each RTT sample.

Tests cover:
  - a single ping returns a positive RTT
  - multiple pings produce a sequence of increasing timestamps
  - probe does not update delivery_rate (that value comes from the snapshot)
  - probe stops when the stop event is set
  - RTT values are in milliseconds (float)
"""

import socket
import threading
import time

import pytest

from tcp_congestion import probe, server


@pytest.fixture()
def srv():
    s = server.Server(host="127.0.0.1", port=0)
    t = threading.Thread(target=s.serve_forever, daemon=True)
    t.start()
    time.sleep(0.05)
    yield s
    s.shutdown()


@pytest.fixture()
def conn(srv):
    c = socket.create_connection((srv.host, srv.port), timeout=5)
    yield c, srv
    c.close()


def test_single_ping_returns_positive_rtt(conn):
    c, srv = conn
    sample = probe.ping(c, host=srv.host)
    assert sample["rtt_ms"] > 0


def test_ping_rtt_is_float_milliseconds(conn):
    c, srv = conn
    sample = probe.ping(c, host=srv.host)
    assert isinstance(sample["rtt_ms"], float)
    # loopback RTT < 50ms
    assert sample["rtt_ms"] < 50


def test_ping_sample_has_timestamp(conn):
    c, srv = conn
    sample = probe.ping(c, host=srv.host)
    assert "ts" in sample
    assert sample["ts"] > 0


def test_probe_does_not_carry_delivery_rate(conn):
    """The probe result intentionally omits delivery_rate.
    Callers inject the snapshot value themselves."""
    c, srv = conn
    sample = probe.ping(c, host=srv.host)
    assert "delivery_rate" not in sample


def test_run_probes_collects_multiple_samples(conn):
    c, srv = conn
    stop = threading.Event()
    results = []
    t = threading.Thread(
        target=probe.run_probes,
        args=(c,),
        kwargs={"host": srv.host, "interval_ms": 20, "stop": stop,
                "out": results},
        daemon=True,
    )
    t.start()
    time.sleep(0.15)
    stop.set()
    t.join(timeout=1)
    assert len(results) >= 3


def test_run_probes_stops_on_event(conn):
    c, srv = conn
    stop = threading.Event()
    results = []
    stop.set()  # stop immediately
    probe.run_probes(c, host=srv.host, interval_ms=20, stop=stop, out=results)
    assert results == []


def test_run_probes_timestamps_are_monotonic(conn):
    c, srv = conn
    stop = threading.Event()
    results = []
    t = threading.Thread(
        target=probe.run_probes,
        args=(c,),
        kwargs={"host": srv.host, "interval_ms": 10, "stop": stop,
                "out": results},
        daemon=True,
    )
    t.start()
    time.sleep(0.1)
    stop.set()
    t.join(timeout=1)
    ts = [r["ts"] for r in results]
    assert ts == sorted(ts)
