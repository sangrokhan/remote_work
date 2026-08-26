"""aipt.backends.mock.probe: idle-period HTTP PING probe.

Migrated from tcp_congestion/tests/test_probe.py (DESIGN.md 5, A3).
"""

import socket
import threading
import time

import pytest

from aipt.backends.mock import probe, server


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
    assert sample["rtt_ms"] < 50


def test_ping_sample_has_timestamp(conn):
    c, srv = conn
    sample = probe.ping(c, host=srv.host)
    assert "ts" in sample
    assert sample["ts"] > 0


def test_probe_does_not_carry_delivery_rate(conn):
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
        kwargs={"host": srv.host, "interval_ms": 20, "stop": stop, "out": results},
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
    stop.set()
    probe.run_probes(c, host=srv.host, interval_ms=20, stop=stop, out=results)
    assert results == []


def test_run_probes_timestamps_are_monotonic(conn):
    c, srv = conn
    stop = threading.Event()
    results = []
    t = threading.Thread(
        target=probe.run_probes,
        args=(c,),
        kwargs={"host": srv.host, "interval_ms": 10, "stop": stop, "out": results},
        daemon=True,
    )
    t.start()
    time.sleep(0.1)
    stop.set()
    t.join(timeout=1)
    ts = [r["ts"] for r in results]
    assert ts == sorted(ts)
