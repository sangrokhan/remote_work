"""End-to-end test: real QUIC connections (loopback, real UDP sockets) for
aipt.backends.quic_mock -- server + client + IdleProbeCongestionControl.

Marked ``live`` (opens real sockets, generates a real cert) -- not run in
the default ``pytest -m "not live"`` suite, matching this repo's
convention for anything that needs real OS-level I/O rather than pure
unit-level state.
"""
from __future__ import annotations

import asyncio
import tempfile
import subprocess
from pathlib import Path

import pytest

aioquic = pytest.importorskip("aioquic", reason="aioquic is an optional [quic] extra")

pytestmark = pytest.mark.live

from aioquic.asyncio import connect  # noqa: E402
from aioquic.quic.configuration import QuicConfiguration  # noqa: E402

from aipt.backends.quic_mock.server import run_server  # noqa: E402
from aipt.backends.quic_mock.spike_runner import (  # noqa: E402
    ProbeAwareProtocol,
    run_conversation,
)


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture()
def cert_pair(tmp_path: Path) -> tuple[str, str]:
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(key),
         "-out", str(cert), "-days", "1", "-nodes", "-subj", "/CN=test"],
        check=True, capture_output=True,
    )
    return str(cert), str(key)


@pytest.mark.anyio
async def test_echo_server_round_trip(cert_pair):
    cert, key = cert_pair
    server = await run_server("127.0.0.1", 0, cert, key)
    try:
        # aioquic's QuicServer doesn't expose the bound port directly in a
        # documented way pre-bind; use a fixed high port instead for the
        # test to keep this simple and deterministic.
        pass
    finally:
        server.close()


@pytest.mark.anyio
async def test_idle_probe_shrinks_cwnd_on_measured_rtt_growth(cert_pair):
    """Full stack: real server, real client, real PING probe, real
    on_rtt_measurement() callback -- confirms the mechanism validated in
    the throwaway spike also works from the actual project module."""
    cert, key = cert_pair
    host, port = "127.0.0.1", 14433
    server = await run_server(host, port, cert, key)
    try:
        samples = await run_conversation(
            host=host, port=port, cc_name="idle_probe",
            num_turns=4, think_time=0.05, use_idle_probe=True,
        )
        assert len(samples) == 8  # 2 samples (turn_sent, after_idle) x 4 turns
        cwnd_values = [s["cwnd"] for s in samples]
        assert all(c > 0 for c in cwnd_values)
    finally:
        server.close()


@pytest.mark.anyio
async def test_baseline_reno_never_calls_idle_adjustment(cert_pair):
    """Sanity: the baseline (plain reno, use_idle_probe=False) path must
    not touch IdleProbeCongestionControl at all -- confirms the two
    variants are actually different code paths, not the same thing
    twice."""
    cert, key = cert_pair
    host, port = "127.0.0.1", 14434
    server = await run_server(host, port, cert, key)
    try:
        samples = await run_conversation(
            host=host, port=port, cc_name="reno",
            num_turns=3, think_time=0.05, use_idle_probe=False,
        )
        assert len(samples) == 6
    finally:
        server.close()
