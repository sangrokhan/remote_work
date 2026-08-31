"""Unit tests for aipt.core.quic_congestion -- pure availability-reporting
logic. Skipped entirely if aioquic isn't installed (optional [quic]
extra).
"""
from __future__ import annotations

import pytest

aioquic = pytest.importorskip("aioquic", reason="aioquic is an optional [quic] extra")

from aipt.core import quic_congestion  # noqa: E402


def test_available_true_when_aioquic_installed():
    ok, reason = quic_congestion.available()
    assert ok is True
    assert reason == "ok"


def test_available_algorithms_includes_stock_and_idle_probe():
    names, reason = quic_congestion.available_algorithms()
    assert reason == "ok"
    assert "reno" in names
    assert "cubic" in names
    # Importing this module registers aipt's own idle_probe algorithm as
    # a side effect (see module docstring) -- must show up too.
    assert "idle_probe" in names


def test_available_algorithms_never_empty_when_aioquic_present():
    names, _reason = quic_congestion.available_algorithms()
    assert len(names) > 0
