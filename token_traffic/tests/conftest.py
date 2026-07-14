"""Test-suite defaults: nothing live, nothing written outside a temp directory.

Both provider keys sit behind a monthly spend cap, and a test run that quietly bills
one of them is a bug that pays for itself. So the environment is pinned to mock before
anything imports a provider, and every path a run could be written to is redirected
into a per-session temp directory -- a test that saves a run must not be able to land
it in the operator's real history, where it would be indistinguishable from a live
measurement and would silently take a retention slot from one.

The env is set at import time rather than in a fixture, because collection imports the
modules under test and a module that reads its configuration at import must never see
the real one.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="token_traffic_tests_"))

# Mock by default: a test asks for the network by starting its own localhost server,
# never by reaching for a provider.
os.environ.setdefault("TRAFFIC_MOCK", "1")
os.environ["GEMINI_MOCK"] = "1"
os.environ["OPENAI_MOCK"] = "1"

# Every place a run, a capture, or a download could be written. The names must be the
# ones the modules actually read (core.store: TRAFFIC_DATA_DIR, core.capture:
# TRAFFIC_PCAP_DIR) -- a redirect under a name nobody reads leaves the real directory
# live and reads, from here, exactly like a redirect that worked.
os.environ["TRAFFIC_DATA_DIR"] = str(_TMP / "data" / "runs")
os.environ["TRAFFIC_PCAP_DIR"] = str(_TMP / "data" / "pcaps")
# No test may start tcpdump: capture is exercised against its own fakes, and a real
# one here would spawn a process and write packets on the operator's interface.
os.environ["TRAFFIC_PCAP_DISABLE"] = "1"

# A key left in the ambient shell would make a mock-off slip cost money instead of
# failing loudly, which is the wrong direction for that mistake to fall.
for _key in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_key, None)


@pytest.fixture(autouse=True)
def _fresh_session():
    """Each test gets a clean connection pool.

    The counting session is a module global with pooled sockets in it. A socket left
    open by one test is a socket the next test's byte counts would inherit, and the
    resulting flake would look like a counting bug rather than a test bug.
    """
    from core import wire

    wire.reset_session()
    yield
    wire.reset_session()
