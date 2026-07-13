"""Every call must report its own latency.

The comparison measures wall-clock per turn, but `CallResult` had no timing
field, so no generateContent latency existed to compare against the interaction
arm. This adds `elapsed_ms` and asserts it is populated even in mock mode (where
there is no network, so it is small but present and non-negative).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemini_client


def test_mock_call_reports_elapsed_ms(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    r = gemini_client.call_gemini("gemini-3.1-flash-lite",
                                  [{"role": "user", "parts": [{"text": "hi"}]}],
                                  mode="stateless", turn=1)
    assert hasattr(r, "elapsed_ms")
    assert isinstance(r.elapsed_ms, int)
    assert r.elapsed_ms >= 0


def test_elapsed_ms_survives_as_dict(monkeypatch):
    monkeypatch.setenv("GEMINI_MOCK", "1")
    r = gemini_client.call_gemini("gemini-3.1-flash-lite",
                                  [{"role": "user", "parts": [{"text": "hi"}]}],
                                  mode="stateless", turn=1)
    assert "elapsed_ms" in r.as_dict()
