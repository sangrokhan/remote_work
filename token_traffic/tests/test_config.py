"""The mock switch, read once.

There used to be two readings of it. `TRAFFIC_MOCK=true` satisfied OpenAI's parser and
not Gemini's, so half a run was synthetic and half of it was billed -- and the run was
filed as live, because the flag that picks the bucket had a third reading. These tests
exist to keep the readings from drifting apart again.
"""

from __future__ import annotations

import pytest

from core import app as web
from core import config
from providers import base


@pytest.fixture(autouse=True)
def _no_ambient_mock(monkeypatch):
    for var in ("TRAFFIC_MOCK", "GEMINI_MOCK", "OPENAI_MOCK"):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_every_provider_agrees_that_this_means_mock(value, monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", value)
    assert base.get("gemini").is_mock() is True
    assert base.get("openai").mock_mode() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_every_provider_agrees_that_this_does_not(value, monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", value)
    assert base.get("gemini").is_mock() is False
    assert base.get("openai").mock_mode() is False


def test_one_provider_can_be_mocked_while_the_other_is_live(monkeypatch):
    # A Gemini-only key should still be able to exercise the OpenAI arms.
    monkeypatch.setenv("OPENAI_MOCK", "1")
    assert config.is_mock("openai") is True
    assert config.is_mock("gemini") is False


def test_a_run_with_any_synthetic_call_in_it_is_filed_as_mock(monkeypatch):
    # Not "live with a caveat": its OpenAI numbers were never measured, and nothing
    # must ever chart them against numbers that were.
    monkeypatch.setenv("OPENAI_MOCK", "1")
    pairs = [("gemini", "stateless"), ("openai", "chat_stateless")]
    assert web.mock_mode(pairs) is True
    assert web.mock_mode([("gemini", "stateless")]) is False
