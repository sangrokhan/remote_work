"""config.py: the single place all AIPT env-var parsing lives.

Merges test coverage from:
  - token_traffic/tests/test_config.py's mock-switch tests (`flag`/`is_mock`),
    trimmed to the parts that exercise aipt.core.config directly rather than
    the (not-yet-migrated) provider/app layer.
  - new tests for `flag_any` (the offload env-alias helper) and `env_int`
    (generalized from the ad hoc int-env parsing that used to live inline in
    tcp_congestion's cwnd.py).

There used to be two readings of "is this call synthetic": `TRAFFIC_MOCK=true`
satisfied one provider's parser and not the other's, so half a run was
synthetic and half of it was billed -- and the run was then filed in the
*live* bucket, because the flag that decides the bucket had its own third
reading. These tests exist to keep the readings from drifting apart again.
"""

from __future__ import annotations

import pytest

from aipt.core import config


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    for var in ("TRAFFIC_MOCK", "GEMINI_MOCK", "OPENAI_MOCK",
                "NIC_OFFLOAD_DISABLE", "TRAFFIC_PCAP_NO_OFFLOAD",
                "SOME_INT_VAR"):
        monkeypatch.delenv(var, raising=False)


# --- flag() -------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_flag_recognizes_truthy_strings(value, monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", value)
    assert config.flag("TRAFFIC_MOCK") is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_flag_recognizes_falsy_strings(value, monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", value)
    assert config.flag("TRAFFIC_MOCK") is False


def test_flag_missing_env_is_false():
    assert config.flag("SOME_VAR_THAT_IS_NOT_SET") is False


# --- flag_any() (absorbed offload env-alias need) ------------------------

def test_flag_any_true_when_canonical_set(monkeypatch):
    monkeypatch.setenv("NIC_OFFLOAD_DISABLE", "1")
    assert config.flag_any("NIC_OFFLOAD_DISABLE", "TRAFFIC_PCAP_NO_OFFLOAD") is True


def test_flag_any_true_when_only_deprecated_alias_set(monkeypatch):
    monkeypatch.setenv("TRAFFIC_PCAP_NO_OFFLOAD", "1")
    assert config.flag_any("NIC_OFFLOAD_DISABLE", "TRAFFIC_PCAP_NO_OFFLOAD") is True


def test_flag_any_false_when_none_set():
    assert config.flag_any("NIC_OFFLOAD_DISABLE", "TRAFFIC_PCAP_NO_OFFLOAD") is False


# --- env_int() (absorbed from tcp_congestion cwnd.py's ad hoc parsing) ---

def test_env_int_parses_valid_int(monkeypatch):
    monkeypatch.setenv("SOME_INT_VAR", "42")
    assert config.env_int("SOME_INT_VAR", default=7) == 42


def test_env_int_falls_back_on_missing():
    assert config.env_int("SOME_INT_VAR", default=7) == 7


def test_env_int_falls_back_on_garbage(monkeypatch):
    monkeypatch.setenv("SOME_INT_VAR", "banana")
    assert config.env_int("SOME_INT_VAR", default=7) == 7


def test_env_int_falls_back_on_blank(monkeypatch):
    monkeypatch.setenv("SOME_INT_VAR", "   ")
    assert config.env_int("SOME_INT_VAR", default=7) == 7


# --- is_mock() ------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_is_mock_true_for_truthy_traffic_mock(value, monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", value)
    assert config.is_mock() is True
    assert config.is_mock("gemini") is True
    assert config.is_mock("openai") is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_is_mock_false_for_falsy_traffic_mock(value, monkeypatch):
    monkeypatch.setenv("TRAFFIC_MOCK", value)
    assert config.is_mock() is False


def test_one_provider_can_be_mocked_while_the_other_is_live(monkeypatch):
    # A Gemini-only key should still be able to exercise the OpenAI arms.
    monkeypatch.setenv("OPENAI_MOCK", "1")
    assert config.is_mock("openai") is True
    assert config.is_mock("gemini") is False


def test_is_mock_with_no_provider_ignores_per_provider_flags(monkeypatch):
    monkeypatch.setenv("OPENAI_MOCK", "1")
    assert config.is_mock() is False
