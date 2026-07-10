"""The Interactions API request body must match the server's schema.

`tool_choice` is not a top-level field — it belongs to `generation_config`, which
only model interactions accept. Sending it at the top level made the server reply
"unknown parameter tool_choice"; sending `generation_config` on an agent
interaction is rejected the same way.
"""

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ENV_KEYS = ("INTERACTION_AGENT", "INTERACTION_TOOL_CHOICE", "INTERACTION_TOOLS")


class _FakeResponse:
    status_code = 200
    text = ""

    def iter_lines(self, decode_unicode=False):
        return iter([])


def _request_body(monkeypatch, **env):
    """Build one interaction request without touching the network."""
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    ic = importlib.reload(importlib.import_module("interaction_client"))
    monkeypatch.setattr(ic, "_bearer_token", lambda: "test-token")
    monkeypatch.setattr(
        ic, "_session",
        lambda: type("S", (), {"post": lambda self, *a, **kw: _FakeResponse()})(),
    )
    return ic._call_interaction("hi", "", {"type": "remote"}, model="gemini-2.5-flash")["request"]


def test_model_interaction_puts_tool_choice_in_generation_config(monkeypatch):
    body = _request_body(monkeypatch)
    assert "tool_choice" not in body
    assert body["generation_config"] == {"tool_choice": "none"}
    assert body["tools"] == []
    assert body["model"] == "gemini-2.5-flash"


def test_agent_interaction_sends_no_generation_config(monkeypatch):
    # Agent interactions have no generation_config, so tool use can't be constrained.
    body = _request_body(monkeypatch, INTERACTION_AGENT="antigravity-preview-05-2026")
    assert "generation_config" not in body
    assert "tool_choice" not in body
    assert body["agent"] == "antigravity-preview-05-2026"


def test_empty_tool_choice_omits_generation_config(monkeypatch):
    body = _request_body(monkeypatch, INTERACTION_TOOL_CHOICE="")
    assert "generation_config" not in body


def test_tool_choice_value_is_passed_through(monkeypatch):
    body = _request_body(monkeypatch, INTERACTION_TOOL_CHOICE="auto")
    assert body["generation_config"] == {"tool_choice": "auto"}


def test_body_is_json_serialisable(monkeypatch):
    json.loads(json.dumps(_request_body(monkeypatch)))
