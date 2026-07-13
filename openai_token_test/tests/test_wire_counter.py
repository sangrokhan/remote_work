"""The byte counter must actually count bytes, or every number downstream is a lie."""

from __future__ import annotations

import json

import pytest

import wire
from fake_openai import FakeOpenAI


@pytest.fixture
def fake(monkeypatch):
    srv = FakeOpenAI()
    base = srv.start()
    monkeypatch.setenv("OPENAI_BASE_URL", base)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    wire.reset_session()
    yield srv
    srv.stop()
    wire.reset_session()


def test_counts_request_body_plus_headers(fake):
    payload = json.dumps({"blob": "x" * 5000}).encode()
    sess = wire.session()

    with wire.wire_counter() as w:
        resp = sess.post(f"{_base(fake)}/responses", data=payload,
                         headers={"Content-Type": "application/json"})

    assert resp.status_code == 200
    # sent must cover the body, and exceed it by the request line + headers
    assert w.sent > len(payload)
    assert w.sent - len(payload) < 1000, "header overhead should be modest, not another body"
    # the server saw exactly the body we sent
    assert fake.requests[0]["content_length"] == len(payload)


def test_counts_response_bytes(fake):
    sess = wire.session()
    with wire.wire_counter() as w:
        resp = sess.post(f"{_base(fake)}/responses", data=b"{}",
                         headers={"Content-Type": "application/json"})
    assert w.recv >= len(resp.content) > 0


def test_counter_survives_keepalive_reuse(fake):
    """Second request rides the same socket; connect() does not fire again. The
    tally must still move, or every turn after the first would read as zero."""
    sess = wire.session()
    body = json.dumps({"blob": "y" * 3000}).encode()

    with wire.wire_counter() as first:
        sess.post(f"{_base(fake)}/responses", data=body,
                  headers={"Content-Type": "application/json"})
    with wire.wire_counter() as second:
        sess.post(f"{_base(fake)}/responses", data=body,
                  headers={"Content-Type": "application/json"})

    assert first.sent > len(body)
    assert second.sent > len(body)
    assert abs(first.sent - second.sent) < 100


def test_bigger_body_counts_bigger(fake):
    sess = wire.session()
    small = json.dumps({"blob": "a" * 100}).encode()
    large = json.dumps({"blob": "a" * 100_000}).encode()

    with wire.wire_counter() as w_small:
        sess.post(f"{_base(fake)}/responses", data=small,
                  headers={"Content-Type": "application/json"})
    with wire.wire_counter() as w_large:
        sess.post(f"{_base(fake)}/responses", data=large,
                  headers={"Content-Type": "application/json"})

    assert w_large.sent - w_small.sent == pytest.approx(len(large) - len(small), abs=200)


def _base(fake) -> str:
    import os
    return os.environ["OPENAI_BASE_URL"]
