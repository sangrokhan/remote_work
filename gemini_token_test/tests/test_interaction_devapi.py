"""interaction_client, rebuilt for the Developer API plain-model path.

stream:true (SSE; the events reassemble into the body a blocking call would return),
system_instruction re-sent every turn (it is interaction-scoped and the server does
not keep it), previous_interaction_id chains turns. Streaming is not cosmetic: the
store cost lands ~1.8 s after the last token, and only a streamed read can tell the
answer's arrival apart from the server letting go. No agent, no environment, no
background, no warmup, no tools.

Body/usage are pure and tested directly; the call and the chaining are exercised
against a localhost server, so there is no live API in CI.
"""

import http.server
import json
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gemini_client as gc
import interaction_client as ic


# --- body shape ------------------------------------------------------------

def test_turn_one_body_has_system_and_no_previous_id():
    b = ic.interaction_body("gemini-3.1-flash-lite", "hi", "SYSTEM PROMPT", None)
    assert b["model"] == "gemini-3.1-flash-lite"
    assert b["stream"] is True
    assert b["store"] is True
    assert b["system_instruction"] == "SYSTEM PROMPT"
    assert b["input"] == [{"type": "user_input",
                           "content": [{"type": "text", "text": "hi"}]}]
    assert "previous_interaction_id" not in b
    # The plain-model comparison carries none of the agent machinery.
    for k in ("agent", "environment", "background", "tools", "generation_config"):
        assert k not in b


def test_turn_two_still_sends_system_and_adds_previous_id():
    # system_instruction is interaction-scoped, so it MUST be re-sent every turn;
    # previous_interaction_id carries only the history.
    b = ic.interaction_body("m", "next", "SYSTEM PROMPT", "int_1")
    assert b["system_instruction"] == "SYSTEM PROMPT"
    assert b["previous_interaction_id"] == "int_1"


# --- usage mapping ---------------------------------------------------------

def test_usage_maps_to_common_fields():
    u = ic._usage_common({
        "total_input_tokens": 4200, "total_cached_tokens": 4000,
        "total_output_tokens": 12, "total_thought_tokens": 5, "total_tokens": 4217,
    })
    assert u["input_tokens"] == 4200
    assert u["cached_tokens"] == 4000
    assert u["output_tokens"] == 12
    assert u["thought_tokens"] == 5
    assert u["total_tokens"] == 4217


def test_usage_missing_fields_default_to_zero():
    u = ic._usage_common({})
    assert u == {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                 "thought_tokens": 0, "total_tokens": 0}


# --- integration against a local server ------------------------------------

_SEEN = {}


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        _SEEN.setdefault("bodies", []).append(body)
        _SEEN["path"] = self.path
        _SEEN["api_key"] = self.headers.get("x-goog-api-key")
        idx = len(_SEEN["bodies"])
        # The real endpoint streams, and its completed event carries the usage but
        # not the steps -- those exist only as the deltas that went past.
        events = [
            {"event_type": "interaction.created",
             "interaction": {"id": f"int_{idx}", "status": "in_progress"}},
            {"event_type": "step.start", "index": 0, "step": {"type": "model_output"}},
            {"event_type": "step.delta", "index": 0,
             "delta": {"text": f"answer {idx}", "type": "text"}},
            {"event_type": "step.stop", "index": 0},
            {"event_type": "interaction.completed",
             "interaction": {"id": f"int_{idx}", "status": "completed",
                             "usage": {"total_input_tokens": 4200,
                                       "total_cached_tokens": 4000,
                                       "total_output_tokens": 12,
                                       "total_thought_tokens": 5,
                                       "total_tokens": 4217}}},
        ]
        out = ("".join(f"data: {json.dumps(e)}\n\n" for e in events)
               + "data: [DONE]\n\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def _server():
    _SEEN.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _env(monkeypatch, host):
    monkeypatch.delenv("GEMINI_MOCK", raising=False)
    monkeypatch.setenv("GEMINI_API_SCHEME", "http")
    monkeypatch.setenv("GEMINI_API_HOST", host)
    monkeypatch.setenv("GEMINI_API_KEY", "ik")
    gc.reset_session()


def test_call_interaction_reports_metrics_and_parses_usage(monkeypatch):
    srv = _server()
    host = f"127.0.0.1:{srv.server_address[1]}"
    try:
        _env(monkeypatch, host)
        r = ic._call_interaction("gemini-3.1-flash-lite", "hi", "SYS", None)
        assert r["error"] == ""
        assert r["response_text"] == "answer 1"
        assert r["interaction_id"] == "int_1"
        assert r["input_tokens"] == 4200
        assert r["cached_tokens"] == 4000
        assert r["wire_sent"] > len(r["request_raw"])   # headers counted
        assert r["wire_recv"] > 0
        assert r["elapsed_ms"] >= 0
        assert _SEEN["path"] == "/v1beta/interactions"
        assert _SEEN["api_key"] == "ik"
    finally:
        gc.reset_session()
        srv.shutdown()


def test_run_interaction_chains_previous_id_and_resends_system(monkeypatch):
    srv = _server()
    host = f"127.0.0.1:{srv.server_address[1]}"
    try:
        _env(monkeypatch, host)
        out = ic.run_interaction("gemini-3.1-flash-lite", request_name="perf", turns=3)
        recs = out["interaction_records"]
        assert len(recs) == 3
        bodies = _SEEN["bodies"]
        # Turn 1: no previous_interaction_id. Turns 2-3: chained to the prior id.
        assert "previous_interaction_id" not in bodies[0]
        assert bodies[1]["previous_interaction_id"] == "int_1"
        assert bodies[2]["previous_interaction_id"] == "int_2"
        # system_instruction present on every turn (interaction-scoped).
        assert all(b.get("system_instruction") for b in bodies)
    finally:
        gc.reset_session()
        srv.shutdown()
