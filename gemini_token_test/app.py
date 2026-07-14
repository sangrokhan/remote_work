"""Flask app: serves UI, runs the Vertex experiment, exposes history."""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
from datetime import datetime, timezone

from flask import (
    Flask, Response, abort, jsonify, render_template, request, send_file,
)

import capture as pcap
import inspector
import netdiag
import probe
from experiment import run_experiment, run_three_stage, run_comparison, MODES, COMPARE_ARMS
from gemini_client import (
    ready, is_mock, api_host, api_key, DEFAULT_MODEL, list_models,
)
from metrics import summarize, summarize_three_stage, summarize_comparison
from store import save_run, list_runs, get_run, firestore_active, delete_run

THREE_STAGE = "caching-3stage"
# A run of 100 turns on the stateless arm already sends the system prompt 100 times;
# anything past that is a bill, not an experiment.
MAX_TURNS = 100
# Longest gap the UI may ask for between stages/arms. The pause exists to stay under
# per-minute quota, not to park the process for an hour.
MAX_PAUSE_SECONDS = 600.0

app = Flask(__name__)


# --- Request parsing shared by every run endpoint ---------------------------

def _turns(data: dict) -> int:
    """Default 1: a single-turn smoke query. The UI raises it to send more steps."""
    return max(1, min(int(data.get("turns", 1)), MAX_TURNS))


def _model(data: dict) -> str:
    return (data.get("model") or DEFAULT_MODEL).strip()


def _pause_seconds(data: dict, default: float = 0.0) -> float:
    """A mock run hits no quota, so spacing the calls apart would only waste the
    operator's time. Clamp to a sane range otherwise."""
    if is_mock():
        return 0.0
    raw = data.get("pause_seconds")
    # An absent field means "use the default"; an empty one is the UI sending a
    # blank box, which is not a number and must not blow up the run either.
    value = default if raw is None or raw == "" else float(raw)
    return max(0.0, min(value, MAX_PAUSE_SECONDS))


def _new_exec() -> tuple[str, str]:
    """(timestamp, exec_id) for one run. The id carries the timestamp plus enough
    entropy that two runs started in the same second cannot collide."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return timestamp, f"exec_{timestamp.replace(':', '-')}_{secrets.token_hex(4)}"


@app.route("/")
def index():
    """The page reports Developer API readiness. Every arm of the comparison runs
    on generativelanguage with an API key, so Vertex project/location would only
    tell the operator to check the wrong thing when a call fails."""
    ok, reason = ready()
    # A restricted-VIP route refuses every call with a 403 that reads like a bad
    # key. Say so up front rather than letting the operator rotate a working key.
    diag = netdiag.diagnose(api_host()) if not is_mock() else {"reachable": True}
    cap_ok, cap_reason = pcap.available()
    return render_template(
        "index.html",
        ready=ok,
        reason=reason,
        mock=is_mock(),
        api_host=api_host(),
        key_set=bool(api_key()),
        firestore=firestore_active(),
        default_model=DEFAULT_MODEL,
        arms=COMPARE_ARMS,
        diag=diag,
        capture_ok=cap_ok,
        capture_reason=cap_reason,
    )


@app.route("/models")
def models():
    """The catalog, plus what the probe learned about the interaction arm.

    The catalog decides two arms (generateContent, createCachedContent) and is
    silent on the third -- no interactions method is ever advertised -- so a model
    is only fully "ready" once the probe has answered for it. Unprobed stays
    unknown; it is never assumed either way.
    """
    out = list_models()
    verdicts = probe.interaction_verdicts()
    for m in out["models"]:
        v = verdicts.get(m["id"])
        m["can_interact"] = None if v is None else (v == "supported")
        if m["can_interact"] is True:
            m["label"] = f"{m['id']} — all 3 arms (cache + interaction verified)"
        elif m["can_interact"] is False:
            m["label"] = f"{m['id']} — no interaction arm ({v})"
        # can_cache=False already says which arm breaks; leave that label alone.
        m["all_arms_ready"] = bool(m["comparison_ready"] and m["can_interact"])
    out["probed"] = bool(verdicts)
    return jsonify(out)


@app.route("/diagnose")
def diagnose():
    """Where the API host actually resolves. A 403 from a restricted-VIP route
    reads exactly like a rejected key; this is what tells them apart."""
    return jsonify(netdiag.diagnose(api_host()))


def _execute_three_stage(data: dict, model: str, turns: int, want_capture: bool,
                         timestamp: str, exec_id: str, on_progress=None):
    """The 3-stage caching pipeline: stateless scenario -> caches -> stateful replay."""
    cap_ok, cap_reason = pcap.available() if want_capture else (True, "")
    # Pause between stages to stay under Vertex per-minute quotas; the UI value wins
    # over the STAGE_PAUSE_SECONDS default.
    pause = _pause_seconds(data, float(os.environ.get("STAGE_PAUSE_SECONDS", "60")))

    experiment = run_three_stage(model, turns=turns, timestamp=timestamp,
                                 want_capture=(want_capture and cap_ok),
                                 on_progress=on_progress, stage_pause_seconds=pause)
    experiment["params"]["mock"] = is_mock()
    summary = summarize_three_stage(experiment)
    saved = save_run(exec_id, timestamp, experiment, summary)

    resp = {"exec_id": exec_id, "timestamp": timestamp, "saved_to": saved,
            "mock": is_mock(), "mode": THREE_STAGE, "params": experiment["params"],
            "summary": summary, "pcaps": experiment.get("pcaps") or {},
            "comparison": build_comparison(experiment)}
    if want_capture and not cap_ok:
        resp["capture_unavailable"] = cap_reason
    return resp, 200


def _execute_run(data: dict, on_progress=None):
    """Run one experiment and build the response dict. on_progress(event) is
    called per turn (event: {stage, turn, turns}) so callers can stream progress.
    Returns (resp_dict, status_code)."""
    ok, reason = ready()
    if not ok:
        return {"error": reason}, 400

    mode = data.get("mode", "stateless")
    if mode != THREE_STAGE and mode not in MODES:
        mode = "stateless"
    turns, model = _turns(data), _model(data)
    want_capture = bool(data.get("capture", False))
    timestamp, exec_id = _new_exec()

    if mode == THREE_STAGE:
        return _execute_three_stage(data, model, turns, want_capture,
                                    timestamp, exec_id, on_progress)

    def run():
        return run_experiment(mode, model, turns=turns, on_progress=on_progress)

    capture_info = None
    if want_capture:
        cap_ok, cap_reason = pcap.available()
        if not cap_ok:
            capture_info = {"ok": False, "error": cap_reason}
            experiment = run()
        else:
            with pcap.Capture(timestamp, mode) as cap:
                experiment = run()
            capture_info = cap.result()
    else:
        experiment = run()

    # Mark synthetic runs so the result (and saved history) can't be mistaken
    # for real traffic.
    experiment["params"]["mock"] = is_mock()
    summary = summarize(experiment)
    saved = save_run(exec_id, timestamp, experiment, summary)

    resp = {"exec_id": exec_id, "timestamp": timestamp, "saved_to": saved,
            "mock": is_mock(), "mode": mode,
            "params": experiment["params"], "summary": summary}
    if capture_info is not None:
        if capture_info.get("ok") and capture_info.get("file"):
            capture_info["download"] = f"/download/pcap/{capture_info['file']}"
        resp["capture"] = capture_info
    return resp, 200


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True, silent=True) or {}
    resp, code = _execute_run(data)
    return jsonify(resp), code


def _execute_interaction(data: dict, on_progress=None):
    """Replay the scenario over the stateful Interactions API. Same contract as
    _execute_run: returns (resp_dict, status_code)."""
    ok, reason = ready()
    if not ok:
        return {"error": reason}, 400

    turns, model = _turns(data), _model(data)
    timestamp, exec_id = _new_exec()

    from interaction_client import run_interaction
    experiment = run_interaction(model, turns=turns, on_progress=on_progress)
    experiment["params"]["mock"] = is_mock()
    saved = save_run(exec_id, timestamp, experiment, {"mode": "interaction"})
    return {"exec_id": exec_id, "timestamp": timestamp, "mock": is_mock(),
            "mode": "interaction", "params": experiment["params"],
            "records": experiment["interaction_records"], "saved_to": saved}, 200


def _execute_compare(data: dict, on_progress=None):
    """Replay one scenario across the comparison arms (stateless / cached /
    interaction, plus optional nocontext) and summarize per-arm wire bytes,
    input tokens, and latency. Same (resp, code) contract as _execute_run."""
    ok, reason = ready()
    if not ok:
        return {"error": reason}, 400

    turns, model = _turns(data), _model(data)
    arms = data.get("arms") or list(COMPARE_ARMS)
    arms = [a for a in arms if a in COMPARE_ARMS] or list(COMPARE_ARMS)
    pause = _pause_seconds(data)

    want_capture = bool(data.get("capture", False))
    cap_ok, cap_reason = pcap.available() if want_capture else (True, "")
    timestamp, exec_id = _new_exec()

    experiment = run_comparison(model, turns=turns, arms=arms, on_progress=on_progress,
                                pause_seconds=pause,
                                want_capture=(want_capture and cap_ok),
                                timestamp=timestamp)
    experiment["params"]["mock"] = is_mock()
    summary = summarize_comparison(experiment)
    saved = save_run(exec_id, timestamp, experiment, summary)

    pcaps = {arm: {**c, "download": f"/download/pcap/{c['file']}"} if c.get("file") else c
             for arm, c in (experiment.get("pcaps") or {}).items()}
    resp = {"exec_id": exec_id, "timestamp": timestamp, "saved_to": saved,
            "mock": is_mock(), "mode": "comparison", "params": experiment["params"],
            "records": experiment["records"], "summary": summary, "pcaps": pcaps}
    if want_capture and not cap_ok:
        resp["capture_unavailable"] = cap_reason
    return resp, 200


def _sse_response(run):
    """Stream a long run as Server-Sent Events.

    `run(emit)` executes on a worker thread and returns (payload, status_code);
    `emit(event)` publishes a progress event. Clients see any number of
    `progress` events, then exactly one `done` (or `error`). A ": keepalive"
    comment every 15s stops proxies idling out a slow run.
    """
    import queue
    import threading

    q: "queue.Queue" = queue.Queue()

    def work():
        try:
            payload, code = run(lambda ev: q.put({"type": "progress", **ev}))
            q.put({"type": "done", "status": code, "payload": payload})
        except Exception as exc:  # never leak a half-stream; report and end
            q.put({"type": "error", "error": str(exc)})
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def gen():
        while True:
            try:
                ev = q.get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if ev is None:
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/run/stream", methods=["POST"])
def run_stream():
    """Same as /run but streams per-turn progress, then a final 'done' event."""
    data = request.get_json(force=True, silent=True) or {}
    return _sse_response(lambda emit: _execute_run(data, on_progress=emit))


@app.route("/compare", methods=["POST"])
def compare():
    """Run the stateless-vs-cached-vs-interaction comparison in one shot."""
    data = request.get_json(force=True, silent=True) or {}
    resp, code = _execute_compare(data)
    return jsonify(resp), code


@app.route("/compare/stream", methods=["POST"])
def compare_stream():
    """Same as /compare but streams per-arm progress, then a final 'done' event."""
    data = request.get_json(force=True, silent=True) or {}
    return _sse_response(lambda emit: _execute_compare(data, on_progress=emit))


@app.route("/interaction/probe", methods=["GET", "POST"])
def interaction_probe():
    """Which Interactions API surface, if any, serves a plain Gemini model?

    Runs a matrix of small live calls and reports the raw verdicts. Nothing is
    persisted. The page probes on load, so the result is cached; POST {"force":true}
    re-runs it for real.
    """
    data = request.get_json(force=True, silent=True) or {}
    return jsonify(probe.probe_cached(force=bool(data.get("force"))))


@app.route("/interaction/test", methods=["POST"])
def interaction_test():
    """Experimental: the scenario over the Interactions API, streamed. Additive —
    does not touch the generateContent run flow."""
    data = request.get_json(force=True, silent=True) or {}
    return _sse_response(lambda emit: _execute_interaction(data, on_progress=emit))


def _attachment(body: str, mimetype: str, filename: str):
    """A downloadable response. Flask appends charset=utf-8 to the mimetype."""
    return Response(body, mimetype=mimetype,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def _json_attachment(payload, filename: str):
    return _attachment(json.dumps(payload, indent=2, ensure_ascii=False),
                       "application/json", filename)


def _csv_attachment(header: list, rows: list, filename: str):
    """A downloadable CSV. Every cell goes through _csv_safe, and the file opens
    with a BOM so Excel reads it as UTF-8 instead of guessing."""
    buf = io.StringIO()
    buf.write("﻿")
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in rows:
        writer.writerow([_csv_safe(v) for v in row])
    return _attachment(buf.getvalue(), "text/csv", filename)


def _csv_safe(v):
    # Neutralize CSV formula injection: a cell a spreadsheet would treat as a
    # formula (leading = + - @ TAB CR) is prefixed with a quote so it stays text.
    s = "" if v is None else str(v)
    if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
        s = "'" + s
    return s


def _parse_json(s):
    """Turn a raw JSON string back into an object for a clean export; keep the
    original string if it isn't valid JSON (e.g. an error body)."""
    if not isinstance(s, str) or not s:
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def _chat_turns(records, stage):
    out = []
    for r in records or []:
        out.append({
            "turn": r.get("turn"),
            "stage": stage,
            "question": r.get("question", ""),
            "answer": r.get("response_text", ""),
            "request": _parse_json(r.get("request_json", "")),
            "response": _parse_json(r.get("response_json", "")),
            "error": r.get("error", ""),
        })
    return out


def build_chat_export(doc: dict) -> dict:
    """Trim a stored run down to the chat: per-step question/answer + the raw
    request/response JSON sent to and received from the server."""
    mode = doc.get("mode")
    turns = []
    if mode == THREE_STAGE:
        turns += _chat_turns(doc.get("stateless_records"), "stateless")
        turns += _chat_turns(doc.get("nocontext_records"), "nocontext")
        turns += _chat_turns(doc.get("stateful_records"), "stateful")
    else:
        turns += _chat_turns(doc.get("records"), mode)
    return {
        "exec_id": doc.get("exec_id"),
        "mode": mode,
        "timestamp": doc.get("timestamp"),
        "mock": doc.get("mock", False),
        "turns": turns,
    }


def build_comparison(doc: dict) -> list[dict]:
    """Per-step side-by-side rows for the 3-stage run: the query plus the
    stateless and stateful responses, matched by turn."""
    sl = {r.get("turn"): r for r in doc.get("stateless_records") or []}
    nc = {r.get("turn"): r for r in doc.get("nocontext_records") or []}
    sf = {r.get("turn"): r for r in doc.get("stateful_records") or []}
    rows = []
    for turn in sorted(set(sl) | set(nc) | set(sf), key=lambda t: (t is None, t)):
        q = (sl.get(turn) or nc.get(turn) or sf.get(turn) or {}).get("question", "")
        rows.append({
            "turn": turn,
            "query": q,
            "stateless_response": (sl.get(turn) or {}).get("response_text", ""),
            "nocontext_response": (nc.get(turn) or {}).get("response_text", ""),
            "stateful_response": (sf.get(turn) or {}).get("response_text", ""),
        })
    return rows


COMPARE_COLUMNS = ["turn", "query", "stateless_response",
                   "nocontext_response", "stateful_response"]


@app.route("/download/compare/<exec_id>")
def download_compare(exec_id):
    """CSV of the per-step comparison (turn, query, stateless, stateful)."""
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    rows = [[r[c] for c in COMPARE_COLUMNS] for r in build_comparison(doc)]
    return _csv_attachment(COMPARE_COLUMNS, rows, f"compare_{exec_id}.csv")


CASE_COLUMNS = ["arm", "phase", "turn", "wire_sent", "wire_recv", "elapsed_ms",
                "input_tokens", "cached_tokens", "output_tokens", "thought_tokens",
                "total_tokens", "error"]


def build_comparison_cases(doc: dict) -> list[dict]:
    """One entry per call the comparison made, with the raw bodies parsed back into
    objects. This is what makes a run auditable: you can read what each arm actually
    sent and what came back, and check the arms answered the same questions."""
    return [{
        "arm": r.get("arm"),
        "phase": r.get("phase"),
        "turn": r.get("turn"),
        "question": r.get("question", ""),
        "response_text": r.get("response_text", ""),
        "wire_sent": r.get("wire_sent", 0),
        "wire_recv": r.get("wire_recv", 0),
        "elapsed_ms": r.get("elapsed_ms", 0),
        "input_tokens": r.get("input_tokens", 0),
        "cached_tokens": r.get("cached_tokens", 0),
        "output_tokens": r.get("output_tokens", 0),
        "request": _parse_json(r.get("request_raw", "")),
        "response": _parse_json(r.get("response_raw", "")),
        "error": r.get("error", ""),
    } for r in doc.get("records") or []]


@app.route("/download/comparison/<exec_id>.json")
def download_comparison_json(exec_id):
    """Every case of a comparison run: raw request, raw response, metrics."""
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    payload = {
        "exec_id": doc.get("exec_id"),
        "timestamp": doc.get("timestamp"),
        "mock": doc.get("mock", False),
        "params": doc.get("params", {}),
        "summary": doc.get("summary", {}),
        "cases": build_comparison_cases(doc),
    }
    return _json_attachment(payload, f"comparison_{exec_id}.json")


def build_responses_table(doc: dict) -> tuple[list, list]:
    """One row per step, one column per arm. Returns (header, rows).

    The metrics CSV says what each arm spent; it cannot say whether the arms were
    having the same conversation. An arm that quietly degraded -- a cache that never
    hit, a history the server dropped -- still produces perfectly reasonable-looking
    bytes, and the only way to catch it is to read the answers side by side.

    cachegen rows are left out: the cache builds answer nothing, so a row for one
    would be a step that never happened.
    """
    arms = [a for a in (doc.get("params") or {}).get("arms") or []]
    steady = [r for r in doc.get("records") or [] if r.get("phase") == "steady"]
    by_arm = {a: {r["turn"]: r for r in steady if r["arm"] == a} for a in arms}
    turns = sorted({r["turn"] for r in steady})

    header = (["turn", "question"]
              + [f"{a}_response" for a in arms]
              + [f"{a}_request" for a in arms])
    rows = []
    for t in turns:
        present = [by_arm[a].get(t) for a in arms]
        question = next((r["question"] for r in present if r and r.get("question")), "")
        rows.append([t, question]
                    + [(r or {}).get("response_text", "") for r in present]
                    + [(r or {}).get("request_raw", "") for r in present])
    return header, rows


@app.route("/download/comparison/<exec_id>-responses.csv")
def download_comparison_responses(exec_id):
    """Answers side by side: one row per step, one column per arm."""
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    header, rows = build_responses_table(doc)
    return _csv_attachment(header, rows, f"responses_{exec_id}.csv")


@app.route("/download/comparison/<exec_id>.csv")
def download_comparison_csv(exec_id):
    """Flat metrics table: one row per call. For spreadsheets, not for reading raw
    bodies -- those are in the JSON export."""
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    rows = [[r.get(c) for c in CASE_COLUMNS] for r in doc.get("records") or []]
    return _csv_attachment(CASE_COLUMNS, rows, f"comparison_{exec_id}.csv")


@app.route("/download/chat/<exec_id>")
def download_chat(exec_id):
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    return _json_attachment(build_chat_export(doc), f"chat_{exec_id}.json")


@app.route("/download/pcap/<path:name>")
def download_pcap(name):
    path = pcap.safe_pcap_path(name)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=name, mimetype="application/vnd.tcpdump.pcap")


@app.route("/inspect", methods=["POST"])
def inspect_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "url required"}), 400

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    record = inspector.inspect(
        method=data.get("method", "GET"),
        url=url,
        headers_raw=data.get("headers", ""),
        body=data.get("body", ""),
        include_bodies=bool(data.get("include_bodies", False)),
        allow_private=bool(data.get("allow_private", False)),
        timestamp=timestamp,
    )
    name = inspector.save_transcript(timestamp, record)
    if name:
        record["download"] = f"/download/transcript/{name}"
    status = 200 if record.get("ok") else 400
    return jsonify(record), status


@app.route("/download/transcript/<path:name>")
def download_transcript(name):
    path = inspector.safe_transcript_path(name)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True,
                     download_name=name, mimetype="application/json")


@app.route("/history")
def history():
    return jsonify(list_runs())


@app.route("/history/<exec_id>")
def history_one(exec_id):
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    return jsonify(doc)


@app.route("/history/<exec_id>", methods=["DELETE"])
def history_delete(exec_id):
    res = delete_run(exec_id)
    return jsonify(res), (200 if res.get("ok") else 400)


@app.route("/download/run/<exec_id>")
def download_run(exec_id):
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    return _json_attachment(doc, f"{exec_id}.json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
