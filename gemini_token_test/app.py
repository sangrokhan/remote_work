"""Flask app: serves UI, runs the Vertex experiment, exposes history."""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone

from flask import Flask, abort, jsonify, render_template, request, send_file

import capture as pcap
import inspector
from experiment import run_experiment, run_three_stage, MODES
from gemini_client import (
    ready, is_mock, ENDPOINT, PROJECT, LOCATION, DEFAULT_MODEL, list_models,
)
from metrics import summarize, summarize_three_stage

THREE_STAGE = "caching-3stage"
from store import save_run, list_runs, get_run, firestore_active, delete_run

app = Flask(__name__)


@app.route("/")
def index():
    ok, reason = ready()
    cap_ok, cap_reason = pcap.available()
    return render_template(
        "index.html",
        ready=ok,
        reason=reason,
        mock=is_mock(),
        endpoint=ENDPOINT,
        project=PROJECT or "(unset)",
        location=LOCATION,
        firestore=firestore_active(),
        capture_ok=cap_ok,
        capture_reason=cap_reason,
        default_model=DEFAULT_MODEL,
        modes=(*MODES, THREE_STAGE),
    )


@app.route("/models")
def models():
    return jsonify(list_models())


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
    # Default 1 turn: a single-turn smoke query for initial testing. Raise it in
    # the UI to send more steps.
    turns = max(1, min(int(data.get("turns", 1)), 100))
    model = (data.get("model") or DEFAULT_MODEL).strip()
    want_capture = bool(data.get("capture", False))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    exec_id = f"exec_{timestamp.replace(':', '-')}_{secrets.token_hex(4)}"

    # 3-stage caching pipeline: stateless scenario -> caches -> stateful replay.
    if mode == THREE_STAGE:
        cap_ok, cap_reason = (True, "")
        if want_capture:
            cap_ok, cap_reason = pcap.available()
        # Pause between stages to stay under Vertex per-minute quotas; skip in mock
        # (no real quota) so tests/dev runs don't sleep. UI value wins over the
        # STAGE_PAUSE_SECONDS default; clamped to a sane 0..600s.
        if is_mock():
            pause = 0
        else:
            raw = data.get("pause_seconds")
            pause = float(raw) if raw is not None else float(os.environ.get("STAGE_PAUSE_SECONDS", "60"))
            pause = max(0.0, min(pause, 600.0))
        experiment = run_three_stage(model, turns=turns, timestamp=timestamp,
                                     want_capture=(want_capture and cap_ok),
                                     on_progress=on_progress, stage_pause_seconds=pause)
        experiment["params"]["mock"] = is_mock()
        summary = summarize_three_stage(experiment)
        saved = save_run(exec_id, timestamp, experiment, summary)
        resp = {"exec_id": exec_id, "timestamp": timestamp, "saved_to": saved,
                "mock": is_mock(), "mode": mode, "params": experiment["params"],
                "summary": summary, "pcaps": experiment.get("pcaps") or {},
                "comparison": build_comparison(experiment)}
        if want_capture and not cap_ok:
            resp["capture_unavailable"] = cap_reason
        return resp, 200

    capture_info = None
    if want_capture:
        cap_ok, cap_reason = pcap.available()
        if not cap_ok:
            capture_info = {"ok": False, "error": cap_reason}
            experiment = run_experiment(mode, model, turns=turns, on_progress=on_progress)
        else:
            with pcap.Capture(timestamp, mode) as cap:
                experiment = run_experiment(mode, model, turns=turns, on_progress=on_progress)
            capture_info = cap.result()
    else:
        experiment = run_experiment(mode, model, turns=turns, on_progress=on_progress)

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


@app.route("/run/stream", methods=["POST"])
def run_stream():
    """Same as /run but streams per-turn progress as Server-Sent Events, then a
    final 'done' event carrying the full result (or 'error')."""
    import queue
    import threading
    from flask import Response

    data = request.get_json(force=True, silent=True) or {}
    q: "queue.Queue" = queue.Queue()

    def work():
        try:
            resp, code = _execute_run(data, on_progress=lambda ev: q.put({"type": "progress", **ev}))
            q.put({"type": "done", "status": code, "payload": resp})
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
                yield ": keepalive\n\n"  # keep proxies from idle-timing out the stream
                continue
            if ev is None:
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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


@app.route("/download/compare/<exec_id>")
def download_compare(exec_id):
    """CSV of the per-step comparison (turn, query, stateless, stateful)."""
    import csv
    import io
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    rows = build_comparison(doc)

    def _csv_safe(v):
        # Neutralize CSV formula injection: a cell a spreadsheet would treat as a
        # formula (leading = + - @ TAB CR) is prefixed with a quote so it stays text.
        s = "" if v is None else str(v)
        if s and s[0] in ("=", "+", "-", "@", "\t", "\r"):
            s = "'" + s
        return s

    buf = io.StringIO()
    buf.write("﻿")  # BOM: nudge Excel toward UTF-8
    writer = csv.writer(buf)
    writer.writerow(["turn", "query", "stateless_response",
                     "nocontext_response", "stateful_response"])
    for r in rows:
        writer.writerow([_csv_safe(r["turn"]), _csv_safe(r["query"]),
                         _csv_safe(r["stateless_response"]),
                         _csv_safe(r["nocontext_response"]),
                         _csv_safe(r["stateful_response"])])
    from flask import Response
    return Response(
        buf.getvalue(),
        mimetype="text/csv",  # Flask appends charset=utf-8
        headers={"Content-Disposition": f'attachment; filename="compare_{exec_id}.csv"'},
    )


@app.route("/download/chat/<exec_id>")
def download_chat(exec_id):
    doc = get_run(exec_id)
    if doc is None:
        abort(404)
    from flask import Response
    payload = build_chat_export(doc)
    return Response(
        json.dumps(payload, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="chat_{exec_id}.json"'},
    )


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
    from flask import Response
    import json as _json
    return Response(
        _json.dumps(doc, indent=2), mimetype="application/json",
        headers={"Content-Disposition": f'attachment; filename="{exec_id}.json"'},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
