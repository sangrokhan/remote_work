"""Flask app: serves UI, runs the Vertex experiment, exposes history."""

from __future__ import annotations

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
from store import save_run, list_runs, get_run, firestore_active

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


@app.route("/run", methods=["POST"])
def run():
    ok, reason = ready()
    if not ok:
        return jsonify({"error": reason}), 400

    data = request.get_json(force=True, silent=True) or {}
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
        experiment = run_three_stage(model, turns=turns)
        experiment["params"]["mock"] = is_mock()
        summary = summarize_three_stage(experiment)
        saved = save_run(exec_id, timestamp, experiment, summary)
        return jsonify({"exec_id": exec_id, "timestamp": timestamp,
                        "saved_to": saved, "mock": is_mock(), "mode": mode,
                        "params": experiment["params"], "summary": summary})

    capture_info = None
    if want_capture:
        cap_ok, cap_reason = pcap.available()
        if not cap_ok:
            capture_info = {"ok": False, "error": cap_reason}
            experiment = run_experiment(mode, model, turns=turns)
        else:
            with pcap.Capture(timestamp, mode) as cap:
                experiment = run_experiment(mode, model, turns=turns)
            capture_info = cap.result()
    else:
        experiment = run_experiment(mode, model, turns=turns)

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
    return jsonify(resp)


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
