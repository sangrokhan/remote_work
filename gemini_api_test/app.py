"""Flask app: serves UI, runs the Vertex experiment, exposes history."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from flask import Flask, abort, jsonify, render_template, request, send_file

import capture as pcap
from experiment import run_experiment
from gemini_client import ready, is_mock, ENDPOINT, PROJECT, LOCATION
from metrics import summarize
from store import save_run, list_runs, aggregate, firestore_active

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
    )


@app.route("/run", methods=["POST"])
def run():
    ok, reason = ready()
    if not ok:
        return jsonify({"error": reason}), 400

    data = request.get_json(force=True, silent=True) or {}
    turns = max(1, min(int(data.get("turns", 10)), 100))
    message_chars = int(data.get("message_chars", 500))
    model = data.get("model", "gemini-2.0-flash")
    want_capture = bool(data.get("capture", False))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    capture_info = None
    if want_capture:
        cap_ok, cap_reason = pcap.available()
        if not cap_ok:
            capture_info = {"ok": False, "error": cap_reason}
            experiment = run_experiment(turns, message_chars, model)
        else:
            with pcap.Capture(timestamp) as cap:
                experiment = run_experiment(turns, message_chars, model)
            capture_info = cap.result()
    else:
        experiment = run_experiment(turns, message_chars, model)

    summary = summarize(experiment)
    saved = save_run(timestamp, experiment, summary)

    resp = {"timestamp": timestamp, "saved_to": saved,
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


@app.route("/history")
def history():
    return jsonify({"runs": list_runs(), "aggregate": aggregate()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
