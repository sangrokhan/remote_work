"""Flask app: serves UI, runs the Vertex experiment, exposes history."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from experiment import run_experiment
from gemini_client import ready, is_mock, ENDPOINT, PROJECT, LOCATION
from metrics import summarize
from store import save_run, list_runs, aggregate, firestore_active

app = Flask(__name__)


@app.route("/")
def index():
    ok, reason = ready()
    return render_template(
        "index.html",
        ready=ok,
        reason=reason,
        mock=is_mock(),
        endpoint=ENDPOINT,
        project=PROJECT or "(unset)",
        location=LOCATION,
        firestore=firestore_active(),
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

    experiment = run_experiment(turns, message_chars, model)
    summary = summarize(experiment)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    saved = save_run(timestamp, experiment, summary)

    return jsonify({"timestamp": timestamp, "saved_to": saved,
                    "params": experiment["params"], "summary": summary})


@app.route("/history")
def history():
    return jsonify({"runs": list_runs(), "aggregate": aggregate()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
