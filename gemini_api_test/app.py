"""Flask app: serves UI, runs the experiment, exposes history."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from experiment import run_experiment
from metrics import summarize
from store import save_run, list_runs, aggregate

app = Flask(__name__)


def _has_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY")) or os.environ.get("GEMINI_MOCK") == "1"


@app.route("/")
def index():
    return render_template(
        "index.html",
        has_key=_has_key(),
        mock=os.environ.get("GEMINI_MOCK") == "1",
    )


@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(force=True, silent=True) or {}
    turns = int(data.get("turns", 10))
    message_chars = int(data.get("message_chars", 500))
    model = data.get("model", "gemini-2.0-flash")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key and os.environ.get("GEMINI_MOCK") != "1":
        return (
            jsonify({"error": "GEMINI_API_KEY not set. Set it, or run with GEMINI_MOCK=1."}),
            400,
        )

    turns = max(1, min(turns, 100))
    experiment = run_experiment(turns, message_chars, model, api_key)
    summary = summarize(experiment)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    path = save_run(timestamp, experiment, summary)

    return jsonify({"timestamp": timestamp, "saved_to": path,
                    "params": experiment["params"], "summary": summary})


@app.route("/history")
def history():
    return jsonify({"runs": list_runs(), "aggregate": aggregate()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
