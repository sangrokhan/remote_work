"""Web UI: start a run, watch it turn by turn, read the result, take the evidence.

Routes:
  GET  /                       the page
  GET  /status                 model, capture availability, fixture sizes
  POST /run/stream             run the experiment, streaming per-turn progress (SSE)
  GET  /history                past runs, newest first
  GET  /run/<exec_id>          one past run in full
  GET  /download/<exec_id>/<what>   run.json | summary.csv | charts.png | bodies.zip
  GET  /download/pcap/<name>   a capture from this or an earlier run

Single-threaded on purpose. The socket byte counter in wire.py is a process-global
tally read by difference, which is exact only while one request is in flight. Two
concurrent runs would silently attribute each other's bytes, so the app refuses a
second run while one is going.
"""

from __future__ import annotations

import io
import json
import queue
import threading
import zipfile
from pathlib import Path

from flask import (Flask, Response, jsonify, render_template, request,
                   send_file, send_from_directory)

import env  # noqa: F401  — .env before anything reads os.environ
import capture as cap_mod
import experiment
import fixture as fixture_mod
import metrics
import openai_client as oc
import store

app = Flask(__name__)
# Flask sorts JSON keys by default, which silently reordered the fixture list:
# "default" sorts before "perf", so the UI's first-and-selected option became the
# 9k-char fixture while the page claimed to run the 20k-char one. Key order here
# is meaningful — it is the order the user sees and picks from.
app.json.sort_keys = False

# One run at a time. See the module docstring: concurrent runs would corrupt the
# byte tally, and a corrupted byte tally is the one thing this tool cannot ship.
_run_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    cap_ok, cap_why = cap_mod.available()
    fixtures = {}
    for name in ("perf", "default"):
        try:
            fx = fixture_mod.load(name)
            fixtures[name] = {
                "turns": len(fx.steps),
                "system_chars": fx.system_chars,
                "description": fx.description,
            }
        except Exception:
            pass
    return jsonify({
        "model": oc.DEFAULT_MODEL,
        "host": oc.api_host(),
        "arms": list(oc.ARMS),
        "key_present": bool(oc.os.environ.get("OPENAI_API_KEY")),
        "reasoning_effort": oc.DEFAULT_REASONING_EFFORT or None,
        "max_output_tokens": oc.DEFAULT_MAX_OUTPUT_TOKENS,
        "capture": {"available": cap_ok, "reason": cap_why,
                    "dir": str(cap_mod.PCAP_DIR)},
        "fixtures": fixtures,
        "prices": {"input": metrics.PRICE_INPUT,
                   "cached_input": metrics.PRICE_CACHED_INPUT,
                   "output": metrics.PRICE_OUTPUT},
        "busy": _run_lock.locked(),
    })


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/run/stream", methods=["POST"])
def run_stream():
    """Run the experiment, emitting one SSE event per turn, then 'done'."""
    body = request.get_json(silent=True) or {}
    turns = max(1, min(int(body.get("turns", 10)), 50))
    repeats = max(1, min(int(body.get("repeats", 1)), 10))
    fixture_name = body.get("fixture", "perf")
    model = body.get("model") or oc.DEFAULT_MODEL
    arms = tuple(body.get("arms") or oc.ARMS)
    want_capture = bool(body.get("capture"))
    stream = bool(body.get("stream"))

    if not _run_lock.acquire(blocking=False):
        return Response(_sse("error", {"message": "a run is already in progress"}),
                        mimetype="text/event-stream")

    events: queue.Queue = queue.Queue()

    def on_turn(arm, k, n, res):
        events.put(("turn", {
            "arm": arm, "turn": k, "turns": n,
            "upload_bytes": res.req_payload_bytes,
            "wire_sent": res.wire_sent,
            "input_tokens": res.input_tokens,
            "cached_tokens": res.cached_tokens,
            "output_tokens": res.output_tokens,
            "latency_ms": res.latency_ms,
            "ttft_ms": res.ttft_ms,
            "ttlt_ms": res.ttlt_ms,
            "streamed": res.streamed,
        }))

    def work():
        try:
            capture = want_capture and cap_mod.available()[0]
            if want_capture and not capture:
                events.put(("note", {"message": "capture unavailable: "
                                                + cap_mod.available()[1]}))
            exp = experiment.run_experiment(
                fixture_name=fixture_name, model=model, turns=turns,
                repeats=repeats, arms=arms, capture=capture, stream=stream,
                on_turn=on_turn,
            )
            summary = metrics.summarize(exp)
            exec_id = store.new_exec_id()
            manifest = store.save_run(exec_id, exp, summary, exp.get("captures"))
            events.put(("done", {
                "exec_id": exec_id,
                "summary": summary,
                "captures": exp.get("captures", []),
                "manifest": manifest,
            }))
        except Exception as exc:
            # never leave a half-stream: say what broke, then end it
            events.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            events.put((None, None))
            _run_lock.release()

    threading.Thread(target=work, daemon=True).start()

    def gen():
        yield _sse("start", {"turns": turns, "repeats": repeats,
                             "arms": list(arms), "model": model,
                             "stream": stream})
        while True:
            kind, payload = events.get()
            if kind is None:
                break
            yield _sse(kind, payload)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/history")
def history():
    return jsonify({"runs": store.list_runs()})


@app.route("/run/<exec_id>")
def get_run(exec_id):
    doc = store.load_run(exec_id)
    if doc is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc)


_DOWNLOADS = {
    "run.json": ("run.json", "application/json"),
    "summary.csv": ("summary.csv", "text/csv"),
    "charts.png": ("charts.png", "image/png"),
}


@app.route("/download/<exec_id>/<what>")
def download(exec_id, what):
    d = store.run_dir(exec_id)
    if d is None or not d.is_dir():
        return jsonify({"error": "not found"}), 404

    if what == "bodies.zip":
        bodies = d / "bodies"
        if not bodies.is_dir():
            return jsonify({"error": "no bodies recorded"}), 404
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(bodies.iterdir()):
                z.write(f, arcname=f"{exec_id}/{f.name}")
        buf.seek(0)
        return send_file(buf, mimetype="application/zip", as_attachment=True,
                         download_name=f"{exec_id}_bodies.zip")

    if what not in _DOWNLOADS:
        return jsonify({"error": "unknown download"}), 404
    name, mime = _DOWNLOADS[what]
    if not (d / name).is_file():
        return jsonify({"error": "not found"}), 404
    return send_from_directory(d, name, mimetype=mime, as_attachment=True,
                               download_name=f"{exec_id}_{name}")


@app.route("/download/pcap/<name>")
def download_pcap(name):
    p = cap_mod.safe_pcap_path(name)
    if p is None:
        return jsonify({"error": "not found"}), 404
    return send_file(p, mimetype="application/vnd.tcpdump.pcap",
                     as_attachment=True, download_name=name)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
