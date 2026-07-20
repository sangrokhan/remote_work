"""The web face of the lab: start a run, watch it, take the numbers away.

Everything here is a thin skin over `core.runner`. The rules that make a run mean
something -- fresh connection per arm, prep outside the totals, a `both` pass that
would double-bill a stateful arm -- live there, not in a route handler, because a rule
enforced only by the UI is a rule the CLI does not have.

Two things this layer owns and the runner does not:

  Telling the operator what is about to be billed. A run reaches two paid APIs. The
  preflight (`GET /api/config`) reports what is ready, what capture would do, and what
  the current selection would warn about, before anything goes out.

  Not lying about mock data. A mock run is stored in its own bucket, listed in its own
  list, and labelled everywhere it appears. The last iteration of this lab accumulated
  122 synthetic runs indistinguishable from live ones; that is the failure this layer
  is built to make impossible.
"""

from __future__ import annotations

import json
import os
import queue
import io
import json
import threading
import zipfile
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template, request, send_file

from core import capture as pcap
from core import config, export, metrics, runner, scenario, store
from providers import base

app = Flask(__name__, template_folder="../templates", static_folder="../static")


def mock_mode(pairs=None) -> bool:
    """Whether this run contains synthetic calls.

    Any of them is enough. A run where Gemini is mocked and OpenAI is live is not a
    live run with a caveat -- it is a file whose Gemini numbers were never measured,
    and it belongs in the mock bucket where nothing will chart it against a real one.
    """
    names = base.names() if pairs is None else {p for p, _ in pairs}
    return any(config.is_mock(name) for name in names)


def _providers_view() -> list[dict]:
    """What each provider can do and whether it is able to do it right now."""
    out = []
    for name in base.names():
        mod = base.get(name)
        ok, reason = mod.ready()
        out.append({
            "name": name,
            "ready": ok,
            "reason": reason,
            "model": mod.DEFAULT_MODEL,
            "arms": list(mod.ARMS),
            "headline_arms": list(mod.HEADLINE_ARMS),
        })
    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/config")
def api_config():
    """The preflight. Everything the operator needs to decide whether to spend money.

    Not named `config`: that is the module that decides what counts as mock mode, and a
    route function of the same name shadows it -- turning every mock check in this file
    into an AttributeError on a Flask view.
    """
    cap_ok, cap_reason = pcap.available()
    fixture = scenario.load()
    return jsonify({
        "mock": mock_mode(),
        "providers": _providers_view(),
        "measures": list(runner.MEASURES),
        "capture": {"available": cap_ok, "reason": cap_reason,
                    "dir": str(pcap.pcap_dir())},
        "fixtures": scenario.names(),
        "fixture": {"name": fixture["name"], "description": fixture["description"],
                    "turns": len(fixture["steps"])},
        "retention_keep": store.retention_keep(),
    })


def _selection(payload: dict) -> tuple[dict, dict]:
    """Turn a request body into (providers, options), refusing anything the runner
    would have to guess at."""
    providers = payload.get("providers") or None
    if providers is not None and not isinstance(providers, dict):
        raise ValueError("providers must be a map of provider -> [arm, ...]")

    opts = {
        "measure": payload.get("measure") or "bytes",
        "models": payload.get("models") or {},
        "want_capture": bool(payload.get("capture")),
        "pause_seconds": float(payload.get("pause_seconds") or 0),
        "fixture": payload.get("fixture") or scenario.DEFAULT,
        "turns": payload.get("turns"),
        # Absent means "whatever TRAFFIC_CACHE_BUST says", which is not the same as
        # False -- an older UI that sends no such field must not silently turn the
        # arms' cache isolation off.
        "cache_bust": (None if payload.get("cache_bust") is None
                       else bool(payload["cache_bust"])),
        # Off unless asked for: it deliberately makes every number worse.
        "prefix_drift": bool(payload.get("prefix_drift")),
    }
    if opts["measure"] not in runner.MEASURES:
        raise ValueError(f"measure must be one of {', '.join(runner.MEASURES)}")
    return providers, opts


@app.post("/api/preflight")
def preflight():
    """What this exact selection would cost the operator in warnings, before it runs."""
    try:
        providers, opts = _selection(request.get_json(silent=True) or {})
        pairs = runner.plan(providers)
        fixture = scenario.load(opts["fixture"], opts["turns"])
    except (ValueError, KeyError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    passes = 2 if opts["measure"] == "both" else 1
    turns = len(fixture["steps"])
    # Count the billable calls per provider, not per run: with one provider mocked and
    # the other live, a single run-wide flag either hides a real bill or invents one.
    # Prep is not counted -- a cache build is real money, but how many calls it takes is
    # the arm's business and it is reported afterwards, not guessed at here.
    billable = sum(turns * passes for p, _ in pairs if not config.is_mock(p))
    return jsonify({
        "ok": True,
        "pairs": [f"{p}:{a}" for p, a in pairs],
        "turns": turns,
        "billable_calls": billable,
        "mock": mock_mode(pairs),
        "warnings": runner.warnings_for(pairs, opts["measure"]),
    })


def _execute(providers: dict | None, opts: dict, on_progress=None) -> dict:
    fixture = scenario.load(opts["fixture"], opts["turns"])
    timestamp = datetime.now(timezone.utc).isoformat()

    run = runner.run(
        providers,
        system=fixture["system"],
        steps=fixture["steps"],
        measure=opts["measure"],
        models=opts["models"],
        want_capture=opts["want_capture"],
        pause_seconds=opts["pause_seconds"],
        timestamp=timestamp,
        cache_bust=opts["cache_bust"],
        prefix_drift=opts["prefix_drift"],
        on_progress=on_progress,
    )
    run["timestamp"] = timestamp
    run["mock"] = mock_mode(runner.plan(providers))
    run["params"]["fixture"] = fixture["name"]
    # Summarize before saving: the summary is what the history list renders, and
    # recomputing it per page view means an old run's numbers change when the metrics
    # code changes -- which is how a chart quietly disagrees with the CSV beside it.
    run["summary"] = metrics.summarize(run)
    saved = store.save_run(run)
    run["exec_id"] = saved.get("exec_id")
    return run


@app.post("/api/run")
def run_blocking():
    """Run and return the whole document. Fine for a short run; use /api/run/stream for
    anything an operator would sit and watch."""
    try:
        providers, opts = _selection(request.get_json(silent=True) or {})
    except (ValueError, KeyError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    try:
        return jsonify({"ok": True, "run": _execute(providers, opts)})
    except (ValueError, KeyError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/run/stream")
def run_stream():
    """Same run, as server-sent events.

    A ten-turn comparison across six arms takes minutes, and a UI with no progress is a
    UI the operator reloads mid-run -- which abandons the request but not the calls,
    and bills for a run nobody will ever see. So each call announces itself before it
    goes out, and the run document arrives as the final event.
    """
    try:
        providers, opts = _selection(request.get_json(silent=True) or {})
        runner.plan(providers)          # fail on a bad selection before the stream opens
    except (ValueError, KeyError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    events: queue.Queue = queue.Queue()

    def work():
        try:
            run = _execute(providers, opts, on_progress=events.put)
            events.put({"event": "done", "run": run})
        except Exception as exc:      # the client must be told, not left hanging
            events.put({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is None:
                return
            yield f"data: {json.dumps(item)}\n\n"

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.get("/api/runs")
def runs():
    return jsonify(store.list_runs())


@app.get("/api/runs/<exec_id>")
def one_run(exec_id: str):
    doc = store.get_run(exec_id)
    if doc is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return jsonify({"ok": True, "run": doc})


@app.delete("/api/runs/<exec_id>")
def drop_run(exec_id: str):
    result = store.delete_run(exec_id)
    return jsonify(result), (200 if result.get("ok") else 404)


def _csv(exec_id: str, kind: str):
    doc = store.get_run(exec_id)
    if doc is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    body = (export.records_csv if kind == "records" else export.summary_csv)(doc)
    # A mock run's CSV says so in its filename. A number lifted out of a spreadsheet
    # has no other way of remembering it was never measured.
    tag = "mock_" if doc.get("mock") else ""
    return Response(body, mimetype="text/csv", headers={
        "Content-Disposition":
            f'attachment; filename="{tag}{kind}_{exec_id}.csv"'})


@app.get("/api/runs/<exec_id>/records.csv")
def records_csv(exec_id: str):
    return _csv(exec_id, "records")


@app.get("/api/runs/<exec_id>/summary.csv")
def summary_csv(exec_id: str):
    return _csv(exec_id, "summary")


@app.get("/api/pcaps/<name>")
def pcap_file(name: str):
    path = pcap.safe_pcap_path(name)
    if path is None:
        return jsonify({"ok": False, "error": "not_found"}), 404
    return send_file(path, as_attachment=True)


def _pcap_names(doc: dict):
    """Every pcap filename the run recorded, across arms and kinds.

    `pcaps` is {"provider:arm": {"bytes": {...}, "latency": {...}}}; only entries that
    actually captured something carry a `file`. A pcap that failed or got no packets has
    no file to bundle and is skipped -- the run document still explains its absence.
    """
    for by_kind in (doc.get("pcaps") or {}).values():
        for result in (by_kind or {}).values():
            name = (result or {}).get("file")
            if name:
                yield name


@app.get("/api/runs/<exec_id>/bundle.zip")
def bundle(exec_id: str):
    """Everything a run produced, in one download: both CSVs, the run document, and every
    pcap. Assembled here rather than in the browser because the server already holds all
    of it and the pcaps never leave the box until asked for."""
    doc = store.get_run(exec_id)
    if doc is None:
        return jsonify({"ok": False, "error": "not_found"}), 404

    tag = "mock_" if doc.get("mock") else ""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{tag}records_{exec_id}.csv", export.records_csv(doc))
        z.writestr(f"{tag}summary_{exec_id}.csv", export.summary_csv(doc))
        z.writestr(f"{tag}run_{exec_id}.json", json.dumps(doc, indent=2))
        for name in _pcap_names(doc):
            # safe_pcap_path validates the name and confirms the file is inside the pcap
            # directory: a run document is trusted, but the filename still goes through
            # the same gate a download URL does, so a doctored name cannot read elsewhere.
            path = pcap.safe_pcap_path(name)
            if path is not None:
                z.write(path, arcname=f"pcaps/{name}")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"{tag}run_{exec_id}.zip")


def main() -> None:
    app.run(host=os.environ.get("TRAFFIC_HOST", "127.0.0.1"),
            port=int(os.environ.get("TRAFFIC_PORT", "8080")),
            debug=False, threaded=True)


if __name__ == "__main__":
    main()
