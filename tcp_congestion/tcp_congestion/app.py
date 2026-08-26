"""app: web frontend for launching a conversation run and viewing its result.

Endpoints:
  GET  /                          the form page
  POST /api/run                   run a conversation synchronously, return JSON
  GET  /api/config                capability info (cwnd monitor / capture available?)
  GET  /api/download/cwnd.csv     the last run's congestion-window series as CSV
  GET  /api/download/turns.csv    the last run's per-turn summary as CSV
  GET  /api/download/pcap         the last run's packet capture, if it ran with capture=True

Kept intentionally simple: one blocking run per request, since the front-end
button explicitly waits for the result chart. A run of a handful of turns
with a few-second idle each finishes in well under the default HTTP timeout.

The last run's result is kept in memory (module-level, one slot) so the
download endpoints have something to serve without re-running anything --
this is a single-operator lab tool, not a multi-tenant service.
"""

from __future__ import annotations

import io
import os
import re
import zipfile

from tcp_congestion import capture as capture_mod
from tcp_congestion import congestion as congestion_mod
from tcp_congestion import conversation, cwnd, export

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover - exercised only when fastapi missing
    FastAPI = None  # type: ignore

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATE_PATH = os.path.join(_HERE, "..", "templates", "index.html")

# One slot: the most recent run's result, for the download endpoints.
_LAST_RESULT: dict | None = None

# Safe-for-filename slug of the last run's algorithm, so the zip name can
# carry it (e.g. "tcp_congestion_bbr_20260825-101500.zip") without ever
# emitting a path-unsafe or empty component.
_SAFE_SLUG = re.compile(r"[^a-z0-9_-]+")


def _algorithm_slug(result: dict | None) -> str:
    algo = (result or {}).get("algorithm") or (result or {}).get(
        "algorithm_requested") or "default"
    slug = _SAFE_SLUG.sub("-", algo.lower()).strip("-")
    return slug or "default"


def _read_template() -> str:
    with open(_TEMPLATE_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def build_run_params(payload: dict) -> dict:
    """Validate/coerce the JSON body from the form into conversation.run() kwargs.

    Raises ValueError on a bad value, with a message the form can show.
    """
    def _int(name: str, default: int, minimum: int = 1) -> int:
        raw = payload.get(name, default)
        try:
            n = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer")
        if n < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
        return n

    enable_ping_probes = bool(payload.get("enable_ping_probes", True))
    # ping_interval_ms only matters while probes are actually being sent;
    # when the "Send HTTP PING during idle gap" checkbox is off, 0 (or any
    # other placeholder value left in the form) must not block the run --
    # conversation.run() never reads the value in that case.
    if enable_ping_probes:
        ping_interval_ms = _int("ping_interval_ms", 1)
    else:
        ping_interval_ms = _int("ping_interval_ms", 1, minimum=0)

    return {
        "host": str(payload.get("host") or "server"),
        "port": _int("port", 8888),
        "num_turns": _int("num_turns", 20),
        "system_prompt_bytes": _int("system_prompt_bytes", 20000, minimum=0),
        "turn_user_msg_bytes": _int("turn_user_msg_bytes", 1000),
        "mock_response_bytes": _int("mock_response_bytes", 1000, minimum=0),
        "inference_delay_ms": _int("inference_delay_ms", 1000, minimum=0),
        "idle_duration_ms": _int("idle_duration_ms", 0, minimum=0),
        "ping_interval_ms": ping_interval_ms,
        "label": str(payload.get("label") or "conversation"),
        "capture": bool(payload.get("capture", False)),
        "algorithm": (str(payload["algorithm"]).strip().lower()
                      if payload.get("algorithm") else None),
        "enable_ping_probes": enable_ping_probes,
    }


def create_app():
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("fastapi is not installed")

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _read_template()

    @app.get("/api/config")
    def api_config():
        cwnd_ok, cwnd_reason = cwnd.available()
        cap_ok, cap_reason = capture_mod.available()
        cc_status = congestion_mod.status()
        return JSONResponse({
            "cwnd_available": cwnd_ok, "cwnd_reason": cwnd_reason,
            "capture_available": cap_ok, "capture_reason": cap_reason,
            "congestion": cc_status,
        })

    @app.post("/api/run")
    async def api_run(request: Request):
        global _LAST_RESULT
        payload = await request.json()
        try:
            params = build_run_params(payload)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        try:
            result = conversation.run(**params)
        except OSError as exc:
            return JSONResponse({"error": f"connection failed: {exc}"},
                                status_code=502)
        _LAST_RESULT = result
        return JSONResponse(result)

    @app.get("/api/download/cwnd.csv")
    def download_cwnd_csv():
        if _LAST_RESULT is None:
            return JSONResponse({"error": "no run yet"}, status_code=404)
        text = export.cwnd_csv(_LAST_RESULT)
        return Response(content=text, media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="cwnd.csv"'})

    @app.get("/api/download/turns.csv")
    def download_turns_csv():
        if _LAST_RESULT is None:
            return JSONResponse({"error": "no run yet"}, status_code=404)
        text = export.turns_csv(_LAST_RESULT)
        return Response(content=text, media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="turns.csv"'})

    @app.get("/api/download/pcap")
    def download_pcap():
        if _LAST_RESULT is None or not _LAST_RESULT.get("pcap"):
            return JSONResponse({"error": "no capture in the last run"},
                                status_code=404)
        pcap_info = _LAST_RESULT["pcap"]
        name = pcap_info.get("file")
        if not name:
            return JSONResponse({"error": pcap_info.get("error") or "capture failed"},
                                status_code=404)
        path = capture_mod.safe_pcap_path(name)
        if path is None:
            return JSONResponse({"error": "pcap file not found"}, status_code=404)
        return FileResponse(str(path), media_type="application/vnd.tcpdump.pcap",
                            filename=name)

    @app.get("/api/download/bundle.zip")
    def download_bundle_zip():
        """All of the last run's artifacts (cwnd.csv, turns.csv, and the
        pcap if captured) in one zip, named with the algorithm that was
        actually used so runs for different algorithms never overwrite or
        get confused with each other in a downloads folder."""
        if _LAST_RESULT is None:
            return JSONResponse({"error": "no run yet"}, status_code=404)

        slug = _algorithm_slug(_LAST_RESULT)
        stamp = str(_LAST_RESULT.get("label") or "conversation")
        stamp = _SAFE_SLUG.sub("-", stamp.lower()).strip("-") or "conversation"
        zip_name = f"tcp_congestion_{slug}_{stamp}.zip"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{slug}_cwnd.csv", export.cwnd_csv(_LAST_RESULT))
            zf.writestr(f"{slug}_turns.csv", export.turns_csv(_LAST_RESULT))

            pcap_info = _LAST_RESULT.get("pcap") or {}
            pcap_name = pcap_info.get("file")
            if pcap_name:
                path = capture_mod.safe_pcap_path(pcap_name)
                if path is not None:
                    zf.write(str(path), arcname=f"{slug}_capture.pcap")

        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition":
                                 f'attachment; filename="{zip_name}"'})

    static_dir = os.path.join(_HERE, "..", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app
