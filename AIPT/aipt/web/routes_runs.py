"""GET/DELETE /api/runs* -- inspect a stored run, and pull its 3-layer CSV
export set (DESIGN.md 4.6) or a bundle.zip, ported from token_traffic's
``core/app.py`` download routes plus tcp_congestion's ``/api/download/*``,
onto the in-memory store in ``aipt/web/store.py`` (Phase 1 -- see that
module's docstring on persistence).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from aipt.core import capture as capture_mod
from aipt.export import bundle as bundle_mod
from aipt.export import connection as connection_mod
from aipt.export import packets as packets_mod
from aipt.export import turns as turns_mod
from aipt.web import store as run_store

router = APIRouter()


def _not_found() -> JSONResponse:
    return JSONResponse({"ok": False, "error": "not_found"}, status_code=404)


@router.get("/api/runs")
def list_runs():
    return JSONResponse(run_store.list_runs())


@router.get("/api/runs/{exec_id}")
def get_run(exec_id: str):
    doc = run_store.get_run(exec_id)
    if doc is None:
        return _not_found()
    return JSONResponse({"ok": True, "run": doc})


@router.delete("/api/runs/{exec_id}")
def delete_run(exec_id: str):
    ok = run_store.delete_run(exec_id)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


def _csv_response(body: str, filename: str) -> Response:
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _tag(doc: dict) -> str:
    # A mock run's CSV says so in its filename -- token_traffic/core/app.py's
    # rule (see that module's docstring): a number lifted out of a
    # spreadsheet has no other way of remembering it was never measured.
    return "mock_" if doc.get("mock") else ""


@router.get("/api/runs/{exec_id}/turns.csv")
def turns_csv(exec_id: str):
    doc = run_store.get_run(exec_id)
    if doc is None:
        return _not_found()
    body = turns_mod.turns_csv(doc.get("turns") or [])
    return _csv_response(body, f"{_tag(doc)}turns_{exec_id}.csv")


@router.get("/api/runs/{exec_id}/summary.csv")
def summary_csv(exec_id: str):
    """One row per run: turn count, error state, elapsed time.

    token_traffic's ``summary.csv`` summarized per-(provider,arm); this
    backend-selection UI runs one backend/arm per request, so the summary
    collapses to one row. Kept as its own endpoint (rather than folded into
    turns.csv) for URL/route parity with the original app.
    """
    doc = run_store.get_run(exec_id)
    if doc is None:
        return _not_found()
    import csv
    import io

    buf = io.StringIO()
    w = csv.DictWriter(
        buf,
        fieldnames=["exec_id", "backend", "arm", "label", "mock", "turn_count",
                    "elapsed_s", "error", "timestamp"],
    )
    w.writeheader()
    w.writerow({
        "exec_id": exec_id,
        "backend": doc.get("backend", ""),
        "arm": doc.get("arm", ""),
        "label": doc.get("label", ""),
        "mock": doc.get("mock", False),
        "turn_count": len(doc.get("turns") or []),
        "elapsed_s": doc.get("elapsed_s", ""),
        "error": doc.get("error", ""),
        "timestamp": doc.get("timestamp", ""),
    })
    return _csv_response(buf.getvalue(), f"{_tag(doc)}summary_{exec_id}.csv")


@router.get("/api/runs/{exec_id}/cwnd.csv")
def cwnd_csv(exec_id: str):
    doc = run_store.get_run(exec_id)
    if doc is None:
        return _not_found()
    body = connection_mod.connection_csv(doc.get("monitors") or [])
    return _csv_response(body, f"{_tag(doc)}cwnd_{exec_id}.csv")


@router.get("/api/runs/{exec_id}/cwnd_summary.csv")
def cwnd_summary_csv(exec_id: str):
    doc = run_store.get_run(exec_id)
    if doc is None:
        return _not_found()
    body = connection_mod.connection_summary_csv(doc.get("monitors") or [])
    return _csv_response(body, f"{_tag(doc)}cwnd_summary_{exec_id}.csv")


@router.get("/api/runs/{exec_id}/packets.csv")
def packets_csv(exec_id: str):
    doc = run_store.get_run(exec_id)
    if doc is None:
        return _not_found()
    pcap_info = doc.get("pcap") or {}
    name = pcap_info.get("file")
    if not name:
        # No capture on this run: header-only CSV, same "monitored nothing"
        # convention aipt.export.packets.packets_csv uses for a missing pcap.
        body = packets_mod.packets_csv("__no_such_file__.pcap")
    else:
        path = capture_mod.safe_pcap_path(name)
        body = packets_mod.packets_csv(path) if path is not None else \
            packets_mod.packets_csv("__no_such_file__.pcap")
    return _csv_response(body, f"{_tag(doc)}packets_{exec_id}.csv")


@router.get("/api/runs/{exec_id}/bundle.zip")
def bundle_zip(exec_id: str):
    doc = run_store.get_run(exec_id)
    if doc is None:
        return _not_found()

    pcap_info = doc.get("pcap") or {}
    pcap_name = pcap_info.get("file")
    pcap_paths = []
    if pcap_name:
        path = capture_mod.safe_pcap_path(pcap_name)
        if path is not None:
            pcap_paths.append(path)

    zip_bytes = bundle_mod.build_bundle_zip(
        label=doc.get("label") or exec_id,
        connection_csv=connection_mod.connection_csv(doc.get("monitors") or []),
        turns_csv=turns_mod.turns_csv(doc.get("turns") or []),
        packets_csv=packets_mod.packets_csv(pcap_paths[0]) if pcap_paths else None,
        pcap_paths=pcap_paths,
    )
    filename = f"{_tag(doc)}{bundle_mod.bundle_zip_name(doc.get('label') or exec_id)}"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/pcaps/{name}")
def pcap_file(name: str):
    from fastapi.responses import FileResponse

    path = capture_mod.safe_pcap_path(name)
    if path is None:
        return _not_found()
    return FileResponse(str(path), media_type="application/vnd.tcpdump.pcap",
                        filename=name)
