"""Persist runs to BOTH local JSON and Firestore (dual-write).

- Always writes a local JSON file under data/runs/.
- Also writes a Firestore document when Firestore is available (lib installed +
  ADC creds + project). On Cloud Run this uses the service account automatically.
- Reads/aggregate prefer Firestore when available (cluster-wide, survives instance
  recycle); otherwise fall back to the local JSON files.

Firestore uses the same ADC auth as Vertex — no extra key needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("GEMINI_DATA_DIR", "data/runs"))
COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "gemini_runs")
DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")

_fs_client = None
_fs_checked = False


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _firestore():
    """Return a Firestore client, or None if unavailable. Cached."""
    global _fs_client, _fs_checked
    if _fs_checked:
        return _fs_client
    _fs_checked = True
    if os.environ.get("FIRESTORE_DISABLE") == "1":
        return None
    try:
        from google.cloud import firestore
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or None
        kwargs = {"database": DATABASE} if DATABASE and DATABASE != "(default)" else {}
        _fs_client = firestore.Client(project=project, **kwargs)
    except Exception:
        _fs_client = None
    return _fs_client


def firestore_active() -> bool:
    return _firestore() is not None


def _doc_id(timestamp: str) -> str:
    return f"run_{timestamp.replace(':', '-')}"


def save_run(timestamp: str, experiment: dict, summary: dict) -> dict:
    """Write run to local JSON and (best-effort) Firestore. Returns where it went."""
    run = {"timestamp": timestamp, "params": experiment["params"], "summary": summary}
    result = {"json": None, "firestore": None}

    # Local JSON (always).
    _ensure_dir()
    path = DATA_DIR / f"{_doc_id(timestamp)}.json"
    path.write_text(json.dumps(run, indent=2))
    result["json"] = str(path)

    # Firestore (best-effort).
    fs = _firestore()
    if fs is not None:
        try:
            doc = {
                "timestamp": timestamp,
                "params": run["params"],
                "totals": summary["totals"],
                "summary": summary,
            }
            fs.collection(COLLECTION).document(_doc_id(timestamp)).set(doc)
            result["firestore"] = f"{COLLECTION}/{_doc_id(timestamp)}"
        except Exception as exc:
            result["firestore"] = f"error: {exc}"

    return result


def _runs_from_firestore() -> list[dict] | None:
    fs = _firestore()
    if fs is None:
        return None
    try:
        runs = []
        for snap in fs.collection(COLLECTION).stream():
            d = snap.to_dict() or {}
            runs.append(
                {
                    "timestamp": d.get("timestamp", snap.id),
                    "params": d.get("params", {}),
                    "totals": d.get("totals", {}),
                }
            )
        runs.sort(key=lambda r: r["timestamp"])
        return runs
    except Exception:
        return None


def _runs_from_json() -> list[dict]:
    _ensure_dir()
    runs = []
    for p in sorted(DATA_DIR.glob("run_*.json")):
        try:
            data = json.loads(p.read_text())
            runs.append(
                {
                    "timestamp": data["timestamp"],
                    "params": data["params"],
                    "totals": data["summary"]["totals"],
                }
            )
        except Exception:
            continue
    return runs


def list_runs() -> list[dict]:
    """Prefer Firestore (cluster-wide); fall back to local JSON."""
    fs_runs = _runs_from_firestore()
    return fs_runs if fs_runs is not None else _runs_from_json()


def aggregate() -> dict:
    """Total tokens + traffic across all runs, grouped by endpoint."""
    runs = list_runs()
    by_endpoint: dict[str, dict] = {}
    for r in runs:
        ep = r["params"].get("endpoint", "unknown")
        agg = by_endpoint.setdefault(
            ep,
            {
                "runs": 0,
                "stateless_tokens": 0,
                "delta_tokens": 0,
                "stateless_wire_bytes": 0,
                "delta_wire_bytes": 0,
            },
        )
        t = r["totals"]
        agg["runs"] += 1
        agg["stateless_tokens"] += t.get("stateless_tokens", 0)
        agg["delta_tokens"] += t.get("delta_tokens", 0)
        agg["stateless_wire_bytes"] += t.get("stateless_wire_bytes", 0)
        agg["delta_wire_bytes"] += t.get("delta_wire_bytes", 0)
    return {
        "total_runs": len(runs),
        "source": "firestore" if firestore_active() else "local_json",
        "by_endpoint": by_endpoint,
    }
