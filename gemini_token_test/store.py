"""Persist executions to BOTH local JSON and Firestore, keyed by exec_id.

- Always writes a local JSON file under data/runs/<exec_id>.json.
- Also writes a Firestore document (doc id = exec_id) when Firestore is available.
- Reads prefer Firestore (cluster-wide, survives Cloud Run recycles), else local
  JSON. If neither has any execution, a clearly-marked DUMMY dataset is returned so
  the UI/graph still has something to show.

Firestore uses the same ADC auth as Vertex — no extra key needed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATA_DIR = Path(os.environ.get("GEMINI_DATA_DIR", "data/runs"))
# exec_id format: exec_<ts>_<8hex>  or  dummy_<word>. Guards path traversal.
_SAFE_EXEC = re.compile(r"^(exec_[0-9T\-]+_[0-9a-f]{8}|dummy_[a-z]+)$")
COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "gemini_runs")
DATABASE = os.environ.get("FIRESTORE_DATABASE", "(default)")

_fs_client = None
_fs_checked = False


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _firestore():
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


def _list_item(doc: dict) -> dict:
    return {
        "exec_id": doc.get("exec_id"),
        "timestamp": doc.get("timestamp"),
        "mode": doc.get("mode"),
        "mock": doc.get("mock", False),
        "dummy": doc.get("dummy", False),
        "totals": doc.get("summary", {}).get("totals", {}),
    }


def save_run(exec_id: str, timestamp: str, experiment: dict, summary: dict) -> dict:
    """Write one execution (doc id = exec_id) to JSON and Firestore."""
    doc = {
        "exec_id": exec_id,
        "timestamp": timestamp,
        "mode": experiment["params"].get("mode"),
        "mock": experiment["params"].get("mock", False),
        "dummy": False,
        "params": experiment["params"],
        "summary": summary,
    }
    result = {"json": None, "firestore": None, "exec_id": exec_id}

    _ensure_dir()
    path = DATA_DIR / f"{exec_id}.json"
    path.write_text(json.dumps(doc, indent=2))
    result["json"] = str(path)

    fs = _firestore()
    if fs is not None:
        try:
            fs.collection(COLLECTION).document(exec_id).set(doc)
            result["firestore"] = f"{COLLECTION}/{exec_id}"
        except Exception as exc:
            result["firestore"] = f"error: {exc}"
    return result


def _doc_from_firestore(exec_id: str) -> dict | None:
    fs = _firestore()
    if fs is None:
        return None
    try:
        snap = fs.collection(COLLECTION).document(exec_id).get()
        return snap.to_dict() if snap.exists else None
    except Exception:
        return None


def _doc_from_json(exec_id: str) -> dict | None:
    p = DATA_DIR / f"{exec_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def get_run(exec_id: str) -> dict | None:
    """Full execution document (incl. summary.series) for the viewer/graph."""
    if not _SAFE_EXEC.match(exec_id or ""):
        return None
    for d in DUMMY_RUNS():
        if d["exec_id"] == exec_id:
            return d
    return _doc_from_firestore(exec_id) or _doc_from_json(exec_id)


def _runs_from_firestore() -> list[dict] | None:
    fs = _firestore()
    if fs is None:
        return None
    try:
        runs = [_list_item(s.to_dict() or {}) for s in fs.collection(COLLECTION).stream()]
        runs.sort(key=lambda r: r.get("timestamp") or "")
        return runs
    except Exception:
        return None


def _runs_from_json() -> list[dict]:
    _ensure_dir()
    runs = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            runs.append(_list_item(json.loads(p.read_text())))
        except Exception:
            continue
    return runs


def list_runs() -> dict:
    """Executions for the history viewer. Falls back to DUMMY when none found."""
    fs_runs = _runs_from_firestore()
    if fs_runs is not None:
        source = "firestore"
        runs = fs_runs
    else:
        source = "local_json"
        runs = _runs_from_json()
    if not runs:
        return {"source": f"{source} (empty → dummy)", "dummy": True,
                "runs": [_list_item(d) for d in DUMMY_RUNS()]}
    return {"source": source, "dummy": False, "runs": runs}


# --- Dummy backup dataset ----------------------------------------------------
def _dummy_series(growth):
    turns = list(range(1, 9))
    per = [growth(k) for k in turns]
    wire = [v * 6 for v in per]
    cum = []
    acc = 0
    for v in per:
        acc += v
        cum.append(acc)
    cumw = []
    acc = 0
    for v in wire:
        acc += v
        cumw.append(acc)
    return {"turns": turns, "per_turn_tokens": per, "per_turn_prompt_tokens": per,
            "per_turn_wire_bytes": wire, "cum_tokens": cum, "cum_prompt_tokens": cum,
            "cum_wire_bytes": cumw, "cum_payload_bytes": cumw, "errors": []}


def _dummy_doc(exec_id, mode, growth):
    series = _dummy_series(growth)
    return {
        "exec_id": exec_id, "timestamp": "2000-01-01T00:00:00", "mode": mode,
        "mock": False, "dummy": True,
        "params": {"mode": mode, "turns": 8, "model": "dummy",
                   "endpoint": "dummy", "request_source": "dummy"},
        "summary": {"mode": mode, "series": series,
                    "totals": {"mode": mode, "tokens": series["cum_tokens"][-1],
                               "wire_bytes": series["cum_wire_bytes"][-1],
                               "cost_usd": 0.0, "price_per_token": 0.0}},
    }


def DUMMY_RUNS() -> list[dict]:
    # stateless grows ~quadratically, stateful ~flat.
    return [
        _dummy_doc("dummy_stateless", "stateless", lambda k: 100 * k),
        _dummy_doc("dummy_stateful", "stateful", lambda k: 100),
    ]
