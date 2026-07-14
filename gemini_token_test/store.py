"""Persist executions as local JSON files under data/runs/<exec_id>.json.

One file per execution, named by exec_id, holding the params, the summary and
whatever per-arm payload the experiment produced. Reads come from the same
directory. If no execution has been stored yet, a clearly-marked DUMMY dataset is
returned so the UI/graph still has something to show.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from metrics import _cumulative

DATA_DIR = Path(os.environ.get("GEMINI_DATA_DIR", "data/runs"))
# exec_id format: exec_<ts>_<8hex>  or  dummy_<word>. Guards path traversal.
_SAFE_EXEC = re.compile(r"^(exec_[0-9T\-]+_[0-9a-f]{8}|dummy_[a-z]+)$")


def _ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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
    """Write one execution (file name = exec_id) as JSON."""
    doc = {
        "exec_id": exec_id,
        "timestamp": timestamp,
        "mode": experiment["params"].get("mode"),
        "mock": experiment["params"].get("mock", False),
        "dummy": False,
        "params": experiment["params"],
        "summary": summary,
    }
    # Carry any extra payload (records, scenario, cache_set, *_records) verbatim.
    for k, v in experiment.items():
        if k != "params":
            doc[k] = v

    _ensure_dir()
    path = DATA_DIR / f"{exec_id}.json"
    path.write_text(json.dumps(doc, indent=2))
    return {"json": str(path), "exec_id": exec_id}


def delete_run(exec_id: str) -> dict:
    """Delete one stored execution. Rejects bad ids and the synthetic dummy rows."""
    if not _SAFE_EXEC.match(exec_id or "") or exec_id.startswith("dummy_"):
        return {"ok": False, "exec_id": exec_id, "error": "invalid or non-deletable exec_id"}
    p = DATA_DIR / f"{exec_id}.json"
    try:
        if not p.exists():
            return {"ok": False, "exec_id": exec_id, "json_deleted": False, "error": "not_found"}
        p.unlink()
    except Exception as exc:
        return {"ok": False, "exec_id": exec_id, "error": f"json_delete_failed: {exc}"}
    return {"ok": True, "exec_id": exec_id, "json_deleted": True, "error": ""}


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
    return _doc_from_json(exec_id)


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
    runs = _runs_from_json()
    if not runs:
        return {"source": "local_json (empty → dummy)", "dummy": True,
                "runs": [_list_item(d) for d in DUMMY_RUNS()]}
    return {"source": "local_json", "dummy": False, "runs": runs}


# --- Dummy backup dataset ----------------------------------------------------
def _dummy_series(growth):
    turns = list(range(1, 9))
    per = [growth(k) for k in turns]
    wire = [v * 6 for v in per]
    cum, cum_wire = _cumulative(per), _cumulative(wire)
    return {"turns": turns, "per_turn_tokens": per, "per_turn_prompt_tokens": per,
            "per_turn_wire_bytes": wire, "cum_tokens": cum, "cum_prompt_tokens": cum,
            "cum_wire_bytes": cum_wire, "cum_payload_bytes": cum_wire, "errors": []}


def _dummy_doc(exec_id, mode, growth):
    series = _dummy_series(growth)
    return {
        "exec_id": exec_id, "timestamp": "2000-01-01T00:00:00", "mode": mode,
        "mock": False, "dummy": True,
        "params": {"mode": mode, "turns": 8, "model": "dummy",
                   "endpoint": "dummy", "request_source": "dummy"},
        "summary": {"mode": mode, "series": series,
                    "totals": {"mode": mode, "tokens": series["cum_tokens"][-1],
                               "wire_bytes": series["cum_wire_bytes"][-1]}},
    }


def DUMMY_RUNS() -> list[dict]:
    # stateless grows ~quadratically, stateful ~flat.
    return [
        _dummy_doc("dummy_stateless", "stateless", lambda k: 100 * k),
        _dummy_doc("dummy_stateful", "stateful", lambda k: 100),
    ]
