"""One run, one JSON file, with a retention policy and a wall between real and mock.

The previous layout had neither. It accumulated 122 files and 17 MB, most of them
synthetic runs that sat in the same directory as the live ones, under the same
naming scheme, and charted identically -- so "what did the cached arm cost" could
be answered from a run that never touched the network. Two rules follow:

* A mock run lives in its own bucket. It is never listed with the live runs, never
  charted alongside one, and never counted against their retention -- a week of
  offline development cannot evict the one live run somebody paid for.
* Every saved run carries `schema_version`. A run written by an older layout must
  be identifiable as such, not silently charted next to a current one whose columns
  mean something else.

Storage is a directory of JSON files. There is no second datastore: a run is a few
hundred kilobytes and this experiment runs on one machine.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from core.record import SCHEMA_VERSION

# Mock runs sit in a subdirectory rather than behind a flag in the filename,
# because a flag in the filename is a rule a future reader has to remember and a
# directory is one the filesystem enforces.
_MOCK_BUCKET = "mock"

# Anything that reaches the filesystem as a name gets checked first: an exec_id
# arrives from an HTTP route, and "../../etc/passwd" is a perfectly good string.
_SAFE_EXEC = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def data_dir() -> Path:
    """Read the env on every call, not at import: the tests point this at a tmpdir,
    and a module-level constant would have frozen the real one into them."""
    return Path(os.environ.get("TRAFFIC_DATA_DIR", "data/runs"))


def retention_keep() -> int:
    return int(os.environ.get("TRAFFIC_RETENTION_KEEP", "20"))


def new_exec_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"exec_{ts}_{secrets.token_hex(4)}"


def _bucket(mock: bool) -> Path:
    d = data_dir()
    return d / _MOCK_BUCKET if mock else d


def _path(exec_id: str, mock: bool) -> Path | None:
    if not _SAFE_EXEC.match(exec_id or ""):
        return None
    b = _bucket(mock)
    p = (b / f"{exec_id}.json").resolve()
    if p.parent != b.resolve():
        return None
    return p


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        # A half-written or hand-edited file is not a run. Skipping it keeps one bad
        # file from taking the whole history page down.
        return None


def _files(mock: bool) -> list[Path]:
    b = _bucket(mock)
    if not b.is_dir():
        return []
    return [p for p in b.glob("*.json") if p.is_file()]


def _sort_key(path: Path) -> tuple:
    """Newest first, by the run's own timestamp with mtime as the tiebreak.

    mtime alone would misorder a run copied in from elsewhere; the timestamp alone
    ties whenever two runs land in the same second, which the tests do routinely.
    """
    doc = _load(path) or {}
    return (str(doc.get("timestamp") or ""), path.stat().st_mtime, path.name)


def _list_item(doc: dict) -> dict:
    return {
        "exec_id": doc.get("exec_id"),
        "timestamp": doc.get("timestamp"),
        "schema_version": doc.get("schema_version"),
        "mock": doc.get("mock", False),
        "measure": (doc.get("params") or {}).get("measure", ""),
        "providers": (doc.get("params") or {}).get("providers", []),
        "totals": (doc.get("summary") or {}).get("totals", {}),
        "failures": len((doc.get("summary") or {}).get("failures", [])),
    }


def save_run(run: dict) -> dict:
    """Write one run as JSON, then prune its own bucket back to the keep limit.

    Pruning here rather than in a cron job or a cleanup route means retention holds
    even if nobody ever remembers it exists -- which is exactly what happened last
    time.
    """
    exec_id = run.get("exec_id") or new_exec_id()
    mock = bool(run.get("mock", (run.get("params") or {}).get("mock", False)))

    doc = dict(run)
    doc["exec_id"] = exec_id
    doc["schema_version"] = SCHEMA_VERSION
    doc["mock"] = mock
    doc.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    path = _path(exec_id, mock)
    if path is None:
        return {"ok": False, "exec_id": exec_id, "error": "invalid exec_id"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))

    pruned = prune()
    return {"ok": True, "exec_id": exec_id, "path": str(path), "mock": mock,
            "schema_version": SCHEMA_VERSION,
            "pruned": pruned["deleted_live"] + pruned["deleted_mock"]}


def get_run(exec_id: str) -> dict | None:
    """The whole run document, live bucket first, then mock."""
    for mock in (False, True):
        p = _path(exec_id, mock)
        if p is not None and p.is_file():
            return _load(p)
    return None


def list_runs() -> dict:
    """Runs for the history view: live and mock kept apart, newest first.

    They are returned as two lists rather than one flagged list, because a caller
    that has to remember to filter is a caller that will forget to filter, and the
    forgetting is what put synthetic numbers on a chart last time.
    """
    live, mock = [], []
    for bucket, out in ((False, live), (True, mock)):
        for p in sorted(_files(bucket), key=_sort_key, reverse=True):
            doc = _load(p)
            if doc is not None:
                out.append(_list_item(doc))
    return {"runs": live, "mock_runs": mock,
            "keep": retention_keep(), "dir": str(data_dir())}


def delete_run(exec_id: str) -> dict:
    for mock in (False, True):
        p = _path(exec_id, mock)
        if p is not None and p.is_file():
            try:
                p.unlink()
            except Exception as exc:
                return {"ok": False, "exec_id": exec_id, "error": f"delete_failed: {exc}"}
            return {"ok": True, "exec_id": exec_id, "mock": mock}
    return {"ok": False, "exec_id": exec_id, "error": "not_found"}


def prune(keep: int | None = None) -> dict:
    """Keep the newest `keep` runs in each bucket; delete the rest.

    Each bucket gets its own budget. Mock runs cannot evict live ones -- that is the
    whole point of the split -- and live runs do not evict mock ones either, so a
    fixture run stays reproducible while real runs come and go.
    """
    n = retention_keep() if keep is None else keep
    out = {"keep": n, "deleted_live": 0, "deleted_mock": 0}
    for mock, field in ((False, "deleted_live"), (True, "deleted_mock")):
        paths = sorted(_files(mock), key=_sort_key, reverse=True)
        for p in paths[max(n, 0):]:
            try:
                p.unlink()
                out[field] += 1
            except Exception:
                continue
    return out
