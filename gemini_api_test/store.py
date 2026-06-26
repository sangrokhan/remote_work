"""Persist runs to data/runs/ and aggregate across all runs (background collection)."""

from __future__ import annotations

import json
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("GEMINI_DATA_DIR", "data/runs"))


def _ensure() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_run(timestamp: str, experiment: dict, summary: dict) -> str:
    """Write one run JSON. timestamp passed in (caller stamps time)."""
    _ensure()
    run = {"timestamp": timestamp, "params": experiment["params"], "summary": summary}
    path = DATA_DIR / f"run_{timestamp.replace(':', '-')}.json"
    path.write_text(json.dumps(run, indent=2))
    return str(path)


def list_runs() -> list[dict]:
    _ensure()
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
    return {"total_runs": len(runs), "by_endpoint": by_endpoint}
