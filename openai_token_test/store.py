"""Persist a run: metrics, raw bodies, CSV — everything needed to re-check the claim.

Layout under results/runs/<exec_id>/:

    run.json        config + per-turn metrics + summary + capture info
    summary.csv     one row per (arm, turn), the table the charts are drawn from
    charts.png      cumulative upload bytes and cumulative input tokens
    bodies/         the JSON actually sent and received, one file per call

The bodies are the part that makes this auditable rather than merely tabulated.
A number saying "the stateless arm uploaded 21 kB on turn 1" is a claim; the
21 kB file sitting next to it is the evidence. Bodies only — never headers, which
carry the bearer token.

Local files only. No Firestore, no cloud: this experiment runs on one machine and
the result is a handful of megabytes.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RUNS_DIR = RESULTS_DIR / "runs"

_SAFE_EXEC_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")


def new_exec_id() -> str:
    import secrets
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{secrets.token_hex(4)}"


def run_dir(exec_id: str) -> Path | None:
    """Map an exec_id to its directory, refusing anything that could traverse out."""
    if not _SAFE_EXEC_ID.match(exec_id):
        return None
    p = (RUNS_DIR / exec_id).resolve()
    if p.parent != RUNS_DIR.resolve():
        return None
    return p


def _body_name(arm: str, repeat: int, turn: int, kind: str) -> str:
    # turn 0 is the conversation-create setup call, not a model turn
    stem = "setup" if turn == 0 else f"turn{turn:02d}"
    return f"{arm}_r{repeat}_{stem}_{kind}.json"


def save_bodies(exec_id: str, experiment: dict) -> int:
    """Write every request/response body seen during the run. Returns file count."""
    d = run_dir(exec_id)
    if d is None:
        raise ValueError(f"bad exec_id: {exec_id!r}")
    bodies = d / "bodies"
    bodies.mkdir(parents=True, exist_ok=True)

    written = 0
    for run in experiment["runs"]:
        arm, repeat = run["arm"], run["repeat"]
        calls = ([run["setup"]] if run.get("setup") else []) + run["turns"]
        for call in calls:
            for kind in ("request", "response"):
                raw = call.get(f"{kind}_json")
                if not raw:
                    continue
                name = _body_name(arm, repeat, call["turn"], kind)
                (bodies / name).write_text(raw)
                written += 1
    return written


def write_csv(summary: dict, path: Path) -> Path:
    rows = []
    for arm, s in summary["arms"].items():
        for k in range(s["turns"]):
            rows.append({
                "arm": arm,
                "turn": k + 1,
                "upload_bytes": round(s["per_turn"]["req_payload_bytes"][k]),
                "wire_sent": round(s["per_turn"]["wire_sent"][k]),
                "download_bytes": round(s["per_turn"]["resp_payload_bytes"][k]),
                "input_tokens": round(s["per_turn"]["input_tokens"][k]),
                "cached_tokens": round(s["per_turn"]["cached_tokens"][k]),
                "billed_uncached_tokens": round(s["per_turn"]["billed_uncached_tokens"][k]),
                "output_tokens": round(s["per_turn"]["output_tokens"][k]),
                "latency_ms": round(s["per_turn"]["latency_ms"][k]),
                "cum_upload_bytes": round(s["cum_req_bytes"][k]),
                "cum_input_tokens": round(s["cum_input_tokens"][k]),
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def save_run(exec_id: str, experiment: dict, summary: dict,
             captures: list[dict] | None = None) -> dict:
    """Write run.json, summary.csv, charts.png and the raw bodies. Returns a manifest."""
    d = run_dir(exec_id)
    if d is None:
        raise ValueError(f"bad exec_id: {exec_id!r}")
    d.mkdir(parents=True, exist_ok=True)

    doc = {
        "exec_id": exec_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": experiment["config"],
        "summary": summary,
        "captures": captures or [],
        # per-turn metrics without the bodies, which live in bodies/ instead
        "runs": [
            {
                "arm": r["arm"],
                "repeat": r["repeat"],
                "setup": _strip_bodies(r.get("setup")),
                "turns": [_strip_bodies(t) for t in r["turns"]],
            }
            for r in experiment["runs"]
        ],
    }
    (d / "run.json").write_text(json.dumps(doc, indent=2))

    write_csv(summary, d / "summary.csv")

    import report
    report.write_charts(summary, d / "charts.png")

    n_bodies = save_bodies(exec_id, experiment)

    return {
        "exec_id": exec_id,
        "dir": str(d),
        "bodies": n_bodies,
        "captures": len(captures or []),
    }


def _strip_bodies(call: dict | None) -> dict | None:
    if not call:
        return None
    return {k: v for k, v in call.items()
            if k not in ("request_json", "response_json")}


def list_runs(limit: int = 50) -> list[dict]:
    """Newest first. One line per run, enough for the history panel."""
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True)[:limit]:
        f = d / "run.json"
        if not f.is_file():
            continue
        try:
            doc = json.loads(f.read_text())
        except Exception:
            continue
        cfg = doc.get("config", {})
        ratios = (doc.get("summary") or {}).get("ratios", {})
        first = next(iter(ratios.values()), {})
        out.append({
            "exec_id": doc.get("exec_id", d.name),
            "timestamp": doc.get("timestamp", ""),
            "model": cfg.get("model", ""),
            "fixture": cfg.get("fixture", ""),
            "turns": cfg.get("turns", 0),
            "repeats": cfg.get("repeats", 0),
            "upload_ratio": first.get("upload_bytes", 0),
            "token_ratio": first.get("input_tokens", 0),
            "captures": len(doc.get("captures") or []),
        })
    return out


def load_run(exec_id: str) -> dict | None:
    d = run_dir(exec_id)
    if d is None:
        return None
    f = d / "run.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())
