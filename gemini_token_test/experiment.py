"""Run an N-turn conversation in ONE mode and collect per-turn metrics.

stateless: turn k sends the full history (steps 1..k)  -> O(N^2) tokens
stateful : turn k sends only step k (client-side delta) -> O(N) tokens

Request texts are fixed, loaded from a JSON request file, so the request is
constant even when the response size varies. Only one mode runs per execution;
compare modes by loading two executions from history.
"""

from __future__ import annotations

import json
from pathlib import Path

from gemini_client import call_gemini, ENDPOINT

MODES = ("stateless", "stateful")
REQUESTS_DIR = Path(__file__).resolve().parent / "requests"


def load_request_steps(name: str = "default") -> tuple[list[str], str]:
    """Return (step_texts, source_label). Falls back to synthetic if missing."""
    path = REQUESTS_DIR / f"{name}.json"
    try:
        data = json.loads(path.read_text())
        steps = [s["text"] for s in data.get("steps", []) if s.get("text")]
        if steps:
            return steps, f"file:{name}.json"
    except Exception:
        pass
    # Fallback: deterministic synthetic steps.
    steps = [f"Turn {k}. " + ("the quick brown fox. " * 8) for k in range(1, 9)]
    return steps, "synthetic"


def _user(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def run_experiment(mode: str, model: str, request_name: str = "default",
                   turns: int | None = None) -> dict:
    if mode not in MODES:
        mode = "stateless"
    steps, source = load_request_steps(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]

    records = []
    history: list[dict] = []
    for k, text in enumerate(steps, start=1):
        if mode == "stateless":
            history.append(_user(text))
            contents = list(history)
            history.append({"role": "model", "parts": [{"text": "(ack)"}]})
        else:  # stateful = client-side delta: only this step
            contents = [_user(text)]
        res = call_gemini(model, contents, mode=mode, turn=k)
        records.append(res.as_dict())

    return {
        "params": {
            "mode": mode,
            "turns": len(steps),
            "model": model,
            "endpoint": ENDPOINT,
            "request_source": source,
        },
        "records": records,
    }
