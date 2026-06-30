"""Run an N-turn conversation in ONE mode and collect per-turn metrics.

stateless: turn k sends the full history (steps 1..k)  -> O(N^2) tokens
stateful : turn k sends only step k (client-side delta) -> O(N) tokens

Request texts are fixed, loaded from a JSON request file, so the request is
constant even when the response size varies. Only one mode runs per execution;
compare modes by loading two executions from history.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from gemini_client import (
    call_gemini, create_cache, delete_cache, ENDPOINT,
)

MODES = ("stateless", "stateful")
REQUESTS_DIR = Path(__file__).resolve().parent / "requests"
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "1800"))


def load_request(name: str = "default") -> tuple[str, list[str], str]:
    """Return (system_prompt, step_texts, source_label).

    The system prompt is a large fixed prefix (persona + tool descriptions) that
    is reused every turn — ideal for caching (>=2048 tokens). Falls back to
    synthetic if the file is missing.
    """
    path = REQUESTS_DIR / f"{name}.json"
    try:
        data = json.loads(path.read_text())
        steps = [s["text"] for s in data.get("steps", []) if s.get("text")]
        system = data.get("system", "")
        if isinstance(system, list):
            system = "\n\n".join(system)
        if steps:
            return system, steps, f"file:{name}.json"
    except Exception:
        pass
    steps = [f"Turn {k}. " + ("the quick brown fox. " * 8) for k in range(1, 9)]
    return "", steps, "synthetic"


def _user(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def run_experiment(mode: str, model: str, request_name: str = "default",
                   turns: int | None = None) -> dict:
    if mode not in MODES:
        mode = "stateless"
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]

    records = []
    # stateless carries the big system prompt every turn; stateful (delta) does not.
    history: list[dict] = [_user(system)] if system else []
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


def _model(text: str) -> dict:
    return {"role": "model", "parts": [{"text": text}]}


def run_three_stage(model: str, request_name: str = "default",
                    turns: int | None = None) -> dict:
    """Stage 1 stateless scenario -> Stage 2 cumulative caches -> Stage 3 stateful
    replay (cache + question only). Returns one combined document."""
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    # --- Stage 1: stateless scenario, capture every request + response ---------
    # The big system prompt sits at history[0] and is resent every stateless turn;
    # it becomes part of every cache (off=1 accounts for it in the indices below).
    scenario, stateless_records = [], []
    off = 1 if system else 0
    history: list[dict] = [_user(system)] if system else []
    for k, q in enumerate(steps, start=1):
        history.append(_user(q))
        res = call_gemini(model, list(history), mode="stateless", turn=k)
        ans = res.response_text or ""
        history.append(_model(ans))
        scenario.append({
            "turn": k, "question": q, "answer": ans,
            "req_bytes": res.req_payload_bytes, "resp_bytes": res.resp_payload_bytes,
            "wire_sent": res.wire_sent, "wire_recv": res.wire_recv, "error": res.error,
        })
        stateless_records.append(res.as_dict())

    # --- Stage 2: cumulative caches. cache_k = history[:2k] (k Q&A pairs) -------
    cache_set = []
    for k in range(1, n + 1):
        c = create_cache(model, history[:off + 2 * k], CACHE_TTL_SECONDS)
        cache_set.append({
            "k": k, "cache_id": c["name"], "cached_tokens": c["cached_tokens"],
            "skipped": c["name"] is None, "error": c["error"],
        })

    # --- Stage 3: stateful replay. turn k uses cache_(k-1) + question only ------
    stateful_records = []
    for k, q in enumerate(steps, start=1):
        cache = cache_set[k - 2] if k >= 2 else None
        cache_id = cache["cache_id"] if cache else None
        hint = cache["cached_tokens"] if cache else 0
        if cache_id:
            contents = [_user(q)]                       # prefix is server-side
        else:
            contents = history[:off + 2 * (k - 1)] + [_user(q)]  # no cache -> send it
        res = call_gemini(model, contents, mode="stateful", turn=k,
                          cached_content=cache_id, cached_tokens_hint=hint)
        rec = res.as_dict()
        rec["cache_id"] = cache_id
        rec["used_cache"] = cache_id is not None
        stateful_records.append(rec)

    # --- cleanup caches (best-effort) ------------------------------------------
    if os.environ.get("KEEP_CACHE") != "1":
        for c in cache_set:
            if c["cache_id"]:
                delete_cache(c["cache_id"])

    return {
        "params": {"mode": "caching-3stage", "turns": n, "model": model,
                   "endpoint": ENDPOINT, "request_source": source},
        "scenario": scenario,
        "cache_set": cache_set,
        "stateless_records": stateless_records,
        "stateful_records": stateful_records,
    }
