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
import time
from pathlib import Path

from gemini_client import (
    call_gemini, create_cache, delete_cache, reset_session, ENDPOINT,
)

MODES = ("stateless", "stateful")
REQUESTS_DIR = Path(__file__).resolve().parent / "requests"
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "1800"))
# Per-stage capture pacing: settle time after tcpdump starts before the first
# request (clean handshake), and drain time after the socket closes before
# tcpdump stops (clean teardown). Both configurable via env.
CAPTURE_WARMUP_SECONDS = float(os.environ.get("CAPTURE_WARMUP_SECONDS", "2"))
CAPTURE_DRAIN_SECONDS = float(os.environ.get("CAPTURE_DRAIN_SECONDS", "1"))


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
                    turns: int | None = None, want_capture: bool = False,
                    timestamp: str = "") -> dict:
    """Stage 1 stateless scenario -> Stage 2 cumulative caches -> Stage 3 stateful
    replay (cache + question only). Returns one combined document.

    When want_capture is set, each stage is captured to its own pcap
    (stateless / cachebuild / stateful) so the traffic of each stage is separable.
    """
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    import capture as pcap  # lazy: avoids hard dependency when capture unused
    pcaps: dict = {}

    def _begin(stage_mode):
        if not want_capture:
            return None
        # Drop any pooled socket from the previous stage so this stage opens a
        # fresh TCP connection; start tcpdump, then wait so the first request's
        # 3-way handshake lands inside this stage's pcap.
        reset_session()
        cap = pcap.Capture(timestamp or "0", stage_mode)
        cap.__enter__()
        time.sleep(CAPTURE_WARMUP_SECONDS)
        return cap

    def _end(cap, key):
        if cap is None:
            return
        # Close the socket while tcpdump is still running so the FIN teardown is
        # captured, drain briefly, then stop the capture.
        reset_session()
        time.sleep(CAPTURE_DRAIN_SECONDS)
        cap.__exit__(None, None, None)
        r = cap.result()
        if r.get("ok") and r.get("file"):
            r["download"] = f"/download/pcap/{r['file']}"
        pcaps[key] = r

    # --- Stage 1: stateless scenario, capture every request + response ---------
    # The big system prompt sits at history[0] and is resent every stateless turn;
    # it becomes part of every cache (off=1 accounts for it in the indices below).
    scenario, stateless_records = [], []
    off = 1 if system else 0
    history: list[dict] = [_user(system)] if system else []
    cap = _begin("stateless")
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
    _end(cap, "stateless")

    # --- Stage 2: cumulative caches. cache_k = history[:2k] (k Q&A pairs) -------
    cache_set = []
    cap = _begin("cachebuild")
    for k in range(1, n + 1):
        c = create_cache(model, history[:off + 2 * k], CACHE_TTL_SECONDS)
        cache_set.append({
            "k": k, "cache_id": c["name"], "cached_tokens": c["cached_tokens"],
            "skipped": c["name"] is None, "error": c["error"],
        })
    _end(cap, "cachebuild")

    # --- Stage 3: stateful replay. turn k uses cache_(k-1) + question only ------
    stateful_records = []
    cap = _begin("stateful")
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
    _end(cap, "stateful")

    # --- cleanup caches (best-effort) ------------------------------------------
    if os.environ.get("KEEP_CACHE") != "1":
        for c in cache_set:
            if c["cache_id"]:
                delete_cache(c["cache_id"])

    return {
        "params": {"mode": "caching-3stage", "turns": n, "model": model,
                   "endpoint": ENDPOINT, "request_source": source},
        "pcaps": pcaps,
        "scenario": scenario,
        "cache_set": cache_set,
        "stateless_records": stateless_records,
        "stateful_records": stateful_records,
    }
