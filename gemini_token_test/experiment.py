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
                   turns: int | None = None, on_progress=None) -> dict:
    if mode not in MODES:
        mode = "stateless"
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]

    records = []
    # stateless carries the big system prompt every turn; stateful (delta) does not.
    history: list[dict] = [_user(system)] if system else []
    for k, text in enumerate(steps, start=1):
        if on_progress:
            on_progress({"stage": mode, "turn": k, "turns": len(steps)})
        if mode == "stateless":
            history.append(_user(text))
            contents = list(history)
            history.append({"role": "model", "parts": [{"text": "(ack)"}]})
        else:  # stateful = client-side delta: only this step
            contents = [_user(text)]
        res = call_gemini(model, contents, mode=mode, turn=k)
        rec = res.as_dict()
        rec["question"] = text  # pair the sent question with response_text (answer)
        records.append(rec)

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
                    timestamp: str = "", on_progress=None,
                    stage_pause_seconds: float = 0) -> dict:
    """Stage order: stateless -> caches -> stateful -> no-context. Returns one
    combined document.

    When want_capture is set, each stage is captured to its own pcap
    (stateless / cachebuild / stateful / nocontext) so each stage's traffic is
    separable. on_progress(event) is called once per turn with {stage, turn, turns}.
    stage_pause_seconds inserts a pause between stages (spreads calls out to stay
    under Vertex per-minute quotas); during it on_progress emits {stage:'pause'}.
    """
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    def _prog(stage, turn):
        if on_progress:
            on_progress({"stage": stage, "turn": turn, "turns": n})

    def _pause():
        # Space stages out so a burst of turns doesn't trip Vertex RPM/TPM limits.
        # Tick every few seconds so the UI (and the SSE connection) stays alive.
        rem = int(stage_pause_seconds)
        while rem > 0:
            if on_progress:
                on_progress({"stage": "pause", "turn": rem, "turns": int(stage_pause_seconds)})
            step = min(5, rem)
            time.sleep(step)
            rem -= step

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
        _prog("stateless", k)
        history.append(_user(q))
        res = call_gemini(model, list(history), mode="stateless", turn=k)
        ans = res.response_text or ""
        history.append(_model(ans))
        scenario.append({
            "turn": k, "question": q, "answer": ans,
            "req_bytes": res.req_payload_bytes, "resp_bytes": res.resp_payload_bytes,
            "wire_sent": res.wire_sent, "wire_recv": res.wire_recv, "error": res.error,
        })
        rec = res.as_dict()
        rec["question"] = q  # store question alongside response_text (answer)
        stateless_records.append(rec)
    _end(cap, "stateless")
    _pause()

    # --- Stage 2: cumulative caches. cache_k = history[:2k] (k Q&A pairs) -------
    cache_set = []
    cap = _begin("cachebuild")
    for k in range(1, n + 1):
        _prog("cachebuild", k)
        c = create_cache(model, history[:off + 2 * k], CACHE_TTL_SECONDS)
        cache_set.append({
            "k": k, "cache_id": c["name"], "cached_tokens": c["cached_tokens"],
            "skipped": c["name"] is None, "error": c["error"],
        })
    _end(cap, "cachebuild")
    _pause()

    # --- Stage 3: stateful replay. turn k uses cache_(k-1) + question only ------
    stateful_records = []
    cap = _begin("stateful")
    for k, q in enumerate(steps, start=1):
        _prog("stateful", k)
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
        rec["question"] = q  # store question alongside response_text (answer)
        rec["cache_id"] = cache_id
        rec["used_cache"] = cache_id is not None
        stateful_records.append(rec)
    _end(cap, "stateful")
    _pause()

    # --- Stage 4: stateless no-context. The system prompt rides ONLY the first
    # query; every later turn sends just the bare question — no system prompt, no
    # prior question/answer, no cache. Models a naive client that primes context
    # once and then forgets it, so continuity-dependent turns go ambiguous. Run
    # last since it has no dependency on the earlier stages. ----------------------
    nocontext_records = []
    cap = _begin("nocontext")
    for k, q in enumerate(steps, start=1):
        _prog("nocontext", k)
        if k == 1 and system:
            contents = [_user(system), _user(q)]  # system prompt sent with first query only
        else:
            contents = [_user(q)]                 # later turns: bare question, nothing else
        res = call_gemini(model, contents, mode="stateless-nocontext", turn=k)
        rec = res.as_dict()
        rec["question"] = q  # store question alongside response_text (answer)
        nocontext_records.append(rec)
    _end(cap, "nocontext")

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
        "nocontext_records": nocontext_records,
        "stateful_records": stateful_records,
    }


# --- Comparison across arms (the headline experiment) ----------------------

DEFAULT_ARMS = ("stateless", "cached", "interaction")
COMPARE_ARMS = ("stateless", "cached", "interaction", "nocontext")


def _common_from_call(res, arm: str, phase: str, turn: int, question: str) -> dict:
    """Map a CallResult (generateContent) to the shared per-turn record."""
    return {
        "arm": arm, "phase": phase, "turn": turn, "question": question,
        "wire_sent": res.wire_sent, "wire_recv": res.wire_recv,
        "elapsed_ms": res.elapsed_ms,
        "input_tokens": res.prompt_tokens, "cached_tokens": res.cached_tokens,
        "output_tokens": res.resp_tokens, "thought_tokens": res.thought_tokens,
        "total_tokens": res.total_tokens,
        "request_raw": res.request_json, "response_raw": res.response_json,
        "response_text": res.response_text, "error": res.error,
    }


def _arm_stateless(model, system, steps) -> list:
    """Full history every turn: system + q1..qk + a1..a(k-1)."""
    recs = []
    history = [_user(system)] if system else []
    for k, q in enumerate(steps, start=1):
        history.append(_user(q))
        res = call_gemini(model, list(history), mode="stateless", turn=k)
        history.append(_model(res.response_text or ""))
        recs.append(_common_from_call(res, "stateless", "steady", k, q))
    return recs


def _arm_nocontext(model, system, steps) -> list:
    """New question only; the system prompt rides the first turn alone."""
    recs = []
    for k, q in enumerate(steps, start=1):
        contents = [_user(system), _user(q)] if (k == 1 and system) else [_user(q)]
        res = call_gemini(model, contents, mode="stateless-nocontext", turn=k)
        recs.append(_common_from_call(res, "nocontext", "steady", k, q))
    return recs


def _arm_cached(model, system, steps) -> list:
    """Stage 2 (cachebuild, setup bucket) + stage 3 (cached, steady).

    Cache k holds the system prompt + the first k Q&A pairs; turn k>=2 references
    cache_(k-1) and sends only the new question, so the prefix is not re-uploaded.
    Turn 1 has no prior cache and sends system + question.
    """
    n = len(steps)
    off = 1 if system else 0
    # Reconstruct the history the caches are built from by replaying stateless
    # answers is unnecessary here; the cache prefix only needs the questions and a
    # placeholder answer, but to stay faithful we cache system + user turns.
    history = [_user(system)] if system else []
    for q in steps:
        history.append(_user(q))
        history.append(_model("(cached-context placeholder answer)"))

    recs = []
    cache_set = []
    for k in range(1, n + 1):
        c = create_cache(model, history[:off + 2 * k], CACHE_TTL_SECONDS,
                         system_instruction="")
        cache_set.append(c)
        rec = {
            "arm": "cached", "phase": "setup", "turn": k,
            "wire_sent": c.get("wire_sent", 0), "wire_recv": c.get("wire_recv", 0),
            "elapsed_ms": c.get("elapsed_ms", 0),
            "input_tokens": c.get("cached_tokens", 0), "cached_tokens": c.get("cached_tokens", 0),
            "output_tokens": 0, "thought_tokens": 0,
            "total_tokens": c.get("cached_tokens", 0),
            "request_raw": c.get("request_raw", ""), "response_raw": c.get("response_raw", ""),
            "response_text": "", "error": c.get("error", ""),
            "cache_id": c.get("name"), "skipped": c.get("name") is None,
        }
        recs.append(rec)

    for k, q in enumerate(steps, start=1):
        cache = cache_set[k - 2] if k >= 2 else None
        cache_id = cache["name"] if cache else None
        hint = cache["cached_tokens"] if cache else 0
        if cache_id:
            contents = [_user(q)]                                 # prefix server-side
        else:
            contents = history[:off + 2 * (k - 1)] + [_user(q)]   # no cache -> send it
        res = call_gemini(model, contents, mode="stateful", turn=k,
                          cached_content=cache_id, cached_tokens_hint=hint)
        rec = _common_from_call(res, "cached", "steady", k, q)
        rec["cache_id"] = cache_id
        recs.append(rec)

    if os.environ.get("KEEP_CACHE") != "1":
        for c in cache_set:
            if c.get("name"):
                delete_cache(c["name"])
    return recs


def _arm_interaction(model, request_name, turns, on_progress) -> list:
    """Interactions API arm, mapped into the shared per-turn record."""
    from interaction_client import run_interaction   # lazy: avoids import cycle
    out = run_interaction(model, request_name=request_name, turns=turns,
                          on_progress=on_progress)
    recs = []
    for r in out["interaction_records"]:
        recs.append({
            "arm": "interaction", "phase": "steady", "turn": r["turn"],
            "question": r.get("question", ""),
            "wire_sent": r["wire_sent"], "wire_recv": r["wire_recv"],
            "elapsed_ms": r["elapsed_ms"],
            "input_tokens": r["input_tokens"], "cached_tokens": r["cached_tokens"],
            "output_tokens": r["output_tokens"], "thought_tokens": r["thought_tokens"],
            "total_tokens": r["total_tokens"],
            "request_raw": r["request_raw"], "response_raw": r["response_raw"],
            "response_text": r.get("response_text", ""), "error": r.get("error", ""),
            "interaction_id": r.get("interaction_id"),
        })
    return recs


def run_comparison(model: str, request_name: str = "perf",
                   turns: int | None = None, arms=None, on_progress=None,
                   pause_seconds: float = 0) -> dict:
    """Replay one scenario across the arms and collect the shared per-turn record
    for each, so wire bytes, tokens, and latency are comparable. Headline arms are
    stateless, cached, and interaction; nocontext is a lower-bound diagnostic.

    Each arm resets the session first, so it opens a fresh TCP connection and its
    traffic is attributable (and separately capturable). Arms hit the same
    rate-limited project back to back, so pause_seconds spaces them apart; the gap
    goes between arms only, never after the last one. Returns {params, records}.
    """
    arms = list(arms) if arms else list(DEFAULT_ARMS)
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    records: list[dict] = []
    for i, arm in enumerate(arms):
        if i and pause_seconds:
            if on_progress:
                on_progress({"stage": "pause", "turn": pause_seconds, "turns": n})
            time.sleep(pause_seconds)
        reset_session()
        if on_progress:
            on_progress({"stage": arm, "turn": 0, "turns": n})
        if arm == "stateless":
            records += _arm_stateless(model, system, steps)
        elif arm == "cached":
            records += _arm_cached(model, system, steps)
        elif arm == "nocontext":
            records += _arm_nocontext(model, system, steps)
        elif arm == "interaction":
            records += _arm_interaction(model, request_name, n, on_progress)

    return {
        "params": {"mode": "comparison", "turns": n, "model": model,
                   "arms": arms, "endpoint": ENDPOINT, "request_source": source,
                   "pause_seconds": pause_seconds},
        "records": records,
    }
