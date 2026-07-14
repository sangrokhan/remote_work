"""Replay one N-turn conversation across the arms and collect what each turn cost.

The arms are the ways of keeping a conversation going: the client resends the whole
history (stateless), the server keeps it (interaction, interaction_inline), the
prefix lives in an explicit cache (cached), or nothing is kept at all (nocontext,
the lower bound). The scenario -- one system prompt and a fixed list of questions --
is the same for all of them, so the only thing that varies is who stores the context.

Every turn produces the same record: wire bytes sent and received, tokens in and
out, and five clocks (req_sent, ttfb, ttft, ttlt, turn_end), because one latency
number cannot separate a history still going up the wire from a model thinking from
a server persisting the turn.
"""

from __future__ import annotations

import json
import os
import time

import capture as pcap
from pathlib import Path

from gemini_client import (
    call_gemini, create_cache, delete_cache, reset_session, ENDPOINT,
)
from payloads import user_content as _user, model_content as _model
from payloads import model_content_from_response

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


# --- Comparison across arms (the headline experiment) ----------------------

DEFAULT_ARMS = ("stateless", "cached", "interaction", "interaction_inline",
                "interaction_stateless")
COMPARE_ARMS = ("stateless", "cached", "interaction", "interaction_inline",
                "interaction_stateless", "nocontext")


def _model_turn(res) -> dict:
    """The model's turn as generateContent sent it, for a client that keeps history.

    The candidate's parts carry `thoughtSignature` alongside the text. Rebuilding the
    turn from `response_text` throws the signature away, which is not what a real
    client does -- and it hides ~1 KB of upload per turn from the arm that is
    supposed to be paying for its own history. Falls back to the text when the call
    errored and there is no candidate to echo.
    """
    try:
        data = json.loads(res.response_json or "{}")
    except Exception:
        data = {}
    return model_content_from_response(data, fallback_text=res.response_text or "")


def _common_from_call(res, arm: str, phase: str, turn: int, question: str) -> dict:
    """Map a CallResult (generateContent) to the shared per-turn record."""
    return {
        "arm": arm, "phase": phase, "turn": turn, "question": question,
        "wire_sent": res.wire_sent, "wire_recv": res.wire_recv,
        "elapsed_ms": res.elapsed_ms,
        # Five marks, because one number cannot separate "my history is still going
        # up the wire" from "the model is thinking" from "the server is persisting".
        "req_sent_ms": res.req_sent_ms, "ttfb_ms": res.ttfb_ms,
        "ttft_ms": res.ttft_ms, "ttlt_ms": res.ttlt_ms,
        "turn_end_ms": res.turn_end_ms or res.elapsed_ms,
        "input_tokens": res.prompt_tokens, "cached_tokens": res.cached_tokens,
        "output_tokens": res.resp_tokens, "thought_tokens": res.thought_tokens,
        "total_tokens": res.total_tokens,
        "request_raw": res.request_json, "response_raw": res.response_json,
        "response_text": res.response_text, "error": res.error,
    }


def _tick(on_progress, arm: str, phase: str, turn: int, turns: int) -> None:
    """Announce the call about to be made. Without this an arm sits at turn 0/N for
    its whole run and a stall is indistinguishable from progress."""
    if on_progress:
        on_progress({"stage": arm, "phase": phase, "turn": turn, "turns": turns})


def _arm_stateless(model, system, steps, on_progress=None) -> list:
    """Full history every turn: system + q1..qk + a1..a(k-1)."""
    recs = []
    n = len(steps)
    history = [_user(system)] if system else []
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "stateless", "steady", k, n)
        history.append(_user(q))
        res = call_gemini(model, list(history), mode="stateless", turn=k)
        history.append(_model_turn(res))
        recs.append(_common_from_call(res, "stateless", "steady", k, q))
    return recs


def _arm_nocontext(model, system, steps, on_progress=None) -> list:
    """New question only; the system prompt rides the first turn alone."""
    recs = []
    n = len(steps)
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "nocontext", "steady", k, n)
        contents = [_user(system), _user(q)] if (k == 1 and system) else [_user(q)]
        res = call_gemini(model, contents, mode="stateless-nocontext", turn=k)
        recs.append(_common_from_call(res, "nocontext", "steady", k, q))
    return recs


def _arm_cached_prep(model, system, steps, transcript, on_progress=None) -> tuple:
    """Build one cache per turn from the real stateless transcript. Runs before the
    capture window opens; its cost is not part of what is being measured.

    `transcript` is what the model actually answered in the stateless arm. The
    caches are built from it, not from a placeholder: a cache of a conversation that
    never happened measures nothing.

    Cache k holds the system prompt plus the first k real Q&A pairs. Returns
    (records, cache_set); cache_set is the state the steady stage and the teardown
    stage both need.

    The generation calls are recorded under the `cachegen` phase and left out of the
    comparison totals. They are preparation, not the thing being measured -- and
    counting them would drown everything else, since each build re-uploads the whole
    system prompt and n turns then cost O(n^2).
    """
    n = len(steps)
    off = 1 if system else 0
    history = [_user(system)] if system else []
    for q, a in zip(steps, transcript):
        history.append(_user(q))
        history.append(_model(a))

    recs = []
    cache_set = []
    for k in range(1, n + 1):
        _tick(on_progress, "cached", "cachegen", k, n)
        c = create_cache(model, history[:off + 2 * k], CACHE_TTL_SECONDS,
                         system_instruction="")
        cache_set.append(c)
        recs.append({
            "arm": "cached", "phase": "cachegen", "turn": k, "question": "",
            "wire_sent": c.get("wire_sent", 0), "wire_recv": c.get("wire_recv", 0),
            "elapsed_ms": c.get("elapsed_ms", 0),
            # A cache build is a plain POST, not a generation: no tokens stream out
            # of it, so every mark is the one number it has.
            "req_sent_ms": c.get("elapsed_ms", 0), "ttfb_ms": c.get("elapsed_ms", 0),
            "ttft_ms": c.get("elapsed_ms", 0), "ttlt_ms": c.get("elapsed_ms", 0),
            "turn_end_ms": c.get("elapsed_ms", 0),
            "input_tokens": c.get("cached_tokens", 0), "cached_tokens": c.get("cached_tokens", 0),
            "output_tokens": 0, "thought_tokens": 0,
            "total_tokens": c.get("cached_tokens", 0),
            "request_raw": c.get("request_raw", ""), "response_raw": c.get("response_raw", ""),
            "response_text": "", "error": c.get("error", ""),
            "cache_id": c.get("name"), "skipped": c.get("name") is None,
        })
    return recs, cache_set


def _arm_cached_steady(model, system, steps, cache_set, on_progress=None) -> list:
    """The measured turns. Turn k >= 2 references cache_(k-1) and sends only the new
    question, so the prefix never goes back over the wire; turn 1 has no prior cache
    and sends the system prompt with its question, exactly as stateless would.

    Runs inside the capture window, on a fresh connection -- this is the thing being
    measured.
    """
    n = len(steps)
    recs = []
    for k, q in enumerate(steps, start=1):
        _tick(on_progress, "cached", "steady", k, n)
        cache = cache_set[k - 2] if k >= 2 else None
        cache_id = cache["name"] if cache else None
        hint = cache["cached_tokens"] if cache else 0
        if cache_id:
            contents = [_user(q)]                                 # prefix server-side
        else:
            contents = ([_user(system)] if system else []) + [_user(q)]
        res = call_gemini(model, contents, mode="stateful", turn=k,
                          cached_content=cache_id, cached_tokens_hint=hint)
        rec = _common_from_call(res, "cached", "steady", k, q)
        rec["cache_id"] = cache_id
        recs.append(rec)
    return recs


def _arm_cached_teardown(cache_set) -> None:
    """Delete the caches built in prep. Runs after the capture window closes and
    after wall_ms is recorded; its cost is neither captured nor timed, and (like
    today) its DELETE calls are not recorded at all."""
    if os.environ.get("KEEP_CACHE") == "1":
        return
    for c in cache_set or []:
        if c.get("name"):
            delete_cache(c["name"])
    # The DELETEs above ran on a freshly-opened session and would otherwise leave
    # it open for the *next* arm's pre-capture close to tear down with no settle --
    # putting this arm's teardown FIN/ACK in the next arm's pcap, the same leak
    # _close_connection exists to prevent. Close it here, at no settle: nothing is
    # capturing right now, so there is nothing for the FIN to land in.
    _close_connection()


def _arm_interaction(model, request_name, turns, on_progress,
                     arm: str = "interaction", inline_system: bool = False,
                     client_history: bool = False) -> list:
    """Interactions API arm, mapped into the shared per-turn record.

    inline_system moves the system prompt out of system_instruction and into the
    first user turn, so the server-side history carries it and later turns send only
    their question. Same content reaches the model; a different party stores it.

    client_history takes the server-side history away entirely: store:false, no
    previous_interaction_id, and the whole conversation resent as steps every turn.
    """
    from interaction_client import run_interaction   # lazy: avoids import cycle
    out = run_interaction(model, request_name=request_name, turns=turns,
                          on_progress=on_progress, inline_system=inline_system,
                          stage=arm, client_history=client_history)
    recs = []
    for r in out["interaction_records"]:
        recs.append({
            "arm": arm, "phase": "steady", "turn": r["turn"],
            "question": r.get("question", ""),
            "wire_sent": r["wire_sent"], "wire_recv": r["wire_recv"],
            "elapsed_ms": r["elapsed_ms"],
            "req_sent_ms": r["req_sent_ms"], "ttfb_ms": r["ttfb_ms"],
            "ttft_ms": r["ttft_ms"], "ttlt_ms": r["ttlt_ms"],
            "turn_end_ms": r["turn_end_ms"],
            "input_tokens": r["input_tokens"], "cached_tokens": r["cached_tokens"],
            "output_tokens": r["output_tokens"], "thought_tokens": r["thought_tokens"],
            "total_tokens": r["total_tokens"],
            "request_raw": r["request_raw"], "response_raw": r["response_raw"],
            "response_text": r.get("response_text", ""), "error": r.get("error", ""),
            "interaction_id": r.get("interaction_id"),
        })
    return recs


def _close_connection(settle: float = 0.0) -> None:
    """Close the arm's sockets while its capture is still running.

    keep-alive means an arm's TCP connection outlives its last call. Closing it at
    the *start* of the next arm — which is what reset_session() used to do — puts
    this arm's FIN, and the peer's FIN/ACK, into the *next* arm's pcap: a stray
    teardown from "the previous test". Close it here instead, so each pcap is one
    self-contained SYN..FIN conversation and the next arm starts from a fresh
    connection either way.

    The teardown is a round trip (our FIN, their FIN/ACK) and tcpdump still has to
    flush, so a capture needs a beat before it is torn down or the FIN is simply
    missing from the file — intermittently, which is worse than always. With no
    capture running there is nothing to wait for, so settle is 0.
    """
    reset_session()
    if settle:
        time.sleep(settle)


def _settle_seconds() -> float:
    return float(os.environ.get("PCAP_SETTLE_SECONDS", "1.0"))


def _arm_prep(arm, model, system, steps, on_progress=None, transcript=None) -> tuple:
    """Runs before the capture window opens. A no-op for every arm except cached,
    which needs its caches built first. Returns (records, state); state is handed
    to the matching steady/teardown call."""
    if arm == "cached":
        return _arm_cached_prep(model, system, steps, transcript or [], on_progress)
    return [], None


def _arm_steady(arm, model, system, steps, request_name, n, on_progress, state) -> list:
    """The measured stage: runs inside the capture window, on a fresh connection."""
    if arm == "stateless":
        return _arm_stateless(model, system, steps, on_progress)
    if arm == "cached":
        return _arm_cached_steady(model, system, steps, state, on_progress)
    if arm == "nocontext":
        return _arm_nocontext(model, system, steps, on_progress)
    if arm == "interaction":
        return _arm_interaction(model, request_name, n, on_progress)
    if arm == "interaction_inline":
        return _arm_interaction(model, request_name, n, on_progress,
                                arm="interaction_inline", inline_system=True)
    if arm == "interaction_stateless":
        return _arm_interaction(model, request_name, n, on_progress,
                                arm="interaction_stateless", client_history=True)
    return []


def _arm_teardown(arm, state) -> None:
    """Runs after the capture window closes and after wall_ms is recorded. A no-op
    for every arm except cached, which deletes the caches it built in prep."""
    if arm == "cached":
        _arm_cached_teardown(state)


def _order_arms(arms: list) -> list:
    """stateless first when cached is asked for: the cached arm caches the answers
    stateless got, so the transcript has to exist before the caches are built."""
    arms = list(dict.fromkeys(arms))
    if "cached" in arms and "stateless" not in arms:
        arms.insert(0, "stateless")
    return sorted(arms, key=lambda a: 0 if a == "stateless" else 1)


def run_comparison(model: str, request_name: str = "perf",
                   turns: int | None = None, arms=None, on_progress=None,
                   pause_seconds: float = 0, want_capture: bool = False,
                   timestamp: str = "") -> dict:
    """Replay one scenario across the arms and collect the shared per-turn record
    for each, so wire bytes, tokens, and latency are comparable. Headline arms are
    stateless, cached, and interaction; nocontext is a lower-bound diagnostic.

    Each arm resets the session first, so it opens a fresh TCP connection and its
    traffic is attributable -- and, with want_capture, separately capturable: one
    pcap per arm, which is what lets the socket-level wire counter be cross-checked
    against what actually went out on the wire.

    Arms hit the same rate-limited project back to back, so pause_seconds spaces
    them apart; the gap goes between arms only, never after the last one.

    Returns {params, records, pcaps, wall_ms}. wall_ms covers only the arm's steady
    stage -- the same window the pcap captures -- not any prep (e.g. the cached
    arm's cache builds) or teardown (its cache deletes), which run outside both.
    """
    arms = _order_arms(list(arms) if arms else list(DEFAULT_ARMS))
    system, steps, source = load_request(request_name)
    if turns:
        steps = steps[:max(1, min(turns, len(steps)))]
    n = len(steps)

    records: list[dict] = []
    pcaps: dict = {}
    wall_ms: dict = {}
    transcript: list[str] = []   # what stateless got back; the cached arm caches it
    # Drop anything left pooled by an earlier request (the probe, the model list):
    # its FIN would otherwise be the first thing the first arm's pcap records.
    reset_session()
    for i, arm in enumerate(arms):
        if i and pause_seconds:
            # Tick per second rather than sleeping the whole gap in one go: a single
            # "pausing" event followed by a minute of silence is indistinguishable
            # from a hang, and this pause is routinely a minute long.
            for remaining in range(int(pause_seconds), 0, -1):
                if on_progress:
                    on_progress({"stage": "pause", "remaining": remaining,
                                 "pause_total": int(pause_seconds),
                                 "next_arm": arm, "turns": n})
                time.sleep(1)
        if on_progress:
            on_progress({"stage": arm, "turn": 0, "turns": n})

        prep_recs, state = _arm_prep(arm, model, system, steps, on_progress, transcript)
        # Prep (e.g. the cached arm's cache builds) runs on the shared session,
        # outside the capture window, and leaves a keep-alive connection open behind
        # it. If the steady stage reused that connection, its capture would open
        # onto an already-established connection (no SYN in the pcap) and prep's own
        # FIN could land inside the steady pcap. Close it now, before the window
        # opens, so the steady stage always starts from a fresh connection -- same
        # hazard _close_connection's own docstring is about, one arm earlier.
        #
        # This close needs the same settle as the post-capture one when a capture
        # is about to open: tcpdump starts within ~1ms of the `with` block below,
        # but the peer's FIN/ACK for *this* close takes a round trip to arrive, so
        # with no settle it lands inside the steady pcap anyway -- the exact
        # pollution this restructuring exists to remove, just moved earlier. With
        # no capture running there is nothing to keep clean, so no settle is spent.
        # Runs before t0 so it must never count toward wall_ms.
        _close_connection(_settle_seconds() if want_capture else 0.0)

        t0 = time.monotonic()
        if want_capture:
            with pcap.Capture(timestamp, arm) as cap:
                steady_recs = _arm_steady(arm, model, system, steps, request_name, n,
                                          on_progress, state)
                _close_connection(_settle_seconds())
            pcaps[arm] = cap.result()
        else:
            steady_recs = _arm_steady(arm, model, system, steps, request_name, n,
                                      on_progress, state)
            _close_connection()
        wall_ms[arm] = int((time.monotonic() - t0) * 1000)

        # Teardown (e.g. deleting the cached arm's caches) runs after the capture
        # window has closed and after wall_ms is recorded, so neither counts it.
        _arm_teardown(arm, state)

        records += prep_recs + steady_recs
        if arm == "stateless":
            transcript = [r["response_text"] for r in steady_recs
                          if r["phase"] == "steady"]

    return {
        "params": {"mode": "comparison", "turns": n, "model": model,
                   "arms": arms, "endpoint": ENDPOINT, "request_source": source,
                   "pause_seconds": pause_seconds, "capture": want_capture},
        "records": records,
        "pcaps": pcaps,
        "wall_ms": wall_ms,
    }
