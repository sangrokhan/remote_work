"""Replay one scenario across providers and arms, and keep the runs attributable.

The arms are the experiment; this module is the harness around them. What it owns is
everything that would make two arms incomparable if it were done differently for each
one: the connection they open, the window a capture covers, the clock that says how
long the measured stage took, and the order they run in.

Three rules the arms depend on and cannot enforce themselves:

  A fresh connection per arm. The session pools TLS connections, so without a reset
  the second arm rides the first one's socket: its pcap opens onto an established
  connection with no handshake in it, and the first arm's teardown lands inside the
  second arm's capture.

  Prep runs outside the window. A Gemini cache build re-uploads the whole system
  prompt, and a run of n turns then costs O(n^2) in setup alone. Counting that as the
  arm's traffic would drown every number the arm exists to produce, so prep records
  are phased and core.metrics keeps them out of the totals -- and the capture window
  opens only after prep has finished and its connection has been closed.

  wall_ms covers the measured stage only. It is the same window the pcap covers, so
  the two can be read against each other.

  A cold prefix per arm. Every arm sends the same system prompt, and both vendors cache
  on the prefix, so the arm that runs second reads the cache the first one left warm --
  measured: a stateful arm billed 4224 cached tokens on its own turn 1. core.cachebust
  puts a per-(run, provider, arm) marker on the front of the prompt, and this is where
  it is applied, because the arms cannot see each other and cannot know they are being
  compared.
"""

from __future__ import annotations

import time

from core import cachebust
from core import capture as pcap
from core import wire
from providers import base

# What a turn is measured for. Bytes come from a blocking call because a streamed
# one is padded (OpenAI) and framed (both); the marks come from a streamed call
# because a blocking one has no first token to time. `both` pays for two calls and
# is never a default.
MEASURES = ("bytes", "latency", "both")

# The arm where `both` is not merely expensive but wrong: every pass carries the
# conversation id, and OpenAI appends each pass to the shared conversation object, so
# the second call of turn k makes turn k+1's input_tokens count turn k twice.
#
# `responses` is deliberately absent. It carries previous_response_id, not a conversation:
# the two passes branch from the same parent rather than appending to one object, and the
# chain follows the blocking pass. The streamed pass is an orphan branch -- it costs money
# and corrupts nothing.
_BOTH_IS_UNSAFE = {("openai", "responses_inline")}

# Between arms, not after the last one: the gap exists to keep a rate-limited project
# from refusing the next arm, and a pause at the end delays nothing but the operator.
_SETTLE_SECONDS = 1.0


def plan(providers: dict | None = None) -> list[tuple[str, str]]:
    """The (provider, arm) pairs a run will execute, in order.

    `providers` maps a provider name to the arms wanted from it; None means every
    provider's headline arms. Arms run grouped by provider so that one provider's
    rate limit cannot be tripped by the other's burst.
    """
    out: list[tuple[str, str]] = []
    for name in (providers or {n: None for n in base.names()}):
        mod = base.get(name)
        arms = (providers or {}).get(name) or list(mod.HEADLINE_ARMS)
        for arm in arms:
            if arm not in mod.ARMS:
                raise ValueError(f"{name} has no arm {arm!r}")
            out.append((name, arm))
    return out


def warnings_for(pairs: list[tuple[str, str]], measure: str) -> list[str]:
    """What the operator should know before the calls go out, not after they are
    billed."""
    out = []
    for provider, arm in pairs:
        if measure == "both" and (provider, arm) in _BOTH_IS_UNSAFE:
            out.append(
                f"{provider}:{arm} cannot be measured with `both`: both passes carry "
                "the conversation id, the server appends each of them, and every "
                "later turn is then billed for this turn twice. Run it with `bytes` "
                "or `latency`.")
    return out


def run(providers: dict | None = None, *, system: str, steps: list[str],
        measure: str = "bytes", models: dict | None = None,
        want_capture: bool = False, pause_seconds: float = 0,
        timestamp: str = "", cache_bust: bool | None = None,
        prefix_drift: bool = False, on_progress=None) -> dict:
    """Replay `steps` across every (provider, arm) pair and return the run document.

    Returns {params, records, pcaps, wall_ms}. Records carry provider and arm, so one
    run holds both vendors and a reader can group either way.

    `cache_bust=None` defers to TRAFFIC_CACHE_BUST (on unless set to 0). False makes
    every arm send the byte-identical system prompt again, which is what the vendors'
    implicit prefix caches feed on -- worth doing deliberately, never by accident.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}, not {measure!r}")

    pairs = plan(providers)
    warnings = warnings_for(pairs, measure)
    models = dict(models or {})

    # Before the first arm: every tag a prompt or a cache key will carry is derived from
    # this timestamp, and the openai adapter reads its own tag out of here rather than
    # being handed one, so the run has to be open before any arm runs.
    cachebust.begin(timestamp, cache_bust, drift=prefix_drift)
    if prefix_drift:
        # The negative control, and it has to announce itself: every number it produces is
        # meant to be worse, and a reader who did not know the knob was on would file the
        # collapse in cached_tokens as a cache outage rather than as the result.
        warnings.append(
            "prefix drift ON: a turn counter rides in front of the system prompt, so the "
            "prefix moves every turn and the KV cache misses every turn. Expect "
            "cached_tokens near zero and TTFT to grow with the prompt. This is the "
            "control for 'the system prompt must not change mid-task', not a run to "
            "compare against a still-prefix run's cost.")
        stuck = [f"{p}:{a}" for p, a in pairs
                 if a in getattr(base.get(p), "PROMPT_SENT_ONCE_ARMS", ())]
        if stuck:
            warnings.append(
                "prefix drift cannot apply to " + ", ".join(stuck) + ": these send the "
                "system prompt once and let the server (or an explicit cache) keep it, so "
                "there is no per-turn send for the counter to ride. Their marker sits on "
                "turn 1 and never moves -- read them as still-prefix arms.")
    if not cachebust.enabled():
        warnings.append(
            "cache-bust off: every arm sends the same system prompt, so an arm can be "
            "billed for -- and answered from -- the prefix cache a previous arm or a "
            "previous run left warm. Its cached_tokens and TTFT are not its own.")

    # Ask once, before the first arm. A capture that cannot start must not stop the
    # run -- the byte counts come from the socket and stand on their own -- but the
    # reason has to reach the operator, or a run with no pcaps looks like a run that
    # was never asked for one.
    if want_capture:
        ok, reason = pcap.available()
        if not ok:
            want_capture = False
            warnings.append(f"capture unavailable, running without it: {reason}")

    records: list[dict] = []
    pcaps: dict = {}
    wall_ms: dict = {}

    # Anything the probe or the model list left pooled would otherwise put its FIN in
    # the first arm's pcap.
    wire.reset_session()

    for i, (provider, arm) in enumerate(pairs):
        mod = base.get(provider)
        model = models.get(provider) or mod.DEFAULT_MODEL
        key = f"{provider}:{arm}"

        if i and pause_seconds:
            # Tick once a second rather than sleeping the gap in one go: a "pausing"
            # event followed by a minute of silence is indistinguishable from a hang.
            for remaining in range(int(pause_seconds), 0, -1):
                if on_progress:
                    on_progress({"provider": provider, "arm": arm, "phase": "pause",
                                 "remaining": remaining,
                                 "pause_total": int(pause_seconds)})
                time.sleep(1)

        ok, reason = mod.ready()
        if not ok:
            records.append({"provider": provider, "arm": arm, "phase": "steady",
                            "turn": 0, "error": f"not_ready: {reason}"})
            continue

        prompt = cachebust.apply(system, provider, arm)
        host = mod.api_host()

        if measure == "both" and want_capture:
            # One capture cannot hold `both`: the blocking and streamed passes interleave
            # on the same host and port, so a single pcap matches neither the bytes number
            # (which drops the streamed frames) nor the latency number. So the arm runs
            # twice -- the whole conversation in bytes, then the whole conversation in
            # latency -- each sweep under its own window, and the per-turn records are
            # merged back the way core.call.send merges the two passes of one turn: bytes
            # and body from the bytes sweep, marks from the latency sweep.
            sweeps = {}
            for kind in ("bytes", "latency"):
                window = _Window(provider, arm, host, timestamp or "0",
                                 want_capture, on_progress, kind=kind)
                try:
                    sweeps[kind] = mod.run_arm(arm, model, prompt, steps, kind,
                                               window.tick)
                finally:
                    window.close()
                if window.pcap is not None:
                    pcaps.setdefault(key, {})[kind] = window.pcap
                # The latency sweep is the one whose clock the marks come from, so its
                # wall is the arm's wall; the bytes sweep's is kept for its own pcap.
                wall_ms[key] = window.wall_ms
            arm_records = _merge_sweeps(sweeps["bytes"], sweeps["latency"])
        else:
            # Single kind: the capture, if any, is labelled with the measure it holds.
            kind = measure if measure in ("bytes", "latency") else "bytes"
            window = _Window(provider, arm, host, timestamp or "0",
                             want_capture, on_progress, kind=kind)
            try:
                arm_records = mod.run_arm(arm, model, prompt, steps, measure,
                                          window.tick)
            finally:
                window.close()
            wall_ms[key] = window.wall_ms
            if window.pcap is not None:
                pcaps.setdefault(key, {})[kind] = window.pcap

        records.extend(arm_records)

    return {
        "params": {
            "mode": "comparison",
            "measure": measure,
            "pairs": [f"{p}:{a}" for p, a in pairs],
            "providers": sorted({p for p, _ in pairs}),
            "models": {p: models.get(p) or base.get(p).DEFAULT_MODEL
                       for p, _ in pairs},
            "turns": len(steps),
            "capture": bool(want_capture),
            # The tags, not just the flag: a run whose arms came back suspiciously warm
            # can only be explained if the prefixes it actually sent are recoverable.
            "cache_bust": {"enabled": cachebust.enabled(),
                           "tags": cachebust.tags(pairs),
                           "prefix_drift": cachebust.drift_enabled()},
            "warnings": warnings,
        },
        "records": records,
        "pcaps": pcaps,
        "wall_ms": wall_ms,
    }


# The marks are the only thing a latency pass is entitled to report; everything else on
# the record -- bytes, tokens, body -- belongs to the bytes pass. Same split core.call
# makes between the two passes of one `both` turn, applied here to two whole sweeps.
_LATENCY_MARKS = ("req_sent_ms", "ttfb_ms", "ttft_ms", "ttlt_ms", "turn_end_ms",
                  "store_tail_ms")


def _merge_sweeps(bytes_recs: list[dict], latency_recs: list[dict]) -> list[dict]:
    """Fold a bytes sweep and a latency sweep of one arm into `both` records.

    A steady record keeps its bytes, tokens and body from the bytes sweep and takes its
    marks from the latency sweep's record for the same turn; `measure` becomes `both`.
    Prep records are kept from the bytes sweep alone -- the latency sweep re-ran prep too
    (a second cache build, a second conversation create), but that is duplicated setup, not
    a second measurement, and folding both would double an arm's reported prep.

    A turn the latency sweep failed or never reached leaves the bytes record's own marks
    in place, which are pinned rather than zero -- an honest "not measured", not "instant".
    """
    lat_by_turn = {r.get("turn"): r for r in latency_recs
                   if r.get("phase") == "steady"}
    out: list[dict] = []
    for r in bytes_recs:
        if r.get("phase") != "steady":
            out.append(r)
            continue
        merged = dict(r)
        merged["measure"] = "both"
        lat = lat_by_turn.get(r.get("turn"))
        if lat:
            for m in _LATENCY_MARKS:
                if m in lat:
                    merged[m] = lat[m]
        out.append(merged)
    return out


class _Window:
    """The measured stage of one arm: what the pcap covers and what wall_ms times.

    The arm tells us where that stage is, through the phase on its own progress events
    -- it opens on the first `steady` call and closes on a `teardown` one, or when the
    arm returns. Everything before the first steady call is prep (a cache build re-
    uploads the whole prefix and costs O(n^2) in bytes) and everything after a teardown
    event is cleanup (a cache DELETE). Neither is traffic the arm's turns produced, and
    a pcap holding either cannot be read as evidence of what a turn cost.

    An arm with no prep and no teardown announces `steady` first and never announces
    teardown, so it is captured whole -- which is right: all of it is traffic.
    """

    def __init__(self, provider: str, arm: str, host: str, timestamp: str,
                 want_capture: bool, on_progress, kind: str = ""):
        self.provider, self.arm, self.host = provider, arm, host
        self.timestamp = timestamp
        self.want_capture = want_capture
        self.on_progress = on_progress
        # Which pass this window captures -- "bytes" or "latency". It rides into the pcap
        # filename so a `both` run's two sweeps do not collide and a reader can tell the
        # blocking capture from the streamed one.
        self.kind = kind
        self.cap: pcap.Capture | None = None
        self.pcap: dict | None = None
        self.wall_ms = 0
        self._t0: float | None = None

    def tick(self, event: dict) -> None:
        phase = event.get("phase")
        if phase == "steady" and self._t0 is None:
            self._open()
        elif phase == "teardown":
            self.close()
        if self.on_progress:
            self.on_progress(event)

    def _open(self) -> None:
        # Prep left a pooled TLS connection open. Drop it, so the measured stage starts
        # from a handshake -- and when a pcap is about to open, wait out the peer's
        # FIN/ACK too, because that round trip would otherwise land inside it. The wait
        # is only worth its second when something is watching the wire.
        wire.reset_session()
        if self.want_capture:
            time.sleep(_SETTLE_SECONDS)
            self.cap = pcap.Capture(self.timestamp, self.provider, self.arm, self.host,
                                    kind=self.kind)
            self.cap.__enter__()
        self._t0 = time.monotonic()

    def close(self) -> None:
        """Idempotent: an arm may announce teardown and then also return."""
        if self._t0 is None:
            return
        self.wall_ms = int((time.monotonic() - self._t0) * 1000)
        self._t0 = None
        if self.cap is not None:
            # Close the socket before stopping tcpdump, so the FIN that ends the last
            # measured turn is inside the capture that measured it.
            wire.reset_session()
            time.sleep(_SETTLE_SECONDS)
            self.cap.__exit__(None, None, None)
            self.pcap = self.cap.result()
            self.cap = None
