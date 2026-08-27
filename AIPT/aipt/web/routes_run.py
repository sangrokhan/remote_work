"""POST /api/run -- pick a backend by name, drive one conversation through
it, and record the result (DESIGN.md 5, web UI FastAPI 통합 방침).

Synchronous, blocking-per-request, same posture as
``tcp_congestion/tcp_congestion/app.py``'s ``/api/run`` (DESIGN.md 3: "Flask의
동기 blocking 실행 ... FastAPI에서 run_in_threadpool로 감싸 이벤트 루프
블로킹 방지") -- the whole conversation runs in a worker thread via
``run_in_threadpool`` so it never blocks the event loop, but the HTTP
response only comes back once the run has finished.

``POST /api/run/stream`` is the streaming sibling (was the ``/api/run/stream``
TODO in ``aipt/web/app.py``'s module docstring): same ``RunRequest`` body,
but the response is ``text/event-stream`` and a ``{"type": "turn", ...}``
event is pushed the moment *each* turn finishes, rather than making the
caller wait for every turn before seeing anything. It is POST (not GET)
because the request body carries the full experiment config -- the browser
``EventSource`` API cannot send a POST body/JSON, so the frontend must read
this with ``fetch()`` + a manual ``ReadableStream`` reader instead of
``new EventSource(...)``.

Internally both routes share one generator, ``_run_conversation_stream()``:
it does the exact same connect/send/close work ``_run_conversation()``
always did, but ``yield``s a ``turn`` event after every
``backend.send_turn()`` instead of only appending to a list and returning
once at the very end. ``_run_conversation()` is now a thin wrapper that
drains that generator and returns just the final ``done`` event's result
dict, so ``/api/run`` and every existing caller/test keep working
unchanged.

The turn loop itself still runs in a worker thread (``backend.send_turn()``
is blocking socket I/O) -- for ``/api/run/stream`` that thread pushes each
yielded event onto a ``queue.Queue``, and the async route reads that queue
one item at a time via ``anyio.to_thread.run_sync(q.get)`` (blocks a
threadpool slot waiting for the next item, never the event loop) before
turning it into an SSE ``data: ...\\n\\n`` line. This is the standard
sync-generator-to-async-SSE bridge: a real async generator can't safely
wrap a blocking one without doing exactly this hand-off.

``local_llm`` is still a stub (``aipt.backends.local_llm.NotImplementedBackend``,
a parallel work stream owns the real implementation -- DESIGN.md 5 B4): a
request naming it is accepted at the route level (the registry knows the
name) and turned into a 501 the moment ``aipt.backends.get("local_llm")``'s
constructor raises ``NotImplementedError``, rather than surfacing as an
unhandled 500. ``/api/run/stream`` surfaces the same case as a single
``{"type": "error", ...}`` SSE event instead of an HTTP 501, since the
streaming response has already started (status 200) by the time a
mid-stream failure can happen.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import anyio
import anyio.to_thread
from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import aipt.backends as backends_registry
from aipt.backends.mock import conversation as mock_conversation
from aipt.backends.mock import replay as mock_replay
from aipt.backends.record import turn_record
from aipt.backends.public_ai import recorder as public_ai_recorder
from aipt.core import capture as capture_mod
from aipt.core import wire
from aipt.web import store as run_store

router = APIRouter()

log = logging.getLogger(__name__)

#: DESIGN.md 4.7.1 -- the *only* persistent-on-disk output this app writes.
#: Everything else (cwnd/pcap/mock/local_llm turns, CSV) stays in
#: aipt/web/store.py's in-memory cache; the user downloads bundle.zip.
PUBLIC_AI_RECORDS_DIR_ENV = "PUBLIC_AI_RECORDS_DIR"
DEFAULT_PUBLIC_AI_RECORDS_DIR = "data/public_ai_records"


def public_ai_records_dir() -> Path:
    return Path(os.environ.get(PUBLIC_AI_RECORDS_DIR_ENV, DEFAULT_PUBLIC_AI_RECORDS_DIR))


class RunRequest(BaseModel):
    """One experiment's parameters. Only ``backend`` and ``arm`` are
    required beyond the input mode -- everything else has a default
    matching a small, fast smoke run so the "just try it" path from the
    landing page needs no configuration.
    """

    backend: str = Field(..., description="public_ai | mock | local_llm")
    engine: str | None = Field(
        default=None,
        description=(
            "public_ai only: 'gemini' or 'openai'. Lets the UI's Gemini/"
            "ChatGPT cards (which both map to backend='public_ai') pin the "
            "engine explicitly instead of relying on arm-name inference."
        ),
    )
    arm: str = Field(..., description="Backend-specific arm name.")
    model: str = ""
    system: str = ""
    turns: list[str] = Field(
        default_factory=lambda: ["Hello, how are you?"],
        description="Question text for each turn, one call per entry.",
    )
    measure: str = "bytes"
    label: str = ""
    # mock-only knobs (ignored by other backends, kept flat rather than a
    # nested per-backend object -- the request body is small enough that a
    # single shared shape beats a discriminated union for this phase).
    # Defaults match the dummy byte-size sweep form's defaults (per operator
    # spec): a stress-shaped conversation (20 KB system prompt, 1 KB user
    # turns, 1 KB responses, 10 turns, 1 s inference delay) rather than the
    # earlier near-trivial smoke-test sizes, so a bare "just run it" request
    # (no form, direct /api/run call) exercises the same cumulative-context
    # growth the UI's own default now does.
    mock_response_bytes: int = 1000
    inference_delay_ms: int = 1000
    algorithm: str | None = None
    # Packet capture (tcpdump), TODO #5 (MIGRATION.md): default ON per
    # operator decision (2026-08-27) -- a run without a pcap is not
    # evidence of anything (aipt.core.capture's module docstring), so the
    # UI checkbox now ships pre-checked and this field defaults True to
    # match a bare /api/run call (no form) with the same posture. Actual
    # capture only happens if aipt.core.capture.available() is True
    # (tcpdump + NET_RAW present) -- unavailable environments fall back to
    # pcap=None exactly as before, never a hard failure.
    capture: bool = True

    # --- input mode ---------------------------------------------------
    # Two modes, per user decision: "dummy" and "record". The original
    # three mock arms (dummy/record/replay) collapsed to two -- a
    # hand-authored Q&A scenario record and a captured real run are the same
    # concept (a JSON of question/answer turns to replay), so there is
    # only one "record" source now: data/public_ai_records/<record_id>.json
    # (DESIGN.md 4.7.1). A hand-written scenario just gets dropped into
    # that same directory in the same schema instead of living in a
    # separate "records/" tree with its own loader.
    #
    # "dummy": mock-only. No record, no real question text -- the caller
    #   picks byte sizes and a turn count, and the size of the request
    #   text is computed per turn (system prompt once, own message every
    #   turn, PLUS everything already exchanged so far -- the same
    #   cumulative-context growth a real stateless multi-turn chat client
    #   produces) via aipt.backends.mock.conversation.build_turns().
    # "record": every backend supports this; it's the *only* mode
    #   public_ai/local_llm expose (they talk to a real model/engine, so
    #   there's no "dummy byte size" concept for them). Loads
    #   data/public_ai_records/<record_id>.json via
    #   aipt.backends.mock.replay.from_public_ai_record_doc. Real-model
    #   backends (public_ai/local_llm) only ever send the *question* half
    #   of each turn -- they're actually calling a live model/engine and
    #   need its real answer back, so a record's ``response_text`` is
    #   never sent to them, only used by the mock backend (which has no
    #   real model to ask).
    input_mode: str = Field(
        default="record",
        description="'dummy' (mock-only, byte-size knobs) or 'record' (replay data/public_ai_records/<record_id>.json).",
    )
    record_id: str = Field(
        default="",
        description="input_mode='record': exec_id under data/public_ai_records/ (no .json).",
    )
    # input_mode='dummy' (mock-only) knobs. Every byte-size input the user
    # sees maps 1:1 onto conversation.build_turns()'s own parameter names.
    # Defaults match the form's dummy-fields defaults (operator spec).
    system_prompt_bytes: int = 20000
    turn_user_msg_bytes: int = 1000
    num_turns: int = 10


def _load_record_scenario(record_id: str):
    """Loads ``data/public_ai_records/<record_id>.json`` and rebuilds it as
    a byte-pattern-only replay :class:`~aipt.backends.mock.records.ScenarioRecord`
    (question verbatim, answer -> same-length placeholder). Raises
    ``ValueError``/``OSError`` on a bad/missing record -- callers turn that
    into a normal run failure, never a raw 500.
    """
    if not record_id:
        raise ValueError("record_id is required")
    path = public_ai_records_dir() / f"{record_id}.json"
    doc = json.loads(path.read_text())
    return mock_replay.from_public_ai_record_doc(doc, name=record_id)


def _resolve_turns(req: RunRequest) -> tuple[list[str], str, str | None]:
    """Returns ``(question_texts, system_prompt, error)``. ``error`` is set
    (and the other two are empty/``""``) when the requested input mode
    can't be satisfied -- an unknown/missing record, or ``dummy``
    requested for a backend that doesn't support it -- so the caller can
    surface it as a normal run failure rather than a raw 500.

    ``dummy`` mode never touches a record at all: it synthesizes filler
    question text (and reuses conversation.build_turns()'s existing
    cumulative-context growth math) purely from byte-size knobs, and is
    only meaningful for the mock backend (public_ai/local_llm always talk
    to a real model/engine, so there is no filler-byte concept for them).

    ``record`` mode only ever surfaces the *question* half of each loaded
    turn here -- the answer half is a separate concern MockBackend alone
    consumes (see ``_build_backend``), since a real-model backend needs
    its own live answer, not a replayed one.
    """
    if req.input_mode == "dummy":
        if req.backend != "mock":
            return [], "", f"input_mode='dummy' is mock-only, not {req.backend!r}"
        try:
            specs = mock_conversation.build_turns(
                num_turns=req.num_turns,
                system_prompt_bytes=req.system_prompt_bytes,
                turn_user_msg_bytes=req.turn_user_msg_bytes,
                mock_response_bytes=req.mock_response_bytes,
                inference_delay_ms=req.inference_delay_ms,
                idle_duration_ms=0,
            )
        except ValueError as exc:
            return [], "", str(exc)
        # build_turns() returns each turn's *total* prompt_bytes (system
        # prompt folded into turn 0, cumulative history folded into every
        # turn after) -- that IS the filler text to send; MockBackend has
        # no separate "history" concept of its own; the request body sent
        # each turn simply has to already be that size.
        questions = ["x" * spec["prompt_bytes"] for spec in specs]
        return questions, "", None

    # "record": every backend replays the same recorded-run schema.
    if not req.record_id:
        return [], "", "input_mode='record' requires record_id"
    try:
        scenario_record = _load_record_scenario(req.record_id)
    except (ValueError, OSError) as exc:
        return [], "", f"failed to load record_id {req.record_id!r}: {exc}"
    questions = [t.question for t in scenario_record.turns]
    return questions, scenario_record.system_prompt, None


def _build_backend(name: str, engine: str | None = None, *, req: "RunRequest | None" = None):
    """``aipt.backends.get(name)``'s module, resolved to a constructable
    facade. Raises ``KeyError`` for an unknown name (propagates as 400).

    ``local_llm`` is owned by a parallel work stream (DESIGN.md 5 B4); at
    the time this route was written it could still be a
    ``NotImplementedBackend`` stub whose constructor raises
    ``NotImplementedError`` -- that must propagate as 501, never swallowed.
    If/when it lands as a real ``LocalLLMBackend``, this branch picks it up
    automatically (same attribute lookup pattern as public_ai/mock).

    ``mock`` + ``input_mode='record'``: the loaded ScenarioRecord (question AND
    answer) is bound to MockBackend so its own server actually serves the
    record's answer text for each turn -- unlike public_ai/local_llm, mock
    has no real model to ask, so replaying the recorded answer bytes IS
    the point.
    """
    module = backends_registry.get(name)
    if name == "public_ai":
        return module.PublicAIBackend(engine=engine) if engine else module.PublicAIBackend()
    if name == "mock":
        scenario_record = None
        if req is not None and req.input_mode == "record" and req.record_id:
            try:
                scenario_record = _load_record_scenario(req.record_id)
            except (ValueError, OSError):
                scenario_record = None  # _resolve_turns() already reports this as a run failure
        return module.MockBackend(record=scenario_record)
    if name == "local_llm":
        facade = getattr(module, "LocalLLMBackend", None) or getattr(
            module, "NotImplementedBackend", None
        )
        if facade is None:
            raise NotImplementedError(f"local_llm backend module has no usable class")
        return facade()
    raise KeyError(f"unhandled backend: {name!r}")


def _split_api_host(api_host: str, default_port: int = 443) -> tuple[str, int]:
    """``backend.api_host()`` returns different shapes per backend --
    ``host:port`` (mock's own bind address), a bare hostname (public_ai,
    always TLS/443), or a full ``scheme://host:port`` URL (local_llm's
    engine URL). ``aipt.core.capture.Capture`` needs a plain
    ``(host, port)`` pair to build its tcpdump filter, so this normalizes
    all three into that shape. Never raises -- a value this can't parse
    falls back to ``(api_host, default_port)`` so capture still gets
    *something* to filter on rather than crashing the run.
    """
    from urllib.parse import urlparse

    value = api_host or ""
    if "://" in value:
        parsed = urlparse(value)
        if parsed.hostname:
            return parsed.hostname, parsed.port or (
                443 if parsed.scheme == "https" else default_port
            )
        return value, default_port
    if ":" in value:
        host, _, port_s = value.rpartition(":")
        try:
            return host, int(port_s)
        except ValueError:
            return value, default_port
    return value, default_port


def _run_conversation_stream(req: RunRequest) -> Iterator[dict]:
    """Generator core shared by ``/api/run`` and ``/api/run/stream``: does
    the exact connect/send-every-turn/close work ``_run_conversation()``
    used to do inline, but ``yield``s an event after each meaningful step
    instead of only building up a ``records`` list silently.

    Events (each a plain ``dict``, JSON-serializable as-is):
      - ``{"type": "start", "exec_id", "backend", "arm", "label", "total_turns"}``
        once, right before the first ``send_turn()`` call.
      - ``{"type": "turn", "turn": i, "total_turns": N, "record": {...}}``
        after each turn completes (record has the same shape
        ``turn_record()`` always produced).
      - ``{"type": "done", "result": {...}}`` exactly once, always last --
        ``result`` is the same run-document dict ``_run_conversation()``
        used to return directly (``ok``/``error``/``turns``/``exec_id``/...).

    A ready()/input-resolution failure short-circuits straight to a single
    ``done`` event (no ``start``/``turn``), matching the early-return
    shapes ``_run_conversation()`` always used for those cases.
    """
    backend = _build_backend(req.backend, engine=req.engine, req=req)
    ok, reason = backend.ready()
    if not ok:
        yield {
            "type": "done",
            "result": {
                "ok": False,
                "error": reason,
                "backend": req.backend,
                "arm": req.arm,
                "turns": [],
            },
        }
        return

    questions, resolved_system, resolve_error = _resolve_turns(req)
    if resolve_error:
        yield {
            "type": "done",
            "result": {
                "ok": False,
                "error": resolve_error,
                "backend": req.backend,
                "arm": req.arm,
                "turns": [],
            },
        }
        return
    system = req.system or resolved_system

    label = req.label or f"{req.backend}:{req.arm}"
    timestamp = datetime.now(timezone.utc).isoformat()
    if req.backend == "mock":
        backend.mock_response_bytes = req.mock_response_bytes
        backend.inference_delay_ms = req.inference_delay_ms
        backend.algorithm = req.algorithm
        backend.label = label
    else:
        # public_ai/local_llm never open their own raw socket (mock's
        # _connect_with_algorithm does) -- their connections come from
        # aipt.core.wire's shared, pooled session, so pinning the algorithm
        # goes through wire.set_congestion_algorithm() instead of a
        # per-backend attribute. Cleared (None) rather than left over from
        # a previous run's setting whenever this run does not ask for one --
        # see that function's docstring on why a stale value would otherwise
        # silently leak into the next run. reset_session() is required here
        # (unlike LocalLLMBackend.connect(), which already calls it, and
        # unlike GeminiBackend/OpenAIBackend.connect(), which do not): the
        # algorithm sockopt is only applied on a fresh socket in
        # _CountingConnection._new_conn(), so a request reusing an
        # already-pooled connection from an earlier run would silently keep
        # that run's algorithm (or the kernel default) regardless of what
        # this run just asked for.
        wire.set_congestion_algorithm(req.algorithm or None)
        wire.reset_session()

    # exec_id is generated here (rather than left to run_store.save_run) so
    # the public_ai record file on disk and the in-memory run doc share the
    # same id -- a caller can go from one to the other.
    exec_id = run_store.new_exec_id()

    writer = None
    if req.backend == "public_ai":
        engine = req.engine or ""
        if not engine:
            try:
                engine = backends_registry.get("public_ai").engine_for_arm(req.arm)
            except ValueError:
                engine = ""
        writer = public_ai_recorder.RecordWriter(system=system, steps=list(questions))
        backend = public_ai_recorder.recording_backend(backend, writer, engine=engine)

    total_turns = len(questions)
    yield {
        "type": "start",
        "exec_id": exec_id,
        "backend": req.backend,
        "arm": req.arm,
        "label": label,
        "total_turns": total_turns,
    }

    records: list[dict] = []
    error = "" 
    cap = None
    if req.capture:
        cap_ok, _cap_reason = capture_mod.available()
        if cap_ok:
            # api_host() is only meaningful once connect() has picked a
            # concrete target (mock starts its own server on a random
            # port at connect() time), so the capture window opens right
            # after connect() below, not before -- unlike the label/
            # timestamp naming above, which does not depend on the
            # connection existing yet. Label sanitized to Capture's
            # filename alphabet ([a-z0-9_-]); the display `label` above
            # may contain ':' (backend:arm) which Capture's _SAFE_LABEL
            # rejects.
            cap_label = re.sub(r"[^a-z0-9_-]", "_", label.lower()) or "run"
        else:
            cap_label = ""
    else:
        cap_label = ""
    t0 = time.monotonic()
    try:
        backend.connect(req.arm, req.model, system)
        if cap_label:
            host, port = _split_api_host(backend.api_host())
            cap = capture_mod.Capture(
                timestamp=timestamp, label=cap_label, host=host, port=port)
            cap.__enter__()
        for i, question in enumerate(questions):
            exchange = backend.send_turn(i, question, req.measure)
            record = turn_record(
                backend=req.backend,
                arm=req.arm,
                phase="steady",
                turn=i,
                question=question,
                measure=req.measure,
                exchange=exchange,
                usage={},
                transport=getattr(backend, "transport", "http1"),
            )
            records.append(record)
            yield {"type": "turn", "turn": i, "total_turns": total_turns, "record": record}
    except Exception as exc:  # a run that fails mid-conversation still
        # reports what it got, rather than losing the turns already sent.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if cap is not None:
            try:
                cap.__exit__(None, None, None)
            except Exception:
                pass
        try:
            backend.close()
        except Exception:
            pass
    elapsed_s = time.monotonic() - t0

    cwnd_result = None
    if hasattr(backend, "cwnd_result"):
        try:
            cwnd_result = backend.cwnd_result()
        except Exception:
            cwnd_result = None

    # Congestion-algorithm outcome, from whichever path this backend pinned
    # it through -- MockBackend's own attributes (set on the raw socket by
    # _connect_with_algorithm) for mock, aipt.core.wire's module-global
    # state (set on the pooled session's connections) for public_ai/
    # local_llm. Surfaced uniformly here so the run document/UI does not
    # need to know which path a given backend used.
    if req.backend == "mock":
        algorithm_result = {
            "requested": getattr(backend, "algorithm_requested", req.algorithm or ""),
            "actual": getattr(backend, "algorithm_actual", ""),
            "error": getattr(backend, "algorithm_error", ""),
        }
    else:
        algorithm_result = wire.congestion_algorithm_result()

    result = {
        "ok": not error,
        "error": error,
        "backend": req.backend,
        "arm": req.arm,
        "label": label,
        "model": req.model,
        "measure": req.measure,
        "mock": req.backend in ("mock",),
        "timestamp": timestamp,
        "elapsed_s": round(elapsed_s, 3),
        "turns": records,
        "monitors": [cwnd_result] if cwnd_result else [],
        "pcap": cap.result() if cap is not None else None,
        "algorithm": algorithm_result,
        "exec_id": exec_id,
    }

    # DESIGN.md 4.7.1: persist public_ai records regardless of run success --
    # a failed-partway-through run still spent real API-call money on the
    # turns it did make, and those must not be silently dropped. A disk
    # write failure here (permissions, disk full, ...) must never crash the
    # experiment itself -- same "honest failure reporting, never a hard
    # crash" posture as recorder.py's masking: log it, surface it in the
    # response, keep going.
    if writer is not None:
        try:
            path = public_ai_records_dir() / f"{exec_id}.json"
            writer.write(path)
            result["record_saved"] = True
            result["record_path"] = str(path)
        except Exception as exc:
            log.exception("failed to persist public_ai record for exec_id=%s", exec_id)
            result["record_saved"] = False
            result["record_error"] = f"{type(exc).__name__}: {exc}"

    yield {"type": "done", "result": result}


def _run_conversation(req: RunRequest) -> dict:
    """Blocking: drains :func:`_run_conversation_stream` and returns just
    its final ``done`` event's ``result`` dict -- the same run-document
    shape this function always returned directly, before the per-turn
    ``yield`` events existed. Runs on a threadpool worker (see
    ``api_run`` below), never on the event loop thread.
    """
    for event in _run_conversation_stream(req):
        if event["type"] == "done":
            return event["result"]
    raise RuntimeError("_run_conversation_stream() ended without a 'done' event")


@router.post("/api/run")
async def api_run(req: RunRequest):
    try:
        backends_registry.get(req.backend)
    except KeyError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    try:
        result = await run_in_threadpool(_run_conversation, req)
    except NotImplementedError as exc:
        return JSONResponse(
            {"ok": False, "error": f"backend {req.backend!r} not implemented: {exc}"},
            status_code=501,
        )
    except Exception as exc:
        return JSONResponse(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500
        )

    saved = run_store.save_run(result)
    return JSONResponse({"ok": result["ok"], "run": saved})


#: Sentinel put on the bridge queue once the worker thread's generator is
#: fully drained (normal completion or an uncaught exception) -- lets the
#: async consumer loop tell "no more items, stop reading" apart from "no
#: item yet, keep waiting" without needing a separate flag/event.
_STREAM_DONE = object()


def _stream_log_path(exec_id: str) -> Path:
    """``<run_store_dir()>/<exec_id>.stream.jsonl`` -- one line per SSE
    event for that run, in the same directory ``run_store.save_run()``
    already writes ``<exec_id>.json`` to (reuses ``RUN_STORE_DIR``, no new
    env var). Kept as a sibling ``.stream.jsonl`` file rather than folded
    into the run doc itself so it can be appended to incrementally (one
    ``open(..., "a")`` per event) while the run doc is only ever written
    once, whole, at the end.

    A ``start`` event's ``exec_id`` is only known once the generator gets
    past backend/input validation -- events before that point (an early
    ``done`` with ``ok: False``, or a pre-backend ``error``) have no
    ``exec_id`` yet and are logged (see ``_log_stream_event``) but not
    written to a per-run file, since there is no run to name the file
    after.
    """
    return run_store.run_store_dir() / f"{exec_id}.stream.jsonl"


def _log_stream_event(req: RunRequest, event: dict, exec_id: str | None) -> None:
    """Structured log line for every SSE event this route ever produces,
    plus (once ``exec_id`` is known) an appended line in that run's
    ``<exec_id>.stream.jsonl`` on disk -- so a run's turn-by-turn stream
    can be inspected after the fact even though nothing in the client-
    facing contract requires the client itself to have been listening
    (the whole point of "log it server-side instead of requiring a
    frontend consumer").

    All log lines go through ``log.debug`` -- per-turn events are high
    volume (one per turn per run) and are not actionable at INFO/WARNING
    severity by themselves (a genuine failure still fully surfaces to the
    caller as an ``error``/``ok: False`` SSE event and in the persisted
    run doc); DEBUG keeps this quiet by default and opt-in via the
    logger's level, same as any other high-frequency diagnostic trace.
    The on-disk ``.stream.jsonl`` append is unaffected by logger level --
    it is the durable, always-on record; the ``log.debug`` calls are only
    the live/ad-hoc visibility path.

    Never raises: a logging/disk-write failure must not take down the run
    itself, same "honest failure reporting, never a hard crash" posture as
    ``run_store``'s own disk I/O (log the write failure and move on).
    """
    kind = event.get("type", "unknown")
    if kind == "turn":
        log.debug(
            "run/stream backend=%s arm=%s exec_id=%s turn=%s/%s",
            req.backend, req.arm, exec_id, event.get("turn"), event.get("total_turns"),
        )
    elif kind == "start":
        log.debug(
            "run/stream backend=%s arm=%s exec_id=%s start total_turns=%s",
            req.backend, req.arm, exec_id, event.get("total_turns"),
        )
    elif kind == "done":
        result = event.get("result") or {}
        log.debug(
            "run/stream backend=%s arm=%s exec_id=%s done ok=%s turns=%s elapsed_s=%s",
            req.backend, req.arm, exec_id, result.get("ok"),
            len(result.get("turns") or []), result.get("elapsed_s"),
        )
    else:
        log.debug(
            "run/stream backend=%s arm=%s exec_id=%s error=%s",
            req.backend, req.arm, exec_id, event.get("error"),
        )

    if not exec_id:
        return  # nothing to name the per-run log file after yet
    try:
        path = _stream_log_path(exec_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"logged_at": time.time(), **event})
        with path.open("a") as f:
            f.write(line + "\n")
    except OSError as exc:  # pragma: no cover - defensive, disk/perm issues
        log.warning("failed to append stream log for exec_id=%s: %s", exec_id, exc)


def _drive_stream_to_queue(req: RunRequest, q: "queue.Queue[object]") -> None:
    """Runs on a threadpool worker: drains ``_run_conversation_stream()``
    (which itself calls blocking ``backend.send_turn()`` etc.), pushing
    each event onto *q* as it's produced and logging every event via
    :func:`_log_stream_event` (structured log line + per-run
    ``<exec_id>.stream.jsonl`` append) regardless of whether the HTTP
    client is even still connected to read the SSE response -- the
    logging path does not depend on the queue/consumer side at all.

    Any exception escaping the generator (e.g. ``NotImplementedError`` for
    local_llm) is turned into one final ``{"type": "error", ...}`` event
    rather than being raised on a thread nothing awaits -- the streaming
    response already sent a 200 status line by the time this thread
    starts, so there is no HTTP status code left to change; an SSE error
    event (logged like any other) is the only way left to tell anything
    downstream something went wrong. ``_STREAM_DONE`` is always pushed
    last, success or failure, so the consumer loop always terminates.
    """
    exec_id: str | None = None
    try:
        for event in _run_conversation_stream(req):
            if event["type"] == "start":
                exec_id = event.get("exec_id")
            elif event["type"] == "done":
                exec_id = event["result"].get("exec_id") or exec_id
            _log_stream_event(req, event, exec_id)
            q.put(event)
            if event["type"] == "done":
                run_store.save_run(event["result"])
    except Exception as exc:
        error_event = {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
        _log_stream_event(req, error_event, exec_id)
        q.put(error_event)
    finally:
        q.put(_STREAM_DONE)


@router.post("/api/run/stream")
async def api_run_stream(req: RunRequest):
    """Streaming sibling of ``POST /api/run`` (the former ``/api/run/stream``
    TODO): same ``RunRequest`` body, ``text/event-stream`` response with one
    SSE ``data: <json>\\n\\n`` line per event (``start``/``turn``/``done``, or
    a single ``error`` event if something raised before/instead of a normal
    ``done``). The completed run is saved to ``run_store`` exactly once, at
    the moment the ``done`` event is produced -- same persistence ``POST
    /api/run`` always did, just moved earlier (mid-stream, not after
    draining) since there is no final synchronous return value here to hang
    the save off of.

    Unlike ``POST /api/run``, an unknown backend name or a ``local_llm``
    ``NotImplementedError`` is reported as an in-stream ``error`` event
    (still HTTP 200) rather than a 400/501 status code, because SSE headers
    (and therefore the status line) go out before any backend work happens
    -- there is no later point at which this route could still change the
    status code.
    """
    try:
        backends_registry.get(req.backend)
    except KeyError as exc:
        error_message = str(exc)
        error_event = {"type": "error", "error": error_message}
        _log_stream_event(req, error_event, exec_id=None)

        async def _bad_backend():
            yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(_bad_backend(), media_type="text/event-stream")

    q: "queue.Queue[object]" = queue.Queue()

    async def event_gen():
        # Kick off the blocking generator on its own thread immediately --
        # do not await it (that would block this coroutine on the whole
        # run finishing, exactly what streaming exists to avoid).
        async with anyio.create_task_group() as tg:
            tg.start_soon(anyio.to_thread.run_sync, _drive_stream_to_queue, req, q)
            while True:
                # anyio.to_thread.run_sync(q.get) parks a threadpool worker
                # on the blocking Queue.get(), not the event loop -- so
                # other requests keep being served while this one waits
                # for the next event.
                item = await anyio.to_thread.run_sync(q.get)
                if item is _STREAM_DONE:
                    break
                yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
