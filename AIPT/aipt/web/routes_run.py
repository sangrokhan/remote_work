"""POST /api/run -- pick a backend by name, drive one conversation through
it, and record the result (DESIGN.md 5, web UI FastAPI 통합 방침).

Synchronous, blocking-per-request, same posture as
``tcp_congestion/tcp_congestion/app.py``'s ``/api/run`` (DESIGN.md 3: "Flask의
동기 blocking 실행 ... FastAPI에서 run_in_threadpool로 감싸 이벤트 루프
블로킹 방지") -- the whole conversation runs in a worker thread via
``run_in_threadpool`` so it never blocks the event loop, but the HTTP
response only comes back once the run has finished. A streaming
(``/api/run/stream``) variant is out of scope for this phase (see the
module docstring in ``aipt/web/app.py``).

``local_llm`` is still a stub (``aipt.backends.local_llm.NotImplementedBackend``,
a parallel work stream owns the real implementation -- DESIGN.md 5 B4): a
request naming it is accepted at the route level (the registry knows the
name) and turned into a 501 the moment ``aipt.backends.get("local_llm")``'s
constructor raises ``NotImplementedError``, rather than surfacing as an
unhandled 500.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import aipt.backends as backends_registry
from aipt.backends.mock import conversation as mock_conversation
from aipt.backends.mock import fixtures as mock_fixtures
from aipt.backends.record import turn_record
from aipt.backends.public_ai import recorder as public_ai_recorder
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
    required beyond the fixture -- everything else has a default matching
    a small, fast smoke run so the "just try it" path from the landing page
    needs no configuration.
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
    mock_response_bytes: int = 400
    inference_delay_ms: int = 0
    algorithm: str | None = None

    # --- input mode ---------------------------------------------------
    # "fixed": read a Q&A JSON fixture (aipt.backends.mock.fixtures schema
    #   -- system_prompt + turns[{question,answer}]) and drive the
    #   conversation from its questions. Every backend supports this; it
    #   is the *only* mode public_ai/local_llm expose (they talk to a real
    #   model/engine, so there is no "dummy byte size" concept for them --
    #   only mock can synthesize filler text that means nothing).
    # "dummy": mock-only. No fixture, no real question text -- the caller
    #   picks byte sizes and a turn count, and the size of the request
    #   text is computed per turn (system prompt once, own message every
    #   turn, PLUS everything already exchanged so far -- the same
    #   cumulative-context growth a real stateless multi-turn chat client
    #   produces) via aipt.backends.mock.conversation.build_turns().
    input_mode: str = Field(
        default="fixed",
        description="'fixed' (load a Q&A JSON fixture) or 'dummy' (mock-only, byte-size knobs).",
    )
    fixture_name: str = Field(
        default="",
        description="input_mode='fixed': name under aipt/backends/mock/fixtures.FIXTURE_DIR (no .json).",
    )
    # input_mode='dummy' (mock-only) knobs. Every byte-size input the user
    # sees maps 1:1 onto conversation.build_turns()'s own parameter names.
    system_prompt_bytes: int = 0
    turn_user_msg_bytes: int = 0
    num_turns: int = 3


def _resolve_turns(req: RunRequest) -> tuple[list[str], str, str | None]:
    """Returns ``(question_texts, system_prompt, error)``. ``error`` is set
    (and the other two are empty/``""``) when the requested input mode
    can't be satisfied -- an unknown fixture name, or ``dummy`` requested
    for a backend that doesn't support it -- so the caller can surface it
    as a normal run failure rather than a raw 500.

    ``dummy`` mode never touches a fixture file at all: it synthesizes
    filler question text (and reuses conversation.build_turns()'s existing
    cumulative-context growth math) purely from byte-size knobs, and is
    only meaningful for the mock backend (public_ai/local_llm always talk
    to a real model/engine, so there is no filler-byte concept for them).
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

    # "fixed": every backend reads the same Q&A fixture schema.
    if not req.fixture_name:
        return [], "", "input_mode='fixed' requires fixture_name"
    try:
        fixture = mock_fixtures.load(req.fixture_name)
    except (KeyError, ValueError, OSError) as exc:
        return [], "", f"failed to load fixture {req.fixture_name!r}: {exc}"
    questions = [t.question for t in fixture.turns]
    return questions, fixture.system_prompt, None


def _build_backend(name: str, engine: str | None = None):
    """``aipt.backends.get(name)``'s module, resolved to a constructable
    facade. Raises ``KeyError`` for an unknown name (propagates as 400).

    ``local_llm`` is owned by a parallel work stream (DESIGN.md 5 B4); at
    the time this route was written it could still be a
    ``NotImplementedBackend`` stub whose constructor raises
    ``NotImplementedError`` -- that must propagate as 501, never swallowed.
    If/when it lands as a real ``LocalLLMBackend``, this branch picks it up
    automatically (same attribute lookup pattern as public_ai/mock).
    """
    module = backends_registry.get(name)
    if name == "public_ai":
        return module.PublicAIBackend(engine=engine) if engine else module.PublicAIBackend()
    if name == "mock":
        return module.MockBackend()
    if name == "local_llm":
        facade = getattr(module, "LocalLLMBackend", None) or getattr(
            module, "NotImplementedBackend", None
        )
        if facade is None:
            raise NotImplementedError(f"local_llm backend module has no usable class")
        return facade()
    raise KeyError(f"unhandled backend: {name!r}")


def _run_conversation(req: RunRequest) -> dict:
    """Blocking: connect, send every turn, close, and shape the result into
    a run document the export layer and the runs endpoints can consume.
    Runs on a threadpool worker (see ``run_experiment`` below), never on
    the event loop thread.

    DESIGN.md 4.7.1: when ``req.backend == "public_ai"``, every request/
    response this run makes is additionally captured via
    ``aipt.backends.public_ai.recorder.recording_backend`` and written to
    ``data/public_ai_records/<exec_id>.json`` -- the *only* persistent
    on-disk artifact this app produces. mock/local_llm runs never touch
    this path; their turns live only in the in-memory store (Phase 1).
    """
    backend = _build_backend(req.backend, engine=req.engine)
    ok, reason = backend.ready()
    if not ok:
        return {
            "ok": False,
            "error": reason,
            "backend": req.backend,
            "arm": req.arm,
            "turns": [],
        }

    questions, resolved_system, resolve_error = _resolve_turns(req)
    if resolve_error:
        return {
            "ok": False,
            "error": resolve_error,
            "backend": req.backend,
            "arm": req.arm,
            "turns": [],
        }
    system = req.system or resolved_system

    label = req.label or f"{req.backend}:{req.arm}"
    timestamp = datetime.now(timezone.utc).isoformat()
    connect_kwargs = {}
    if req.backend == "mock":
        backend.mock_response_bytes = req.mock_response_bytes
        backend.inference_delay_ms = req.inference_delay_ms
        backend.algorithm = req.algorithm
        backend.label = label

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
        writer = public_ai_recorder.FixtureWriter(system=system, steps=list(questions))
        backend = public_ai_recorder.recording_backend(backend, writer, engine=engine)

    records: list[dict] = []
    error = ""
    t0 = time.monotonic()
    try:
        backend.connect(req.arm, req.model, system)
        for i, question in enumerate(questions):
            exchange = backend.send_turn(i, question, req.measure)
            records.append(
                turn_record(
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
            )
    except Exception as exc:  # a run that fails mid-conversation still
        # reports what it got, rather than losing the turns already sent.
        error = f"{type(exc).__name__}: {exc}"
    finally:
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
        "pcap": None,  # TODO: wire aipt.core.capture once a route asks for it
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

    return result


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
