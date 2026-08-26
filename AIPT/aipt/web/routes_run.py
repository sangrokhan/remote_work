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

import time
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import aipt.backends as backends_registry
from aipt.backends.record import turn_record
from aipt.web import store as run_store

router = APIRouter()


class RunRequest(BaseModel):
    """One experiment's parameters. Only ``backend`` and ``arm`` are
    required beyond the fixture -- everything else has a default matching
    a small, fast smoke run so the "just try it" path from the landing page
    needs no configuration.
    """

    backend: str = Field(..., description="public_ai | mock | local_llm")
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
    """
    backend = _build_backend(req.backend)
    ok, reason = backend.ready()
    if not ok:
        return {
            "ok": False,
            "error": reason,
            "backend": req.backend,
            "arm": req.arm,
            "turns": [],
        }

    label = req.label or f"{req.backend}:{req.arm}"
    timestamp = datetime.now(timezone.utc).isoformat()
    connect_kwargs = {}
    if req.backend == "mock":
        backend.mock_response_bytes = req.mock_response_bytes
        backend.inference_delay_ms = req.inference_delay_ms
        backend.algorithm = req.algorithm
        backend.label = label

    records: list[dict] = []
    error = ""
    t0 = time.monotonic()
    try:
        backend.connect(req.arm, req.model, req.system)
        for i, question in enumerate(req.turns):
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

    return {
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
    }


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
