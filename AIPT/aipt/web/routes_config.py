"""GET / (landing) and GET /api/config -- what the operator needs before
picking a backend and spending anything.

DESIGN.md 4.5 replaced the "external_api lab / synthetic_mock lab" landing
page with a **backend-selection** landing page: three cards (public_ai /
mock / local_llm), because the client now always talks to exactly one
backend chosen at run time, not one lab mounted at its own URL prefix.
`aipt.backends.names()` is the single source of truth for which backends
exist; this module never hardcodes the three names beyond display labels.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import aipt.backends as backends_registry
from aipt.backends.mock import records as mock_records
from aipt.backends.public_ai import gemini as _gemini
from aipt.backends.public_ai import openai as _openai
from aipt.core import capture as capture_mod
from aipt.core import congestion as congestion_mod
from aipt.core import quic_congestion as quic_congestion_mod
from aipt.core import cwnd as cwndmon
from aipt.web.routes_run import public_ai_records_dir

router = APIRouter()

# Congestion-control algorithms surfaced in the experiment form. Read live
# from the running kernel (aipt.core.congestion.available_algorithms(), which
# parses /proc/sys/net/ipv4/tcp_available_congestion_control) rather than a
# fixed guess -- a hardcoded list would offer names this box may not have
# loaded (a CI box or a slim container image commonly ships only cubic/
# reno), and the operator would only discover that after a run failed with
# an opaque algorithm_error. Now the dropdown only ever lists what the
# kernel says it can actually run.
def _congestion_algorithms() -> tuple[list[str], str]:
    return congestion_mod.available_algorithms()


# QUIC congestion-control algorithms, same "read what's actually
# available, never fabricate a fallback list" posture as
# _congestion_algorithms() above -- but for aioquic's userspace registry
# instead of the kernel's /proc/sys/net/ipv4/tcp_available_congestion_control
# (DESIGN.md section 7's QUIC idle-probe spike, now surfaced as the web
# UI's Transport picker). Includes the project's own "idle_probe"
# algorithm alongside aioquic's stock "reno"/"cubic" -- see
# aipt.core.quic_congestion's module docstring on why importing it has
# that registration side effect.
def _quic_congestion_algorithms() -> tuple[list[str], str]:
    return quic_congestion_mod.available_algorithms()


#: Backend display metadata for the landing page. Keyed by the same names
#: aipt.backends.names() returns, so a card is never shown for a backend
#: the registry does not know, and never silently missing one it does.
_BACKEND_DISPLAY = {
    "public_ai": {
        "label": "Public AI",
        "description": "Gemini / ChatGPT over the real network (billed).",
    },
    "mock": {
        "label": "Mock",
        "description": "Fixed or replayed JSON I/O, no network cost.",
    },
    "local_llm": {
        "label": "Local LLM",
        "description": "Standard serving engine (llama.cpp/vLLM) behind an in-repo gateway.",
    },
}

# The attribute names that mark a backend module as having a real
# Backend-protocol implementation, checked by presence rather than a
# hardcoded name list -- so "implemented" flips automatically the moment a
# parallel work stream (e.g. DESIGN.md 5 B4 for local_llm) lands its real
# class, with no change needed here.
_FACADE_ATTRS = ("PublicAIBackend", "MockBackend", "LocalLLMBackend")


def _facade_class(module):
    """The Backend-protocol class this backend module exposes, or None for
    an import-path-only stub (``NotImplementedBackend``)."""
    for attr in _FACADE_ATTRS:
        facade = getattr(module, attr, None)
        if facade is not None:
            return facade
    return None


def _backend_ready(name: str) -> tuple[bool, str]:
    """(ok, reason) for one backend, without ever letting an unimplemented
    or misconfigured backend's constructor crash the config/landing page."""
    try:
        module = backends_registry.get(name)
    except KeyError as exc:
        return False, str(exc)
    facade = _facade_class(module)
    if facade is None:
        return False, "not implemented yet"
    try:
        return facade().ready()
    except NotImplementedError as exc:
        return False, f"not implemented yet: {exc}"
    except Exception as exc:  # never let a landing page 500 on a bad backend
        return False, f"error checking readiness: {exc}"


def _backend_arms(name: str) -> list[str]:
    """The arm names a backend accepts, for the experiment form's arm
    dropdown. Empty for an unimplemented backend rather than raising -- the
    form still renders, the fieldset just says so."""
    try:
        module = backends_registry.get(name)
    except KeyError:
        return []
    facade = _facade_class(module)
    if facade is None:
        return []
    try:
        return list(getattr(facade, "ARMS", ()))
    except Exception:
        return []


#: DESIGN.md 4.5 groups Gemini/ChatGPT under one registry slot
#: ("public_ai") because they share one Backend-protocol facade
#: (``PublicAIBackend(engine=...)``), but a user picking a backend from
#: the landing page thinks in terms of "which vendor", not "which
#: registry slot" -- Gemini and ChatGPT have entirely disjoint arm sets
#: (6 vs 4) and mixing them into one dropdown makes the arm list look
#: like one incoherent pile of 10 options. ``ui_backends()`` below is
#: the UI-facing view: it splits public_ai into two cards (one per
#: engine) so each gets its own arm dropdown, while ``backends_view()``
#: (registry-name-keyed) stays intact for anything that still needs the
#: 3-way public_ai/mock/local_llm split (POST /api/run's ``backend``
#: field, for one -- it still only knows 3 names; the UI card carries
#: the extra ``engine`` value that resolves back to one of them).
_PUBLIC_AI_ENGINE_DISPLAY = {
    "gemini": {
        "label": "Gemini",
        "description": "Google Gemini API -- stateless/stateful-pointer/explicit-cache arms.",
    },
    "openai": {
        "label": "ChatGPT",
        "description": "OpenAI Chat/Responses API -- stateless/stateful-pointer arms.",
    },
}


def _public_ai_engine_arms(engine: str) -> list[str]:
    module = {"gemini": _gemini, "openai": _openai}[engine]
    return list(getattr(module, "ARMS", ()))


#: Every arm belongs to exactly one underlying HTTP API, and that API --
#: not the arm's name -- is what changes the shape of the request on the
#: wire (Gemini's `generateContent` vs `interactions`; OpenAI's
#: `chat/completions` vs `responses`). The operator picks an API Type
#: first (a real, billable endpoint choice), then a Context Handle (how
#: the conversation's history is carried: resent by the client every
#: turn, kept server-side inline with the prompt, or dropped from the
#: prompt and referenced statelessly) -- both are UI-only groupings over
#: the same arm names every backend already validates, so nothing here
#: changes what actually goes over the wire, only how the form presents
#: the choice. Keyed by the same key ``ui_backends()`` cards use: the
#: engine name (\"gemini\"/\"openai\") for the two Public AI cards, and the
#: registry name (\"mock\"/\"local_llm\") for the other two.
_API_TYPES = {
    "gemini": (
        {
            "key": "generate_content",
            "label": "generateContent API",
            "context_handles": (
                ("default", "Default", "stateless"),
                ("no_context", "No Context", "nocontext"),
                ("force_caching", "Force Caching", "cached"),
            ),
        },
        {
            "key": "interactions",
            "label": "Interaction API",
            "context_handles": (
                ("default", "Default", "interaction"),
                ("inline", "Inline", "interaction_inline"),
                ("stateless", "Stateless", "interaction_stateless"),
            ),
        },
    ),
    "openai": (
        {
            "key": "chat_completions",
            "label": "Chat Completion API",
            "context_handles": (
                ("default", "Default", "chat_stateless"),
            ),
        },
        {
            "key": "responses",
            "label": "Responses API",
            "context_handles": (
                ("default", "Default", "responses"),
                ("inline", "Inline", "responses_inline"),
                ("stateless", "Stateless", "responses_stateless"),
            ),
        },
    ),
    "local_llm": (
        {
            "key": "chat_completions",
            "label": "Chat Completion API",
            "context_handles": (
                ("default", "Default", "chat"),
            ),
        },
    ),
    "mock": (
        {
            "key": "dummy_api",
            "label": "Dummy API",
            "context_handles": (
                ("default", "Default", "dummy"),
            ),
        },
    ),
}


def _api_types_for(card_key: str, known_arms: list[str]) -> list[dict]:
    """API Type groups for *card_key*, each carrying only the context
    handles whose underlying arm is in *known_arms*.

    Every card key currently in ``_API_TYPES`` was curated deliberately per
    the operator's exact spec (e.g. Mock is "Dummy API / Default" only,
    even though ``MockBackend.ARMS`` also lists ``\"record\"`` -- that arm is
    already selected via the form's separate Input-mode control, not a
    second API Type), so a curated card never gets a silent \"Other\" catch-
    all appended for arms it deliberately excludes. A card key with **no**
    entry in ``_API_TYPES`` at all (a future backend nobody has curated
    yet) instead falls back to exposing every one of its arms as its own
    Context Handle under one \"Other\" API Type, so a brand-new backend
    still gets a usable (if unlabelled) picker instead of an empty one."""
    known = set(known_arms)
    curated = _API_TYPES.get(card_key)
    if curated is None:
        return [{
            "key": "other",
            "label": "Other",
            "context_handles": [{"key": a, "label": a, "arm": a} for a in known_arms],
        }] if known_arms else []

    groups = []
    for group in curated:
        handles = [
            {"key": hkey, "label": hlabel, "arm": arm}
            for hkey, hlabel, arm in group["context_handles"]
            if arm in known
        ]
        if handles:
            groups.append({"key": group["key"], "label": group["label"], "context_handles": handles})
    return groups


def _public_ai_api_types(engine: str) -> list[dict]:
    return _api_types_for(engine, _public_ai_engine_arms(engine))


def _public_ai_engine_ready(engine: str) -> tuple[bool, str]:
    module = {"gemini": _gemini, "openai": _openai}[engine]
    try:
        return module.ready()
    except Exception as exc:  # never let a landing page 500 on a bad engine
        return False, f"error checking readiness: {exc}"


def public_ai_engine_cards() -> list[dict]:
    """One card per Public AI engine (Gemini, ChatGPT) -- each with its own
    arm list, so the form's arm dropdown is never a merged pile of both
    vendors' arms at once."""
    out = []
    for engine, display in _PUBLIC_AI_ENGINE_DISPLAY.items():
        ok, reason = _public_ai_engine_ready(engine)
        out.append({
            "key": engine,
            "backend": "public_ai",
            "engine": engine,
            "label": display["label"],
            "description": display["description"],
            "implemented": True,
            "ready": ok,
            "reason": "준비됨" if ok else reason,
            "arms": _public_ai_engine_arms(engine),
            "api_types": _public_ai_api_types(engine),
        })
    return out


def ui_backends() -> list[dict]:
    """The landing page's actual card list: Gemini, ChatGPT, then every
    other registered backend (mock, local_llm, ...) one card each. Each
    entry carries ``backend`` (the registry name POST /api/run expects)
    and ``engine`` (``None`` unless the card is a Public AI engine split)
    so the frontend never has to know which backends happen to share a
    registry slot."""
    cards = public_ai_engine_cards()
    for b in backends_view():
        if b["name"] == "public_ai":
            continue  # already represented by its two engine cards above
        cards.append({
            "key": b["name"],
            "backend": b["name"],
            "engine": None,
            "label": b["label"],
            "description": b["description"],
            "implemented": b["implemented"],
            "ready": b["ready"],
            "reason": "준비됨" if b["ready"] else b["reason"],
            "arms": b["arms"],
            "api_types": _api_types_for(b["name"], b["arms"]),
        })
    return cards


def backends_view() -> list[dict]:
    """One entry per registered backend name, for both the landing page and
    /api/config -- the same list, so the UI's cards and the JSON contract
    never drift apart."""
    out = []
    for name in backends_registry.names():
        display = _BACKEND_DISPLAY.get(name, {"label": name, "description": ""})
        ok, reason = _backend_ready(name)
        try:
            implemented = _facade_class(backends_registry.get(name)) is not None
        except KeyError:
            implemented = False
        out.append({
            "name": name,
            "label": display["label"],
            "description": display["description"],
            "implemented": implemented,
            "ready": ok,
            "reason": reason,
            "arms": _backend_arms(name),
        })
    return out


def public_ai_record_names() -> list[str]:
    """Names for the "Record" dropdown in the experiment form: unions two
    sources so both feed the same ``record_id`` field (see
    ``routes_run._load_record_scenario``, which resolves either):

    * ``data/public_ai_records/<exec_id>.json`` -- captured real public_ai
      exchanges (DESIGN.md 4.7.1), auto-written by a previous public_ai run.
    * ``records/<name>.json`` (``aipt.backends.mock.records.RECORD_DIR``)
      -- hand-authored Q&A scenarios shipped with the repo (``perf.json``,
      ``smoke.json``).

    Same honest-empty-list-not-500 posture as the other listing helpers
    here: a missing/unreadable directory just means no records yet.
    A name present in both dirs is deduped (set union) rather than shown
    twice."""
    names: set[str] = set()
    base = public_ai_records_dir()
    if base.exists():
        try:
            names.update(p.stem for p in base.glob("*.json"))
        except OSError:
            pass
    names.update(mock_records.names())
    return sorted(names)


def config_payload() -> dict:
    """The single dict both GET /api/config and the landing page template
    context are built from. ``backends`` stays the 3-way registry view
    (POST /api/run's ``backend`` field only knows public_ai/mock/
    local_llm); ``ui_backends`` is the 4-card view the landing page and
    the form actually render (public_ai split into Gemini/ChatGPT)."""
    cwnd_ok, cwnd_reason = cwndmon.available()
    cap_ok, cap_reason = capture_mod.available()
    algo_names, algo_reason = _congestion_algorithms()
    quic_algo_names, quic_algo_reason = _quic_congestion_algorithms()
    quic_ok, quic_reason = quic_congestion_mod.available()
    return {
        "backends": backends_view(),
        "ui_backends": ui_backends(),
        "public_ai_records": public_ai_record_names(),
        "congestion_algorithms": algo_names,
        "congestion_algorithms_reason": algo_reason,
        # QUIC transport (DESIGN.md section 7/7.1): "http3" is currently
        # mock-only (aipt.backends.quic_mock.backend.QuicMockBackend) --
        # the UI only offers it for the mock card, gated on this being
        # both installed (quic_available) and non-empty
        # (quic_congestion_algorithms), same pattern the TCP algorithm
        # dropdown already uses.
        "quic_available": quic_ok,
        "quic_reason": quic_reason,
        "quic_congestion_algorithms": quic_algo_names,
        "quic_congestion_algorithms_reason": quic_algo_reason,
        "cwnd": {
            "available": cwnd_ok,
            "reason": cwnd_reason,
            "interval_ms": cwndmon.interval_ms(),
        },
        "capture": {
            "available": cap_ok,
            "reason": cap_reason,
            "dir": str(capture_mod.pcap_dir()),
        },
    }


def register(app, templates: Jinja2Templates) -> None:
    """Registers the routes on *app* rather than decorating a module-level
    APIRouter for the template context -- the landing page needs a
    Jinja2Templates instance to render, and create_app() is the one place
    that constructs it (DESIGN.md 5: web UI FastAPI 통합 방침)."""

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request, "index.html", {"config": config_payload()}
        )

    @app.get("/api/config")
    def api_config():
        return JSONResponse(config_payload())
