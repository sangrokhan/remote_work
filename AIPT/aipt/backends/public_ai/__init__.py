"""aipt.backends.public_ai -- Gemini / ChatGPT backend (DESIGN.md 4.5).

Generalizes the former ``token_traffic/providers/{gemini,openai}.py``
adapters under the ``aipt.backends.base.Backend`` protocol (DESIGN.md 5,
A2/B2). The real per-engine adapters live in ``gemini.py``/``openai.py``
(each exposes its own ``GeminiBackend``/``OpenAIBackend`` implementing the
full ``connect``/``send_turn``/``close`` lifecycle plus the legacy
``run_arm`` for parity testing); ``recorder.py`` captures real traffic to
fixtures (B2).

``aipt.backends.get("public_ai")`` resolves to *this* module, and the
client is expected to hold ``PublicAIBackend`` (or
``PublicAIBackend(engine="openai")``) -- a thin engine-selecting facade
over the two adapters, so the client's connect/send_turn/close calls do
not have to know which vendor a run picked. ``ARMS``/``HEADLINE_ARMS`` on
the facade are gemini's by default (the larger of the two arm sets) and
switch to the selected engine's the moment ``engine=`` is given or
``connect`` is called with an arm belonging to the other engine.
"""

from __future__ import annotations

from aipt.backends import base
from aipt.backends.public_ai import gemini as _gemini
from aipt.backends.public_ai import openai as _openai

NAME = "public_ai"

_ENGINES = {"gemini": _gemini, "openai": _openai}


def _engine_for_arm(arm: str) -> str:
    if arm in _gemini.ARMS:
        return "gemini"
    if arm in _openai.ARMS:
        return "openai"
    raise ValueError(
        f"unknown public_ai arm: {arm!r} "
        f"(gemini: {', '.join(_gemini.ARMS)}; openai: {', '.join(_openai.ARMS)})"
    )


def engine_for_arm(arm: str) -> str:
    """Public wrapper over :func:`_engine_for_arm` -- lets a caller (e.g. the
    web run route, which needs to know which engine an arm belongs to
    *before* connecting, in order to wire ``recorder.recording_backend``)
    resolve the engine without importing the private helper directly.
    """
    return _engine_for_arm(arm)


class PublicAIBackend:
    """``aipt.backends.base.Backend`` over Gemini and/or OpenAI.

    Two engines share this one registry slot (DESIGN.md 4.5 groups them as
    one backend, "public_ai"), so this facade picks the right adapter per
    call rather than making the client import ``gemini``/``openai``
    directly -- exactly the reason ``aipt.backends.get()`` exists at all
    (see ``aipt/backends/__init__.py``'s docstring): a caller must not have
    to know which module implements a name to use it.

    ``engine`` can be fixed at construction (``PublicAIBackend("openai")``)
    or left to be inferred from the arm passed to ``connect`` -- an
    OpenAI-only caller that always uses e.g. ``"responses"`` never has to
    say which engine it means.
    """

    NAME = NAME
    #: Advertises gemini's model as the default; a caller targeting openai
    #: should pass ``model=`` explicitly to connect() (as any caller
    #: targeting a non-default model already must).
    DEFAULT_MODEL = _gemini.DEFAULT_MODEL
    #: The union of both engines' arms -- connect() picks the concrete
    #: engine from whichever arm name is actually passed.
    ARMS = _gemini.ARMS + _openai.ARMS
    HEADLINE_ARMS = _gemini.HEADLINE_ARMS + _openai.HEADLINE_ARMS
    transport = base.DEFAULT_TRANSPORT

    def __init__(self, engine: str | None = None) -> None:
        if engine is not None and engine not in _ENGINES:
            raise ValueError(f"unknown public_ai engine: {engine!r} (known: gemini, openai)")
        self._engine_name = engine
        self._backend = None  # concrete GeminiBackend/OpenAIBackend instance

    def _select(self, arm: str):
        name = self._engine_name or _engine_for_arm(arm)
        module = _ENGINES[name]
        if self._backend is None or self._engine_name != name:
            self._engine_name = name
            self._backend = (module.GeminiBackend() if name == "gemini"
                              else module.OpenAIBackend())
        return self._backend

    def ready(self) -> tuple[bool, str]:
        if self._engine_name:
            return _ENGINES[self._engine_name].ready()
        # No engine picked yet: ready only if at least one is (matches the
        # "say which key" contract -- report the first failing reason if
        # both are unready, since a caller with neither key configured needs
        # to know at least one of them).
        g_ok, g_reason = _gemini.ready()
        if g_ok:
            return True, ""
        o_ok, o_reason = _openai.ready()
        if o_ok:
            return True, ""
        return False, f"gemini: {g_reason}; openai: {o_reason}"

    def api_host(self) -> str:
        if self._backend is not None:
            return self._backend.api_host()
        name = self._engine_name or "gemini"
        return _ENGINES[name].api_host()

    def connect(self, arm: str, model: str, system: str) -> None:
        backend = self._select(arm)
        backend.connect(arm, model, system)

    def send_turn(self, turn: int, question: str, measure: str, on_progress=None):
        if self._backend is None:
            raise RuntimeError("send_turn called before connect")
        return self._backend.send_turn(turn, question, measure, on_progress=on_progress)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()


__all__ = ["NAME", "PublicAIBackend", "engine_for_arm"]
