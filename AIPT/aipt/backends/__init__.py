"""aipt.backends -- the 3-backend common client structure (DESIGN.md 4.5).

The client side (cwnd/capture/stats export in ``aipt.core`` / ``aipt.export``)
talks to exactly one of three backends through a single protocol, defined in
``aipt.backends.base``:

  * ``public_ai``  -- Gemini / ChatGPT over the real network (billed).
  * ``mock``       -- fixed/replayed JSON traffic, no network cost.
  * ``local_llm``  -- a standard serving engine (llama.cpp/vLLM) behind an
                       in-repo gateway that owns the transport experiment
                       surface (HTTP/1.1 today).

Backends are looked up by name through :func:`get`, mirroring the lazy,
name-gated lookup in the ``token_traffic`` providers registry: a caller must
never be able to import an arbitrary module by passing a string straight to
``importlib``.
"""

from __future__ import annotations

import importlib

# Registered backend package names. A backend package is expected to expose
# a concrete implementation of aipt.backends.base.Backend, but during the
# parallel-build phase these packages may only contain NotImplementedError
# stubs -- get() still resolves the module, it just doesn't guarantee the
# backend is usable yet.
_KNOWN = ("public_ai", "mock", "local_llm")


def names() -> tuple[str, ...]:
    return _KNOWN


def get(name: str):
    """The backend package registered under ``name``.

    Raises KeyError for an unknown name rather than importing whatever
    string it was handed.
    """
    if name not in _KNOWN:
        raise KeyError(f"unknown backend: {name!r} (known: {', '.join(_KNOWN)})")
    return importlib.import_module(f"{__package__}.{name}")
