"""One reading of the mock switch, for everybody.

There used to be two. `TRAFFIC_MOCK=true` satisfied one provider's parser and not the
other's, so half the run was synthetic and half of it was billed -- and the run was
then filed in the *live* bucket with `mock: false`, because the flag that decides the
bucket had its own third reading. Mock data indistinguishable from measured data is the
one failure this package is built to make impossible, and it is exactly what a
disagreement between two truthy-parsers produces.

So the parse lives here and nowhere else, and it is generous on purpose: anything a
person would plausibly type to mean yes means yes. The dangerous direction is not
accepting `on`; it is one module accepting it while another does not.
"""

from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}


def flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUE


def is_mock(provider: str = "") -> bool:
    """Whether calls are synthetic. `TRAFFIC_MOCK` covers the suite; `<PROVIDER>_MOCK`
    covers one provider, so a Gemini-only key can still exercise the OpenAI arms."""
    if flag("TRAFFIC_MOCK"):
        return True
    return bool(provider) and flag(f"{provider.upper()}_MOCK")
