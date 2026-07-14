"""What a provider must supply, and how the rest of the suite finds one.

The runner replays one scenario across providers x arms. It must be able to do that
without importing either adapter by name: a suite that hard-codes `import gemini`
next to `import openai` grows a third branch the day a third API is added, and the
two adapters start reaching into each other's helpers. So a provider is a module
that satisfies the protocol below, and `get(name)` is the only way to reach one.

The protocol is deliberately small. `run_arm` owns the conversation -- what turn k
sends, how the model's turn is echoed back, what server-side state gets built and
torn down -- and core owns everything a turn costs: bytes, marks, records, metrics,
capture.
"""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Provider(Protocol):
    """One API, and the arms it can be talked to through."""

    NAME: str
    DEFAULT_MODEL: str
    ARMS: tuple[str, ...]
    #: What a default run includes. An arm can be in ARMS and out of HEADLINE_ARMS
    #: because it is a diagnostic rather than a strategy anyone would ship (Gemini's
    #: `nocontext` answers with no history at all -- a lower bound, not an option).
    HEADLINE_ARMS: tuple[str, ...]

    def ready(self) -> tuple[bool, str]:
        """(ok, reason). False with a reason the operator can act on, never a bare
        False: a run that dies on a missing key must say which key."""

    def api_host(self) -> str:
        """The host the arms talk to. Capture needs it to filter tcpdump down to the
        traffic this run produced, and core must not have to know which provider is
        running to build that filter."""

    def run_arm(self, arm: str, model: str, system: str, steps: list[str],
                measure: str, on_progress=None) -> list[dict]:
        """Replay the scenario over one arm and return one record per turn (plus a
        record per prep call, phased so it is never folded into the totals).

        `on_progress(event)` is called before each call with a progress event, not
        with the finished record: a UI has to be able to say "turn 3 of 10, in
        flight" while the call is still out. See `progress()`.
        """


def progress(on_progress, provider: str, arm: str, phase: str, turn: int,
             turns: int) -> None:
    """Announce the call about to be made, *before* making it.

    Every provider emits the same event, because the UI that renders it cannot know
    which one is running. Without the announcement an arm sits at turn 0 of N for its
    whole run, and a stall is indistinguishable from progress.

    The event also carries the measurement window, which is why `phase` must be exact
    and why the announcement must precede the call. The runner opens the capture on the
    first `steady` event and closes it on a `teardown` one, so:

      * a prep call (`cachegen`, `setup`) announced as prep stays outside the pcap --
        a cache build re-uploads the whole prefix, and a capture holding 185 KB of
        setup cannot be read as evidence of what a 23 KB turn cost;
      * a teardown call (a cache DELETE) announced as teardown stays outside it too;
      * an arm with neither is captured whole, which is correct: all of it is traffic.

    A provider that mislabels a phase does not produce a wrong number -- the socket
    counter and the records still phase correctly -- it produces a pcap that disagrees
    with them, which is worse, because the pcap is what the numbers are checked against.
    """
    if on_progress:
        on_progress({"provider": provider, "arm": arm, "phase": phase,
                     "turn": turn, "turns": turns})


# Adapters are imported lazily. A missing or broken adapter must only break the
# provider that is actually asked for -- importing openai's SDK requirements to run
# a Gemini-only experiment is how one provider's dependency becomes everyone's.
_KNOWN = ("gemini", "openai")


def names() -> tuple[str, ...]:
    return _KNOWN


def get(name: str):
    """The provider module registered under `name`.

    Raises KeyError for an unknown name rather than importing whatever string it was
    handed: `get(user_input)` must not be a way to import an arbitrary module.
    """
    if name not in _KNOWN:
        raise KeyError(f"unknown provider: {name!r} (known: {', '.join(_KNOWN)})")
    return importlib.import_module(f"{__package__}.{name}")
