"""The conversation the arms replay, loaded from a fixture.

Every arm answers the same questions against the same system prompt, or the comparison
means nothing: an arm that looks cheap because it was asked something shorter is not
cheap. So the scenario is one file, loaded once per run, and the runner hands the same
`system` and `steps` to every provider.

A fixture is JSON: {name, description, system: [...], steps: [...]}. `system` is a list
of paragraphs rather than one string because the perf fixture's prompt is deliberately
large -- over 4k tokens, so that both implicit and explicit caching engage -- and a
prompt that size is unreadable and unreviewable as a single line.

A step may be a bare string or a `{"text": ...}` object -- the perf fixture uses the
latter, to leave room for per-step annotations. Normalizing that here, at the only door
into a run, is the point of this module: a provider takes `list[str]`, and a dict that
slips past becomes `{"text": {"text": "..."}}` on the wire -- a malformed request the
API would accept the shape of and answer from nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
DEFAULT = "perf"


def names() -> list[str]:
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


def _text_of(step, i: int) -> str:
    """One step, as the string a provider will put in a user turn.

    Refuse anything else loudly. A dict of the wrong shape has to fail here, where the
    fixture is named and the index is known, rather than downstream where it becomes a
    nested part in a live request body.
    """
    if isinstance(step, str):
        return step
    if isinstance(step, dict) and isinstance(step.get("text"), str):
        return step["text"]
    raise ValueError(
        f"step {i} is neither a string nor {{\"text\": ...}}: {type(step).__name__}")


def load(name: str = DEFAULT, turns: int | None = None) -> dict:
    """{name, description, system, steps}. `turns` truncates the thread.

    Truncating rather than cycling: the steps lean on each other through pronouns and
    ellipsis ("roll that back"), so the first n of them is a coherent conversation and
    a resampled n is not.
    """
    if name not in names():
        raise KeyError(f"unknown fixture: {name!r} (have: {', '.join(names())})")
    doc = json.loads((FIXTURE_DIR / f"{name}.json").read_text())

    system = doc.get("system") or []
    if isinstance(system, list):
        system = "\n\n".join(system)
    steps = [_text_of(s, i) for i, s in enumerate(doc.get("steps") or [])]
    if turns is not None:
        if turns < 1:
            raise ValueError("a run needs at least one turn")
        if turns > len(steps):
            raise ValueError(
                f"fixture {name!r} has {len(steps)} turns; {turns} were asked for. "
                "Add steps to the fixture rather than repeating them -- a repeated "
                "question is answered from context and costs nothing like a new one.")
        steps = steps[:turns]

    return {
        "name": doc.get("name", name),
        "description": doc.get("description", ""),
        "system": system,
        "steps": steps,
    }
