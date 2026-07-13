"""Load a conversation fixture: one fixed system prompt + an ordered list of turns.

Fixtures are copied verbatim from gemini_token_test/requests/ so the OpenAI and
Gemini experiments are driven by the exact same words.

Shape on disk:

    {"name": ..., "description": ..., "system": [str, ...], "steps": [{"text": str}, ...]}

The system prompt must be byte-identical on every turn — OpenAI's prompt cache
matches on an exact prefix, so a timestamp or random id in there would silently
destroy every cache hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent / "requests_fixtures"


@dataclass(frozen=True)
class Fixture:
    name: str
    description: str
    system: str
    steps: tuple[str, ...]

    @property
    def system_chars(self) -> int:
        return len(self.system)

    def head(self, turns: int) -> tuple[str, ...]:
        return self.steps[:turns]


def load(name: str = "perf") -> Fixture:
    path = FIXTURE_DIR / f"{name}.json"
    raw = json.loads(path.read_text())
    system = "\n\n".join(raw["system"]) if isinstance(raw["system"], list) else raw["system"]
    steps = tuple(s["text"] for s in raw["steps"])
    return Fixture(
        name=raw.get("name", name),
        description=raw.get("description", ""),
        system=system,
        steps=steps,
    )
