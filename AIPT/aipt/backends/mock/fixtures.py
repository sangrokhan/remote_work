"""aipt.backends.mock.fixtures -- fixture loading for the Mock backend.

DESIGN.md 4.5/5 (B1): the Mock backend can be driven two ways, both
normalized into one :class:`Fixture` here:

  * **byte-size sweep** (the original ``tcp_congestion`` mode, kept per
    DESIGN.md §5 "C. 폐기/대체" as an option): fixed dummy-byte sizes per
    turn, generated from a handful of int knobs. No question/answer
    content -- good for an experiment that only cares about wire byte
    counts under a chosen size sweep, not about resembling a real
    conversation.
  * **Q&A fixture** (new): a JSON file of fixed question/answer pairs, in
    the spirit of ``token_traffic/fixtures/perf.json`` (see
    ``aipt.labs.external_api`` era ``core/scenario.py``) but scoped down to
    what the Mock backend actually needs -- it never calls a real model, so
    there is no "system list of paragraphs" concept to normalize, just a
    system prompt string and a turn list.

Fixture schema (JSON)::

    {
      "name": "example",
      "description": "...",                 # optional
      "system_prompt": "...",                # optional, sent once (turn 0)
      "turns": [
        {"question": "...", "answer": "..."},
        ...
      ]
    }

``aipt.backends.mock.replay`` (DESIGN.md B3) builds the same :class:`Fixture`
shape out of captured real traffic, except with the text hollowed out to
byte-count-only placeholders -- see that module's docstring for why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# aipt/backends/mock/fixtures.py -> parents[3] is the AIPT project root
# (mock -> backends -> aipt -> AIPT), matching token_traffic/core/scenario.py's
# FIXTURE_DIR convention (repo_root/fixtures).
FIXTURE_DIR = Path(__file__).resolve().parents[3] / "fixtures"


@dataclass
class Turn:
    """One fixed question/answer pair."""

    question: str
    answer: str

    @property
    def question_bytes(self) -> int:
        return len(self.question.encode())

    @property
    def answer_bytes(self) -> int:
        return len(self.answer.encode())


@dataclass
class Fixture:
    """A named, ordered sequence of turns the Mock backend can replay.

    Equally the product of :func:`load` (a Q&A JSON file on disk),
    :func:`byte_size_fixture` (pure byte-size sweep, no real content), or
    ``aipt.backends.mock.replay.from_capture`` (captured real traffic,
    byte-size only). Every consumer (``aipt.backends.mock.server``,
    ``aipt.backends.mock.conversation``) only ever depends on this shape,
    not on which of the three built it.
    """

    name: str
    system_prompt: str = ""
    turns: list[Turn] = field(default_factory=list)
    description: str = ""

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self):
        return iter(self.turns)


def _turn_of(raw: object, i: int) -> Turn:
    """One JSON turn object, validated.

    Refuse anything malformed here, where the fixture name and index are
    known, rather than downstream where a missing 'answer' just becomes an
    empty response nobody can explain (mirrors
    ``token_traffic/core/scenario.py``'s ``_text_of`` refusal style).
    """
    if not isinstance(raw, dict):
        raise ValueError(f"turn {i} is not an object: {type(raw).__name__}")
    question = raw.get("question")
    answer = raw.get("answer")
    if not isinstance(question, str):
        raise ValueError(f"turn {i} needs a string 'question' field")
    if not isinstance(answer, str):
        raise ValueError(f"turn {i} needs a string 'answer' field")
    return Turn(question=question, answer=answer)


def load_qa_fixture(path: str | Path) -> Fixture:
    """Load a Q&A fixture straight from a file path (any location)."""
    p = Path(path)
    doc = json.loads(p.read_text())
    turns = [_turn_of(t, i) for i, t in enumerate(doc.get("turns") or [])]
    return Fixture(
        name=doc.get("name", p.stem),
        system_prompt=doc.get("system_prompt", ""),
        description=doc.get("description", ""),
        turns=turns,
    )


def names() -> list[str]:
    """Fixture names available under ``FIXTURE_DIR`` (``*.json`` stems)."""
    if not FIXTURE_DIR.exists():
        return []
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.json"))


def load(name: str) -> Fixture:
    """Load a registered Q&A fixture by name from ``FIXTURE_DIR``."""
    if name not in names():
        raise KeyError(
            f"unknown mock fixture: {name!r} (have: {', '.join(names())})"
        )
    return load_qa_fixture(FIXTURE_DIR / f"{name}.json")


# --- byte-size sweep mode --------------------------------------------------
#
# DESIGN.md 5 "C. 폐기/대체": the original tcp_congestion "N-byte dummy"
# approach is superseded by Q&A fixture replay for realistic conversations,
# but kept as an explicit option for a pure "sweep the byte size and watch
# TCP behaviour" experiment that has no interest in text content at all.


def byte_size_fixture(
    *,
    num_turns: int,
    turn_user_msg_bytes: int,
    mock_response_bytes: int,
    system_prompt_bytes: int = 0,
) -> Fixture:
    """A synthetic :class:`Fixture` of pure-byte-size dummy turns.

    Each question is ``turn_user_msg_bytes`` bytes of filler (the system
    prompt's bytes are folded into turn 0's question only, matching the
    original ``tcp_congestion.conversation.turn_prompt_size`` convention of
    sending the system prompt once). Each answer is
    ``mock_response_bytes`` bytes of filler. No cumulative-history growth
    is baked in here -- that is
    ``aipt.backends.mock.conversation.build_turns``'s job, which computes
    per-turn *request* size from these fixed per-turn contributions plus
    the running history.
    """
    if num_turns <= 0:
        raise ValueError("num_turns must be positive")
    turns = []
    for i in range(num_turns):
        question = "x" * turn_user_msg_bytes
        if i == 0 and system_prompt_bytes:
            question = ("x" * system_prompt_bytes) + question
        turns.append(Turn(question=question, answer="x" * mock_response_bytes))
    return Fixture(name="byte-sweep", turns=turns)
