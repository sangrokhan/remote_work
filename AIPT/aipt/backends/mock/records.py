"""aipt.backends.mock.records -- scenario-record loading for the Mock backend.

DESIGN.md 4.5/5 (B1): the Mock backend can be driven two ways, both
normalized into one :class:`ScenarioRecord` here:

  * **byte-size sweep** (the original ``tcp_congestion`` mode, kept per
    DESIGN.md §5 "C. 폐기/대체" as an option): fixed dummy-byte sizes per
    turn, generated from a handful of int knobs. No question/answer
    content -- good for an experiment that only cares about wire byte
    counts under a chosen size sweep, not about resembling a real
    conversation.
  * **Q&A record** (new): a JSON file of fixed question/answer pairs, in
    the spirit of the original ``token_traffic/fixtures/perf.json``
    (now ``AIPT/records/perf.json``, see
    ``aipt.labs.external_api`` era ``core/scenario.py``) but scoped down to
    what the Mock backend actually needs -- it never calls a real model, so
    there is no "system list of paragraphs" concept to normalize, just a
    system prompt string and a turn list.

ScenarioRecord schema (JSON)::

    {
      "name": "example",
      "description": "...",                 # optional
      "system_prompt": "...",                # optional, sent once (turn 0)
      "turns": [
        {"question": "...", "answer": "..."},
        ...
      ]
    }

A ``steps``-shaped record (the ``aipt.backends.public_ai`` recorder/replay
schema: top-level ``system`` as a list of paragraphs + ``steps`` as
``{"text", "answer"}`` objects, e.g. ``records/perf.json``) is also
accepted -- see :func:`_turn_of_step` / :func:`_system_prompt_of`.

``aipt.backends.mock.replay`` (DESIGN.md B3) builds the same
:class:`ScenarioRecord` shape out of captured real traffic, except with the
text hollowed out to byte-count-only placeholders -- see that module's
docstring for why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# aipt/backends/mock/records.py -> parents[3] is the AIPT project root
# (mock -> backends -> aipt -> AIPT), matching token_traffic/core/scenario.py's
# original FIXTURE_DIR convention, now repo_root/records.
RECORD_DIR = Path(__file__).resolve().parents[3] / "records"


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
class ScenarioRecord:
    """A named, ordered sequence of turns the Mock backend can replay.

    Equally the product of :func:`load` (a Q&A JSON file on disk),
    :func:`byte_size_scenario` (pure byte-size sweep, no real content), or
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

    Refuse anything malformed here, where the record name and index are
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


def _turn_of_step(raw: object, i: int) -> Turn:
    """One ``steps[i]`` object (public_ai-style record), validated.

    ``steps`` entries use ``text``/``answer`` instead of ``turns``'s
    ``question``/``answer`` (matching the ``aipt.backends.public_ai``
    recorder schema -- see that module's docstring), so a record written
    for the real-model arms, such as ``records/perf.json``, can be
    replayed by the Mock backend without duplicating its content into a
    second ``turns`` list.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"step {i} is not an object: {type(raw).__name__}")
    question = raw.get("text")
    answer = raw.get("answer")
    if not isinstance(question, str):
        raise ValueError(f"step {i} needs a string 'text' field")
    if not isinstance(answer, str):
        raise ValueError(f"step {i} needs a string 'answer' field (mock replay requires a canned answer -- add one before using this record with the Mock backend)")
    return Turn(question=question, answer=answer)


def _system_prompt_of(doc: dict) -> str:
    """Normalize either record's system-prompt shape to one string.

    Q&A records (``turns``-shaped) use a single ``system_prompt`` string.
    public_ai-style records (``steps``-shaped, e.g. ``records/perf.json``)
    use ``system`` as a list of paragraphs, mirroring the real Gemini/OpenAI
    system-instruction shape -- joined the same way
    ``aipt.backends.public_ai.gemini`` joins it before sending.
    """
    system = doc.get("system")
    if isinstance(system, list):
        return "\n\n".join(str(s) for s in system)
    if isinstance(system, str):
        return system
    return doc.get("system_prompt", "")


def load_scenario_record(path: str | Path) -> ScenarioRecord:
    """Load a scenario record straight from a file path (any location).

    Accepts either record shape:

    * ``turns``-shaped (native Mock record): ``{"question", "answer"}``
      per turn, ``system_prompt`` as a single string.
    * ``steps``-shaped (public_ai record, e.g. ``records/perf.json``):
      ``{"text", "answer"}`` per step, ``system`` as a list of paragraphs.
      Every step needs an ``answer`` filled in to be replayable here; a
      ``steps`` record with no answers (the public_ai-only case, where a
      real model generates them at run time) will raise.

    ``turns`` takes precedence if a document somehow has both.
    """
    p = Path(path)
    doc = json.loads(p.read_text())
    if doc.get("turns") is not None:
        turns = [_turn_of(t, i) for i, t in enumerate(doc.get("turns") or [])]
    else:
        turns = [_turn_of_step(t, i) for i, t in enumerate(doc.get("steps") or [])]
    return ScenarioRecord(
        name=doc.get("name", p.stem),
        system_prompt=_system_prompt_of(doc),
        description=doc.get("description", ""),
        turns=turns,
    )


def names() -> list[str]:
    """Record names available under ``RECORD_DIR`` (``*.json`` stems)."""
    if not RECORD_DIR.exists():
        return []
    return sorted(p.stem for p in RECORD_DIR.glob("*.json"))


def load(name: str) -> ScenarioRecord:
    """Load a registered scenario record by name from ``RECORD_DIR``."""
    if name not in names():
        raise KeyError(
            f"unknown mock record: {name!r} (have: {', '.join(names())})"
        )
    return load_scenario_record(RECORD_DIR / f"{name}.json")


# --- byte-size sweep mode --------------------------------------------------
#
# DESIGN.md 5 "C. 폐기/대체": the original tcp_congestion "N-byte dummy"
# approach is superseded by Q&A record replay for realistic conversations,
# but kept as an explicit option for a pure "sweep the byte size and watch
# TCP behaviour" experiment that has no interest in text content at all.


def byte_size_scenario(
    *,
    num_turns: int,
    turn_user_msg_bytes: int,
    mock_response_bytes: int,
    system_prompt_bytes: int = 0,
) -> ScenarioRecord:
    """A synthetic :class:`ScenarioRecord` of pure-byte-size dummy turns.

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
    return ScenarioRecord(name="byte-sweep", turns=turns)
