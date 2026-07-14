"""Request-payload shapes shared by every arm, and the one way to read text back out.

Two wire vocabularies are in play and they are not interchangeable:

  generateContent : {"role": "user"|"model", "parts": [{"text": ...}]}   -> Content
  interactions    : {"type": "user_input"|"model_output",
                     "content": [{"type": "text", "text": ...}]}          -> Step

Every module that talks to the API built its own copy of these, and probe.py and
interaction_client.py each carried their own text-extraction walker. One shape, one
definition: a divergence between two copies would silently change what an arm sends,
which is exactly the thing the experiment measures.

No imports from the rest of the package -- everything here is a pure data shape, so
this module can never be part of an import cycle.
"""

from __future__ import annotations


# --- generateContent Content ------------------------------------------------

def user_content(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def model_content(text: str) -> dict:
    return {"role": "model", "parts": [{"text": text}]}


# --- Interactions Step ------------------------------------------------------

def user_step(text: str) -> dict:
    return {"type": "user_input", "content": [{"type": "text", "text": text}]}


def model_step(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def single_step_input(text: str) -> list:
    """The `input` field for a turn that sends one question and nothing else."""
    return [user_step(text)]


# --- Reading text back out --------------------------------------------------

def extract_text(obj) -> str:
    """Collect every {"type": "text", "text": ...} leaf in a payload.

    Interaction responses nest the answer differently depending on whether the call
    streamed, so walking for the leaves is what works against both shapes.
    """
    out: list[str] = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("type") == "text" and isinstance(o.get("text"), str):
                out.append(o["text"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return "".join(out)
