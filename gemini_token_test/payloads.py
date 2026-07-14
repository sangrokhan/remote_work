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


# --- Echoing the model's turn back -------------------------------------------
#
# A client that keeps the history has to put the model's turn back on the wire, and
# the only faithful version of that turn is the one the server sent. Rebuilding it
# from the answer text drops whatever the text never carried -- above all the
# `thought` step and its signature, which every Gemini 3 response returns:
#
#     {"type": "thought", "signature": "EjQKMg..."}
#     {"type": "model_output", "content": [{"type": "text", "text": "..."}]}
#
# Measured (probe.probe_signature_echo, 2026-07-14): echoing the thought step is
# accepted and adds 0 input tokens; dropping it is accepted too, on a text-only
# conversation. What the echo costs is upload -- roughly 1 KB a turn -- and that is
# precisely the number this experiment exists to report honestly. A history that
# quietly omits what a real client sends measures a client nobody runs.

def model_steps_from_response(data: dict, fallback_text: str = "") -> list:
    """The model's turn, exactly as the interactions endpoint returned it.

    Falls back to a rebuilt model_output step when there are no steps to echo --
    an errored call, or a response shape we have never seen.
    """
    steps = (data or {}).get("steps")
    if isinstance(steps, list) and steps:
        return steps
    return [model_step(fallback_text)] if fallback_text else []


def model_content_from_response(data: dict, fallback_text: str = "") -> dict:
    """The model's turn, exactly as generateContent returned it.

    The candidate's `content` already is a Content: role `model`, parts carrying
    `text` and `thoughtSignature`. Echo it whole rather than keeping the text.
    """
    cands = (data or {}).get("candidates") or []
    content = cands[0].get("content") if cands else None
    if isinstance(content, dict) and content.get("parts"):
        return {"role": content.get("role", "model"), "parts": content["parts"]}
    return model_content(fallback_text)


# --- Reading text back out --------------------------------------------------

def answer_text(steps) -> str:
    """The answer, and only the answer: the text of the `model_output` steps.

    `extract_text` collects every text leaf in whatever it is handed, which is right
    for a payload whose shape is unknown but wrong for a response -- with
    `thinking_summaries` on, a thought step carries text too, and stapling it to the
    answer would put the model's reasoning into the conversation history.
    """
    out = []
    for s in steps if isinstance(steps, list) else []:
        if isinstance(s, dict) and s.get("type") == "model_output":
            out.append(extract_text(s.get("content")))
    return "".join(out)


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
