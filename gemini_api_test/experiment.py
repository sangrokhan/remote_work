"""Run the same N-turn conversation two ways and collect per-turn metrics.

stateless: turn k sends the full history (turns 1..k)  -> O(N^2) tokens
delta    : turn k sends only turn k                    -> O(N) tokens

Only traffic/tokens are compared. Delta mode loses context by design.
"""

from __future__ import annotations

from gemini_client import call_gemini, ENDPOINT


def _make_messages(turns: int, message_chars: int) -> list[str]:
    """Synthetic, deterministic user messages of ~message_chars each."""
    msgs = []
    for k in range(1, turns + 1):
        prefix = f"Turn {k}: "
        filler = ("the quick brown fox jumps over the lazy dog. " * 100)
        body = (prefix + filler)[:max(message_chars, len(prefix) + 1)]
        msgs.append(body)
    return msgs


def _user_content(text: str) -> dict:
    return {"role": "user", "parts": [{"text": text}]}


def run_experiment(turns: int, message_chars: int, model: str, api_key: str) -> dict:
    messages = _make_messages(turns, message_chars)
    records = []

    # Stateless: history grows each turn.
    history: list[dict] = []
    for k, msg in enumerate(messages, start=1):
        history.append(_user_content(msg))
        res = call_gemini(model, list(history), api_key, mode="stateless", turn=k)
        records.append(res.as_dict())
        # Append a synthetic assistant turn so next request truly resends more.
        history.append({"role": "model", "parts": [{"text": "(ack)"}]})

    # Delta: only the current turn.
    for k, msg in enumerate(messages, start=1):
        res = call_gemini(model, [_user_content(msg)], api_key, mode="delta", turn=k)
        records.append(res.as_dict())

    return {
        "params": {
            "turns": turns,
            "message_chars": message_chars,
            "model": model,
            "endpoint": ENDPOINT,
        },
        "records": records,
    }
