"""The perf fixture must clear the caching floor, or the whole comparison is
caching-free and pointless.

Gemini 3.x needs >= 4096 input tokens before implicit or explicit caching engages.
The system prompt alone must clear that, so a system-only cache is valid and
implicit caching can fire. Ten turns of ~500 chars each drive the conversation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import load_request

# Real tokenizer measurement (gemini-3.1-flash-lite countTokens, 2026-07-13): the
# system prompt at 20,653 chars counted 4,333 tokens -> 4.77 chars/token. The
# len//4 estimate over-counts English prose, so it cannot be trusted to prove the
# 4096-token floor. CI has no API key, so instead assert a character threshold that
# reliably clears 4096 real tokens at the observed ratio, with margin:
#   4096 tokens * 4.77 chars/token = 19,540 chars; require 20,000 for headroom.
_MIN_SYSTEM_CHARS = 20000


def test_perf_fixture_loads_from_file():
    system, steps, source = load_request("perf")
    assert source == "file:perf.json"
    assert system                       # non-empty system prompt
    assert steps


def test_system_prompt_clears_the_4096_token_floor():
    system, _, _ = load_request("perf")
    # Char proxy for >= 4096 real tokens; see _MIN_SYSTEM_CHARS derivation above.
    assert len(system) >= _MIN_SYSTEM_CHARS


def test_ten_turns_of_about_five_hundred_chars():
    _, steps, _ = load_request("perf")
    assert len(steps) == 10
    for s in steps:
        assert 300 <= len(s) <= 800


def test_fixture_is_deterministic():
    a = load_request("perf")
    b = load_request("perf")
    assert a == b
