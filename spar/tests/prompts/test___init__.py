from __future__ import annotations

import pytest

from spar.prompts import load_prompt


def test_load_prompt_returns_string():
    text = load_prompt("router_system.txt")
    assert isinstance(text, str)
    assert len(text) > 0


def test_load_prompt_strips_whitespace():
    text = load_prompt("router_system.txt")
    assert text == text.strip()


def test_load_prompt_abbrev_conflict_has_placeholder():
    text = load_prompt("abbrev_conflict.txt")
    assert "{items}" in text


def test_load_prompt_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent_prompt.txt")


def test_load_verify_prompt():
    from spar.prompts import load_prompt
    text = load_prompt("verify.txt")
    assert "{query}" in text
    assert "{answer}" in text
    assert "{contexts_summary}" in text


def test_load_tool_call_rewrite_prompt():
    from spar.prompts import load_prompt
    text = load_prompt("tool_call_rewrite.txt")
    assert "{query}" in text
    assert "{reason}" in text
