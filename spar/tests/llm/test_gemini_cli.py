import json
import subprocess
from unittest.mock import patch

import pytest

from spar.llm.gemini_cli import (
    GeminiCliClient,
    GeminiCliError,
    _extract_response,
    _flatten_messages,
)


def _completed(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    return subprocess.CompletedProcess(
        args=["gemini"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_flatten_messages_joins_role_blocks():
    out = _flatten_messages(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
    )
    assert out == "[system]\nbe brief\n\n[user]\nhi"


def test_extract_response_picks_response_field():
    assert _extract_response({"response": "hello"}) == "hello"


def test_extract_response_falls_back_to_text_field():
    assert _extract_response({"text": "hello"}) == "hello"


def test_extract_response_raises_when_missing():
    with pytest.raises(GeminiCliError, match="No response field"):
        _extract_response({"stats": {}})


async def test_chat_returns_parsed_json_response():
    payload = json.dumps({"response": "Hello!"}).encode()
    with patch("spar.llm.gemini_cli.shutil.which", return_value="/usr/bin/gemini"), patch(
        "spar.llm.gemini_cli.subprocess.run",
        return_value=_completed(0, stdout=payload),
    ) as run:
        client = GeminiCliClient(binary="gemini", timeout=5.0)
        out = await client.chat([{"role": "user", "content": "hi"}])

    assert out == "Hello!"
    args, kwargs = run.call_args
    argv = args[0]
    assert argv[0] == "gemini"
    assert "--yolo" in argv
    assert "--output-format" in argv and "json" in argv
    assert kwargs["shell"] is False
    assert kwargs["timeout"] == 5.0


async def test_chat_raises_when_binary_missing():
    with patch("spar.llm.gemini_cli.shutil.which", return_value=None):
        client = GeminiCliClient(binary="gemini")
        with pytest.raises(GeminiCliError, match="binary not found"):
            await client.chat([{"role": "user", "content": "hi"}])


async def test_chat_raises_on_nonzero_exit():
    with patch("spar.llm.gemini_cli.shutil.which", return_value="/usr/bin/gemini"), patch(
        "spar.llm.gemini_cli.subprocess.run",
        return_value=_completed(1, stderr=b"boom"),
    ):
        client = GeminiCliClient()
        with pytest.raises(GeminiCliError, match="exited with 1"):
            await client.chat([{"role": "user", "content": "hi"}])


async def test_chat_raises_on_empty_output():
    with patch("spar.llm.gemini_cli.shutil.which", return_value="/usr/bin/gemini"), patch(
        "spar.llm.gemini_cli.subprocess.run",
        return_value=_completed(0, stdout=b""),
    ):
        client = GeminiCliClient()
        with pytest.raises(GeminiCliError, match="empty output"):
            await client.chat([{"role": "user", "content": "hi"}])


async def test_chat_raises_on_invalid_json():
    with patch("spar.llm.gemini_cli.shutil.which", return_value="/usr/bin/gemini"), patch(
        "spar.llm.gemini_cli.subprocess.run",
        return_value=_completed(0, stdout=b"not json"),
    ):
        client = GeminiCliClient()
        with pytest.raises(GeminiCliError, match="non-JSON output"):
            await client.chat([{"role": "user", "content": "hi"}])


async def test_chat_raises_on_timeout():
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

    with patch("spar.llm.gemini_cli.shutil.which", return_value="/usr/bin/gemini"), patch(
        "spar.llm.gemini_cli.subprocess.run", side_effect=_raise
    ):
        client = GeminiCliClient(timeout=1.0)
        with pytest.raises(GeminiCliError, match="timed out"):
            await client.chat([{"role": "user", "content": "hi"}])
