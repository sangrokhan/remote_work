from __future__ import annotations

import asyncio
import json
import shutil
import subprocess

GEMINI_DEFAULT_TIMEOUT_SECONDS = 300.0


class GeminiCliError(RuntimeError):
    pass


class GeminiCliClient:
    """Headless Gemini CLI fallback. Duck-types LLMClient (chat + model)."""

    def __init__(
        self,
        binary: str = "gemini",
        timeout: float = GEMINI_DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._binary = binary
        self._timeout = timeout

    @property
    def model(self) -> str:
        return "gemini-cli"

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        del temperature, max_tokens  # CLI uses its own defaults
        if shutil.which(self._binary) is None:
            raise GeminiCliError(f"Gemini CLI binary not found: {self._binary}")

        prompt = _flatten_messages(messages)
        argv = [
            self._binary,
            "--yolo",
            "--output-format",
            "json",
            "-p",
            prompt,
        ]

        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                argv,
                capture_output=True,
                timeout=self._timeout,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GeminiCliError(
                f"Gemini CLI timed out after {self._timeout}s"
            ) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.decode(errors="replace").strip()
            raise GeminiCliError(
                f"Gemini CLI exited with {completed.returncode}: {stderr}"
            )

        raw = completed.stdout.decode(errors="replace").strip()
        if not raw:
            raise GeminiCliError("Gemini CLI produced empty output")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeminiCliError(
                f"Gemini CLI returned non-JSON output: {raw[:200]}"
            ) from exc

        return _extract_response(payload)


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _extract_response(payload: object) -> str:
    if isinstance(payload, dict):
        for key in ("response", "text", "output"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    raise GeminiCliError(f"No response field in Gemini CLI output: {payload!r}")
