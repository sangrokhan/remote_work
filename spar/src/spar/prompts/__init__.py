from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """prompts/ 디렉토리에서 프롬프트 파일을 읽어 반환."""
    return (_DIR / name).read_text(encoding="utf-8").strip()
