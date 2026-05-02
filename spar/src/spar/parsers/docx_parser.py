from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def _slugify(text: str, max_len: int) -> str:
    if not text.strip():
        return "unnamed"
    slug = re.sub(r"[^\w\s.\-]", "", text)
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:max_len] if slug else "unnamed"


@dataclass
class ExtractedTable:
    path: Path
    section_path: list[str]
    seq: int


@dataclass
class ExtractedImage:
    path: Path
    section_path: list[str]
    seq: int
    ext: str


@dataclass
class ParseResult:
    markdown: str
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
