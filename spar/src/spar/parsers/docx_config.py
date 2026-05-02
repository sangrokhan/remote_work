from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocxParseConfig:
    heading_depth: int = 2
    output_dir: Path = field(default_factory=lambda: Path("output"))
    slugify_max_len: int = 30
