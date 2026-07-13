"""Load .env into os.environ, without a dependency.

Import this before anything that reads os.environ at module scope
(openai_client, metrics). Values already present in the real environment win —
an explicit `export` should always beat the file.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def load(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


load()
