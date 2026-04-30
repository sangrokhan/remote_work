from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Route(str, Enum):
    STRUCTURED_LOOKUP = "structured_lookup"
    DEFINITION_EXPLAIN = "definition_explain"
    PROCEDURAL = "procedural"
    DIAGNOSTIC = "diagnostic"
    COMPARATIVE = "comparative"
    DEFAULT_RAG = "default_rag"


@dataclass
class RouteResult:
    route: Route
    confidence: float
    layer: str
    entities: dict[str, Any] = field(default_factory=dict)
    product: str | None = None
    release: str | None = None
