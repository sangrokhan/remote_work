from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from spar.router.schemas import Route, RouteResult

ROUTE_EXAMPLES: dict[Route, list[str]] = {
    Route.STRUCTURED_LOOKUP: [
        "What is the default value of maxTxPower?",
        "Show me the range of pMax parameter",
        "List parameters in NRCellDU MO",
        "What alarms are related to RACH failure?",
        "What is the formula for RRC success rate counter?",
    ],
    Route.DEFINITION_EXPLAIN: [
        "What is Carrier Aggregation?",
        "Explain the difference between FDD and TDD",
        "What does BWP stand for?",
        "Describe the handover procedure",
        "What is beam management in NR?",
    ],
    Route.PROCEDURAL: [
        "How do I configure RACH parameters?",
        "Steps to enable Carrier Aggregation",
        "How to install the RAN software package?",
        "Procedure for activating a new cell",
        "How to configure QoS profiles?",
    ],
    Route.DIAGNOSTIC: [
        "Why is the handover failure rate high?",
        "RACH congestion issue after software upgrade",
        "Cell is stuck in blocking state",
        "Throughput dropped after parameter change",
        "Why are users experiencing call drops?",
    ],
    Route.COMPARATIVE: [
        "What changed in v7.0 compared to v6.0?",
        "Difference between SA and NSA mode configuration",
        "How does the new preamble format compare to the old one?",
        "What features were added in the latest release?",
        "Compare LTE and NR RACH procedures",
    ],
    Route.DEFAULT_RAG: [
        "Tell me about the RAN system",
        "General information about 5G",
        "Overview of the network",
    ],
}

_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingRouter:
    """Layer 2: cosine similarity against route centroid embeddings."""

    def __init__(self, threshold: float = 0.65, model_name: str = _MODEL_NAME) -> None:
        self.threshold = threshold
        self._model = SentenceTransformer(model_name)
        self._centroids = self._build_centroids()

    def _build_centroids(self) -> dict[Route, np.ndarray]:
        centroids: dict[Route, np.ndarray] = {}
        for route, examples in ROUTE_EXAMPLES.items():
            embs = self._model.encode(examples, normalize_embeddings=True)
            centroids[route] = np.mean(embs, axis=0)
        return centroids

    def route(self, query: str) -> RouteResult | None:
        q_emb = self._model.encode([query], normalize_embeddings=True)[0]
        best_route, best_score = Route.DEFAULT_RAG, -1.0
        for route, centroid in self._centroids.items():
            score = float(np.dot(q_emb, centroid))
            if score > best_score:
                best_score, best_route = score, route

        if best_score < self.threshold:
            return None

        return RouteResult(route=best_route, confidence=best_score, layer="embedding")
