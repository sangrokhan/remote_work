from pathlib import Path

from causality_graph.extraction.review_queue import ReviewQueue, ReviewStatus
from causality_graph.schema import Edge, EdgeType, Direction, Magnitude
from causality_graph.store.graph import CausalityGraph
from causality_graph.store.db import MetadataDB
from causality_graph.store.embeddings import EmbeddingStore


def _triple_to_edge(triple: dict) -> Edge:
    """Convert a triple dict from the review queue into an Edge object."""
    direction = Direction(triple["direction"]) if triple.get("direction") else None
    magnitude = Magnitude(triple["magnitude"]) if triple.get("magnitude") else None
    return Edge(
        from_id=triple["from"],
        to_id=triple["to"],
        relation=EdgeType(triple["relation"]),
        direction=direction,
        magnitude=magnitude,
        condition=triple.get("condition", ""),
        confidence=triple.get("confidence", 1.0),
        validated=True,
        notes=triple.get("notes", ""),
    )


def commit_approved(
    queue: ReviewQueue,
    graph: CausalityGraph,
    db: MetadataDB,
    embeddings: EmbeddingStore,
    graph_path: Path,
) -> int:
    """
    Move all approved triples from the review queue into the stores.

    Args:
        queue: ReviewQueue with approved items
        graph: CausalityGraph to add edges to
        db: MetadataDB to upsert edges to
        embeddings: EmbeddingStore (for future node embedding; not used in committer yet)
        graph_path: Path to save the graph

    Returns:
        Number of triples committed
    """
    approved = queue.list_by_status(ReviewStatus.APPROVED)
    count = 0
    for item in approved:
        edge = _triple_to_edge(item)
        graph.add_edge(edge)
        db.upsert_edge(edge)
        count += 1
    if count > 0:
        graph.save(graph_path)
    return count
