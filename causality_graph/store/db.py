import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from causality_graph.schema import Edge, NodeType


class MetadataDB:
    def __init__(self, path: Path):
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                node_type TEXT NOT NULL,
                layer TEXT,
                category TEXT,
                unit TEXT,
                good_direction TEXT,
                data_type TEXT,
                range_min REAL,
                range_max REAL,
                default_value TEXT,
                spec_ref TEXT,
                description TEXT
            );
            CREATE TABLE IF NOT EXISTS edges (
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                direction TEXT,
                magnitude TEXT,
                confidence REAL DEFAULT 1.0,
                validated INTEGER DEFAULT 0,
                notes TEXT,
                PRIMARY KEY (from_id, to_id, relation)
            );
        """)
        self._conn.commit()

    def upsert_node(self, node, node_type: NodeType) -> None:
        d = asdict(node)
        layer = d.get("layer")
        if hasattr(layer, "value"):
            layer = layer.value
        self._conn.execute("""
            INSERT OR REPLACE INTO nodes
            (id, name, node_type, layer, category, unit, good_direction,
             data_type, range_min, range_max, default_value, spec_ref, description)
            VALUES (:id, :name, :node_type, :layer, :category, :unit, :good_direction,
                    :data_type, :range_min, :range_max, :default_value, :spec_ref, :description)
        """, {
            "id": d["id"],
            "name": d["name"],
            "node_type": node_type.value,
            "layer": layer,
            "category": d.get("category"),
            "unit": d.get("unit"),
            "good_direction": d.get("good_direction"),
            "data_type": d.get("data_type"),
            "range_min": d.get("range_min"),
            "range_max": d.get("range_max"),
            "default_value": d.get("default_value"),
            "spec_ref": d.get("spec_ref"),
            "description": d.get("description", ""),
        })
        self._conn.commit()

    def upsert_edge(self, edge: Edge) -> None:
        d = asdict(edge)
        self._conn.execute("""
            INSERT OR REPLACE INTO edges
            (from_id, to_id, relation, direction, magnitude, confidence, validated, notes)
            VALUES (:from_id, :to_id, :relation, :direction, :magnitude,
                    :confidence, :validated, :notes)
        """, {
            "from_id": d["from_id"],
            "to_id": d["to_id"],
            "relation": d["relation"].value if hasattr(d["relation"], "value") else d["relation"],
            "direction": d["direction"].value if d["direction"] else None,
            "magnitude": d["magnitude"].value if d["magnitude"] else None,
            "confidence": d.get("confidence", 1.0),
            "validated": int(d.get("validated", False)),
            "notes": d.get("notes", ""),
        })
        self._conn.commit()

    def get_node(self, node_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return dict(row) if row else None

    def filter_nodes(self, node_type: Optional[NodeType] = None) -> list[dict]:
        query = "SELECT * FROM nodes WHERE 1=1"
        params: list = []
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type.value)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def filter_edges(self, validated: Optional[bool] = None,
                     min_confidence: float = 0.0) -> list[dict]:
        query = "SELECT * FROM edges WHERE confidence >= ?"
        params: list = [min_confidence]
        if validated is not None:
            query += " AND validated = ?"
            params.append(int(validated))
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
