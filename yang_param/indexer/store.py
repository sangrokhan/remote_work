from __future__ import annotations
import json
import sqlite3
from dataclasses import asdict

from indexer.normalizer import NodeRecord, normalize_all
from indexer.parser import parse_yang_dir

_DDL = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    schema_path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    module TEXT NOT NULL,
    namespace TEXT NOT NULL,
    prefix TEXT NOT NULL,
    node_kind TEXT NOT NULL,
    config INTEGER NOT NULL DEFAULT 1,
    parent_id TEXT,
    children_ids TEXT NOT NULL DEFAULT '[]',
    keys TEXT NOT NULL DEFAULT '[]',
    type_info TEXT NOT NULL DEFAULT '{}',
    default_value TEXT,
    mandatory INTEGER NOT NULL DEFAULT 0,
    description TEXT NOT NULL DEFAULT '',
    when_expr TEXT,
    must_exprs TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_module ON nodes(module);
CREATE INDEX IF NOT EXISTS idx_kind ON nodes(node_kind);
"""

def _row_to_record(row: dict) -> NodeRecord:
    return NodeRecord(
        node_id=row["node_id"],
        schema_path=row["schema_path"],
        name=row["name"],
        module=row["module"],
        namespace=row["namespace"],
        prefix=row["prefix"],
        node_kind=row["node_kind"],
        config=bool(row["config"]),
        parent_id=row["parent_id"],
        children_ids=json.loads(row["children_ids"]),
        keys=json.loads(row["keys"]),
        type_info=json.loads(row["type_info"]),
        default=row["default_value"],
        mandatory=bool(row["mandatory"]),
        description=row["description"] or "",
        when_expr=row["when_expr"],
        must_exprs=json.loads(row["must_exprs"]),
    )

class SchemaStore:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._cache: dict[str, NodeRecord] = {}
        self._path_index: dict[str, str] = {}
        self._name_index: dict[str, list[str]] = {}

    def build(self, yang_dir: str, modules: list[str] | None = None) -> None:
        raw_nodes = list(parse_yang_dir(yang_dir, modules))
        records = normalize_all(raw_nodes)
        self._persist(records)
        self._load_indexes()

    def _persist(self, records: list[NodeRecord]) -> None:
        con = sqlite3.connect(self._db_path)
        con.executescript(_DDL)
        con.executemany(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (r.node_id, r.schema_path, r.name, r.module, r.namespace, r.prefix,
                 r.node_kind, int(r.config), r.parent_id,
                 json.dumps(r.children_ids), json.dumps(r.keys),
                 json.dumps(r.type_info), r.default,
                 int(r.mandatory), r.description, r.when_expr,
                 json.dumps(r.must_exprs))
                for r in records
            ],
        )
        con.commit()
        con.close()

    def _load_indexes(self) -> None:
        con = sqlite3.connect(self._db_path)
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM nodes").fetchall()
        con.close()
        self._cache.clear()
        self._path_index.clear()
        self._name_index.clear()
        for row in rows:
            rec = _row_to_record(dict(row))
            self._cache[rec.node_id] = rec
            self._path_index[rec.schema_path] = rec.node_id
            self._name_index.setdefault(rec.name.lower(), []).append(rec.node_id)

    def count(self) -> int:
        return len(self._cache)

    def list_modules(self) -> list[str]:
        return sorted({r.module for r in self._cache.values()})

    def get_by_id(self, node_id: str) -> NodeRecord | None:
        return self._cache.get(node_id)

    def get_by_path(self, schema_path: str) -> NodeRecord | None:
        nid = self._path_index.get(schema_path)
        return self._cache.get(nid) if nid else None

    def search_by_name(self, name: str) -> list[NodeRecord]:
        ids = self._name_index.get(name.lower(), [])
        return [self._cache[i] for i in ids if i in self._cache]

    def search_by_keyword(self, keyword: str, kind: str | None = None, top_k: int = 10) -> list[NodeRecord]:
        kw = keyword.lower()
        results = [
            r for r in self._cache.values()
            if kw in r.name.lower() or kw in r.description.lower() or kw in r.schema_path.lower()
        ]
        if kind:
            results = [r for r in results if r.node_kind == kind]
        return results[:top_k]

    def all_records(self) -> list[NodeRecord]:
        return list(self._cache.values())
