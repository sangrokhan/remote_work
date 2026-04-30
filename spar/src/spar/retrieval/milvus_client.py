"""Milvus client wrapper — connection management + collection schemas."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MilvusConfig:
    host: str = field(default_factory=lambda: os.environ.get("MILVUS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.environ.get("MILVUS_PORT", "19530")))
    user: str = field(default_factory=lambda: os.environ.get("MILVUS_USER", ""))
    password: str = field(default_factory=lambda: os.environ.get("MILVUS_PASSWORD", ""))
    collection_prefix: str = field(
        default_factory=lambda: os.environ.get("MILVUS_COLLECTION_PREFIX", "spar_")
    )


# ---------------------------------------------------------------------------
# Collection schemas
# ---------------------------------------------------------------------------
# PRD Task 1.2: 문서 유형별 별도 인덱스 (Task 1.4 참조)
# 모든 컬렉션은 동일 스키마를 공유하되 prefix로 분리

DOC_TYPES = [
    "parameter_ref",
    "counter_ref",
    "alarm_ref",
    "feature_desc",
    "mop",
    "install_guide",
    "release_notes",
]

# Dense embedding dimension — BGE-large-en-v1.5 / E5-large-v2 기준
EMBED_DIM = 1024

HNSW_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200},
}

SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 100}}


def _build_schema(description: str = "") -> CollectionSchema:
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        # PRD Task 1.2 메타데이터
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="product", dtype=DataType.VARCHAR, max_length=16),   # LTE | NR | both
        FieldSchema(name="release", dtype=DataType.VARCHAR, max_length=16),   # v6.0, v7.1, …
        FieldSchema(name="deployment_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="mo_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="page", dtype=DataType.INT32),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65_535),
    ]
    return CollectionSchema(fields=fields, description=description, enable_dynamic_field=True)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class SparMilvusClient:
    """Thin wrapper over pymilvus for SPAR collections."""

    def __init__(self, config: MilvusConfig | None = None) -> None:
        self.cfg = config or MilvusConfig()
        self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        kwargs: dict[str, Any] = {
            "alias": "default",
            "host": self.cfg.host,
            "port": self.cfg.port,
        }
        if self.cfg.user:
            kwargs["user"] = self.cfg.user
            kwargs["password"] = self.cfg.password
        connections.connect(**kwargs)

    def close(self) -> None:
        connections.disconnect("default")

    def __enter__(self) -> "SparMilvusClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def collection_name(self, doc_type: str) -> str:
        return f"{self.cfg.collection_prefix}{doc_type}"

    def collection_exists(self, doc_type: str) -> bool:
        return utility.has_collection(self.collection_name(doc_type))

    def get_collection(self, doc_type: str) -> Collection:
        name = self.collection_name(doc_type)
        col = Collection(name=name)
        col.load()
        return col

    def create_collection(self, doc_type: str, drop_if_exists: bool = False) -> Collection:
        name = self.collection_name(doc_type)
        if utility.has_collection(name):
            if drop_if_exists:
                utility.drop_collection(name)
            else:
                col = Collection(name=name)
                col.load()
                return col

        schema = _build_schema(description=f"SPAR chunks — {doc_type}")
        col = Collection(name=name, schema=schema)
        col.create_index(field_name="embedding", index_params=HNSW_INDEX_PARAMS)
        col.load()
        return col

    def create_all_collections(self, drop_if_exists: bool = False) -> dict[str, Collection]:
        return {dt: self.create_collection(dt, drop_if_exists) for dt in DOC_TYPES}

    def list_collections(self) -> list[str]:
        return [
            name
            for name in utility.list_collections()
            if name.startswith(self.cfg.collection_prefix)
        ]

    # ------------------------------------------------------------------
    # Insert / Search
    # ------------------------------------------------------------------

    def insert(self, doc_type: str, rows: list[dict[str, Any]]) -> None:
        col = self.get_collection(doc_type)
        col.insert(rows)
        col.flush()

    def search(
        self,
        doc_type: str,
        query_vectors: list[list[float]],
        top_k: int = 10,
        output_fields: list[str] | None = None,
        expr: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        col = self.get_collection(doc_type)
        if output_fields is None:
            output_fields = ["chunk_id", "doc_type", "product", "release", "source_doc", "section", "page", "text"]

        results = col.search(
            data=query_vectors,
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k,
            output_fields=output_fields,
            expr=expr,
        )
        return [
            [{"chunk_id": hit.id, "score": hit.score, **hit.fields} for hit in batch]
            for batch in results
        ]

    def delete_by_source(self, doc_type: str, source_doc: str) -> None:
        col = self.get_collection(doc_type)
        col.delete(expr=f'source_doc == "{source_doc}"')
        col.flush()
