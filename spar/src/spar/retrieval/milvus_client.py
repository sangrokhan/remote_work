"""Milvus client wrapper — connection management + collection schemas."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
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
    "spec",  # 3GPP TSpec-LLM 등 외부 표준 문서
]

# Dense embedding dimension — BGE-large-en-v1.5 / E5-large-v2 기준
EMBED_DIM = 1024

HNSW_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200},
}

SEARCH_PARAMS = {"metric_type": "COSINE", "params": {"ef": 100}}

SPARSE_INDEX_PARAMS = {
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "BM25",
    "params": {"drop_ratio_build": 0.0},
}

SPARSE_SEARCH_PARAMS = {"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}}


def _build_schema(description: str = "") -> CollectionSchema:
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="product", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="release", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="deployment_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="mo_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_doc", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page", dtype=DataType.INT32),
        # 3GPP section indexing fields
        FieldSchema(name="section_num", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="section_depth", dtype=DataType.INT32),
        FieldSchema(name="chunk_index", dtype=DataType.INT32),
        FieldSchema(name="chunk_index_in_section", dtype=DataType.INT32),
        FieldSchema(
            name="parent_sections",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=10,
            max_length=64,
        ),
        FieldSchema(
            name="keywords",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=50,
            max_length=128,
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65_535,
            enable_analyzer=True,
        ),
        FieldSchema(name="sparse_vec", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    schema = CollectionSchema(fields=fields, description=description, enable_dynamic_field=True)

    bm25_fn = Function(
        name="bm25",
        input_field_names=["text"],
        output_field_names=["sparse_vec"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_fn)
    return schema


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
        col.create_index(field_name="sparse_vec", index_params=SPARSE_INDEX_PARAMS)
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
            output_fields = [
                "chunk_id", "doc_type", "product", "release", "source_doc",
                "section", "section_num", "section_title", "section_depth",
                "parent_sections", "chunk_index", "chunk_index_in_section",
                "page", "text",
            ]

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

    def hybrid_search(
        self,
        doc_type: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        output_fields: list[str] | None = None,
        expr: str | None = None,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        col = self.get_collection(doc_type)
        if output_fields is None:
            output_fields = [
                "chunk_id", "doc_type", "product", "release", "source_doc",
                "section", "section_num", "section_title", "section_depth",
                "parent_sections", "chunk_index", "chunk_index_in_section",
                "page", "text",
            ]

        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="embedding",
            param=SEARCH_PARAMS,
            limit=top_k * 2,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse_vec",
            param=SPARSE_SEARCH_PARAMS,
            limit=top_k * 2,
            expr=expr,
        )

        results = col.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=rrf_k),
            limit=top_k,
            output_fields=output_fields,
        )
        return [
            {"chunk_id": hit.id, "score": hit.score, **hit.fields}
            for hit in results[0]
        ]

    def delete_by_source(self, doc_type: str, source_doc: str) -> None:
        col = self.get_collection(doc_type)
        col.delete(expr=f'source_doc == "{source_doc}"')
        col.flush()
