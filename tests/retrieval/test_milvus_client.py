"""Unit tests for SparMilvusClient schema and hybrid search."""
from __future__ import annotations

import pytest

from spar.retrieval.milvus_client import _build_schema, SPARSE_INDEX_PARAMS


class TestSchema:
    def test_sparse_vec_field_exists(self):
        schema = _build_schema()
        field_names = [f.name for f in schema.fields]
        assert "sparse_vec" in field_names

    def test_text_field_has_analyzer_enabled(self):
        schema = _build_schema()
        text_field = next(f for f in schema.fields if f.name == "text")
        assert text_field.params.get("enable_analyzer") is True

    def test_bm25_function_registered(self):
        schema = _build_schema()
        fn_names = [fn.name for fn in schema.functions]
        assert "bm25" in fn_names

    def test_bm25_function_maps_text_to_sparse(self):
        schema = _build_schema()
        bm25_fn = next(fn for fn in schema.functions if fn.name == "bm25")
        assert bm25_fn.input_field_names == ["text"]
        assert bm25_fn.output_field_names == ["sparse_vec"]

    def test_sparse_index_params_values(self):
        assert SPARSE_INDEX_PARAMS["metric_type"] == "BM25"
        assert SPARSE_INDEX_PARAMS["index_type"] == "SPARSE_INVERTED_INDEX"

    def test_dense_vec_field_exists(self):
        schema = _build_schema()
        field_names = [f.name for f in schema.fields]
        assert "embedding" in field_names  # dense vector field
