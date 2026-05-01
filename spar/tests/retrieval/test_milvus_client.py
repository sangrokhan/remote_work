"""Unit tests for SparMilvusClient schema and hybrid search."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

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

    def test_keywords_field_max_length(self):
        schema = _build_schema()
        kw_field = next(f for f in schema.fields if f.name == "keywords")
        assert kw_field.max_length == 128

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

    def test_section_indexing_fields_exist(self):
        schema = _build_schema()
        field_names = [f.name for f in schema.fields]
        for name in ("section_num", "section_title", "section_depth", "chunk_index", "chunk_index_in_section"):
            assert name in field_names, f"{name} missing from schema"

    def test_parent_sections_is_array_field(self):
        schema = _build_schema()
        f = next(f for f in schema.fields if f.name == "parent_sections")
        from pymilvus import DataType
        assert f.dtype == DataType.ARRAY
        assert f.element_type == DataType.VARCHAR
        assert f.max_capacity == 10


class TestHybridSearch:
    def _make_mock_hit(self, chunk_id: str, score: float) -> MagicMock:
        hit = MagicMock()
        hit.id = chunk_id
        hit.score = score
        hit.fields = {
            "text": "sample text",
            "doc_type": "parameter_ref",
            "source_doc": "doc.md",
            "section": "sec1",
            "page": 1,
            "product": "NR",
            "release": "v7",
        }
        return hit

    @patch("spar.retrieval.milvus_client.Collection")
    @patch("spar.retrieval.milvus_client.connections")
    @patch("spar.retrieval.milvus_client.utility")
    def test_hybrid_search_returns_ranked_results(self, mock_utility, mock_conn, mock_col_cls):
        mock_utility.has_collection.return_value = True
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col

        hit1 = self._make_mock_hit("c1", 0.9)
        hit2 = self._make_mock_hit("c2", 0.7)
        mock_col.hybrid_search.return_value = [[hit1, hit2]]

        from spar.retrieval.milvus_client import SparMilvusClient
        client = SparMilvusClient()
        results = client.hybrid_search(
            doc_type="parameter_ref",
            query_text="전압 임계값",
            query_vector=[0.1] * 1024,
            top_k=5,
        )

        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["score"] == 0.9
        mock_col.hybrid_search.assert_called_once()

    @patch("spar.retrieval.milvus_client.Collection")
    @patch("spar.retrieval.milvus_client.connections")
    @patch("spar.retrieval.milvus_client.utility")
    def test_hybrid_search_uses_rrf_ranker(self, mock_utility, mock_conn, mock_col_cls):
        from pymilvus import RRFRanker
        mock_utility.has_collection.return_value = True
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col
        mock_col.hybrid_search.return_value = [[]]

        from spar.retrieval.milvus_client import SparMilvusClient
        client = SparMilvusClient()
        client.hybrid_search("parameter_ref", "test query", [0.0] * 1024, top_k=3)

        call_kwargs = mock_col.hybrid_search.call_args
        assert "rerank" in call_kwargs.kwargs, "hybrid_search must be called with rerank= keyword argument"
        assert isinstance(call_kwargs.kwargs["rerank"], RRFRanker)

    @patch("spar.retrieval.milvus_client.Collection")
    @patch("spar.retrieval.milvus_client.connections")
    @patch("spar.retrieval.milvus_client.utility")
    def test_hybrid_search_passes_two_ann_requests(self, mock_utility, mock_conn, mock_col_cls):
        from pymilvus import AnnSearchRequest
        mock_utility.has_collection.return_value = True
        mock_col = MagicMock()
        mock_col_cls.return_value = mock_col
        mock_col.hybrid_search.return_value = [[]]

        from spar.retrieval.milvus_client import SparMilvusClient
        client = SparMilvusClient()
        client.hybrid_search("parameter_ref", "test query", [0.0] * 1024)

        call_kwargs = mock_col.hybrid_search.call_args
        assert "reqs" in call_kwargs.kwargs, "hybrid_search must be called with reqs= keyword argument"
        reqs = call_kwargs.kwargs["reqs"]
        assert len(reqs) == 2
        fields = [r.anns_field for r in reqs]
        assert "embedding" in fields
        assert "sparse_vec" in fields
