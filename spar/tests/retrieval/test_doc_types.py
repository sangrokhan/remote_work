from spar.retrieval.milvus_client import DOC_TYPES


def test_spec_doc_type_present():
    """3GPP TSpec-LLM markdown ingest 위해 spec 유형 필요."""
    assert "spec" in DOC_TYPES


def test_doc_types_unique():
    assert len(DOC_TYPES) == len(set(DOC_TYPES))
