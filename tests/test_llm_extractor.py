import json
from unittest.mock import MagicMock
from causality_graph.extraction.llm_extractor import LLMExtractor, ExtractionResult
from causality_graph.extraction.md_parser import ParsedFeature


SAMPLE_PARSED = ParsedFeature(
    feature_id="feature:CA",
    name="Carrier Aggregation (CA)",
    gen="both",
    category="rrm",
    description="Combines multiple carriers to increase throughput.",
    kpi_impacts=[{"kpi_id": "kpi:dl_throughput", "kpi_name": "DL Throughput",
                   "direction": "+", "magnitude": "high", "condition": ""}],
    controlling_params=[{"param_id": "param:maxCaBands", "effect": "more bands"}],
    dependencies=[],
)

MOCK_LLM_RESPONSE = json.dumps({
    "triples": [
        {
            "from": "feature:CA",
            "to": "kpi:dl_throughput",
            "relation": "AFFECTS",
            "direction": "+",
            "magnitude": "high",
            "condition": "",
            "confidence": 0.95
        },
        {
            "from": "feature:CA",
            "to": "param:maxCaBands",
            "relation": "CONTROLLED_BY",
            "direction": None,
            "magnitude": None,
            "condition": "",
            "confidence": 0.90
        }
    ]
})


def test_returns_extraction_result():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=MOCK_LLM_RESPONSE)]
    )
    extractor = LLMExtractor(client=mock_client)
    result = extractor.extract(SAMPLE_PARSED)
    assert isinstance(result, ExtractionResult)
    assert result.source_feature_id == "feature:CA"
    assert len(result.triples) == 2


def test_triple_fields():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=MOCK_LLM_RESPONSE)]
    )
    extractor = LLMExtractor(client=mock_client)
    result = extractor.extract(SAMPLE_PARSED)
    affects = next(t for t in result.triples if t["relation"] == "AFFECTS")
    assert affects["from"] == "feature:CA"
    assert affects["to"] == "kpi:dl_throughput"
    assert affects["confidence"] == 0.95


def test_malformed_llm_response_returns_empty():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="not valid json")]
    )
    extractor = LLMExtractor(client=mock_client)
    result = extractor.extract(SAMPLE_PARSED)
    assert result.triples == []
    assert len(result.parse_errors) > 0
