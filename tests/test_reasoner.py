import json
from unittest.mock import MagicMock
from causality_graph.agent.reasoner import Reasoner, ReasoningResult

SAMPLE_SUBGRAPH = {
    "nodes": [
        {"id": "kpi:dl_throughput", "name": "DL Throughput", "node_type": "kpi", "unit": "Mbps"},
        {"id": "feature:CA", "name": "Carrier Aggregation", "node_type": "feature", "gen": "both"},
        {"id": "param:maxCaBands", "name": "maxCaBands", "node_type": "parameter",
         "range_min": 1, "range_max": 4, "default_value": "1"},
    ],
    "edges": [
        {"from": "feature:CA", "to": "kpi:dl_throughput", "relation": "AFFECTS",
         "direction": "+", "magnitude": "high"},
        {"from": "feature:CA", "to": "param:maxCaBands", "relation": "CONTROLLED_BY"},
    ],
}

MOCK_RESPONSE = json.dumps({
    "answer": "To improve DL throughput, increase maxCaBands to 4 to enable full Carrier Aggregation.",
    "reasoning_chain": [
        "DL Throughput is improved by Carrier Aggregation (high positive effect)",
        "Carrier Aggregation is controlled by maxCaBands (increase to 4)",
    ],
    "source_nodes": ["feature:CA", "param:maxCaBands"],
})


def test_returns_reasoning_result():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=MOCK_RESPONSE)]
    )
    reasoner = Reasoner(client=mock_client)
    result = reasoner.answer("What parameters improve DL throughput?", SAMPLE_SUBGRAPH)
    assert isinstance(result, ReasoningResult)
    assert "maxCaBands" in result.answer


def test_reasoning_chain_populated():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text=MOCK_RESPONSE)]
    )
    reasoner = Reasoner(client=mock_client)
    result = reasoner.answer("What parameters improve DL throughput?", SAMPLE_SUBGRAPH)
    assert len(result.reasoning_chain) == 2


def test_malformed_response_returns_raw():
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Here is my answer: increase maxCaBands")]
    )
    reasoner = Reasoner(client=mock_client)
    result = reasoner.answer("What parameters improve DL throughput?", SAMPLE_SUBGRAPH)
    assert "maxCaBands" in result.answer
    assert result.reasoning_chain == []
