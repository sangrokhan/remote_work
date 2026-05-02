"""Verify routing layer uses AlarmIndex when alarm_code entity is present."""
from spar.retrieval import alarm_index as ai_mod
from spar.retrieval import routing


def setup_function(_):
    ai_mod._SINGLETON = None


def test_alarm_code_lookup_populates_structured_record():
    result = routing.resolve_alarm_entity({"alarm_code": "ALM-1003"})
    assert result is not None
    assert result["alarm_id"] == "ALM-1003"
    assert result["alarm_name"] == "Cell Down"
    assert "ALM-1003" in result["keywords"]
    assert "Cell Down" in result["keywords"]


def test_alarm_code_unknown_returns_none():
    result = routing.resolve_alarm_entity({"alarm_code": "ALM-9999"})
    assert result is None


def test_no_alarm_code_returns_none():
    result = routing.resolve_alarm_entity({})
    assert result is None
