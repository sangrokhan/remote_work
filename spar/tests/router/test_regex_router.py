import pytest
from spar.router.regex_router import RegexRouter
from spar.router.schemas import Route


@pytest.fixture
def router():
    return RegexRouter()


def test_alarm_code_pattern(router):
    result = router.route("What does ALM-4012 mean?")
    assert result is not None
    assert result.route == Route.STRUCTURED_LOOKUP
    assert result.confidence == 1.0
    assert result.entities.get("alarm_code") == "ALM-4012"


def test_alarm_keyword_pattern(router):
    result = router.route("alarm 1234 is triggered after HO")
    assert result is not None
    assert result.route == Route.STRUCTURED_LOOKUP


def test_no_match_returns_none(router):
    result = router.route("How does carrier aggregation work?")
    assert result is None


def test_mo_name_pattern(router):
    result = router.route("What is NRCellDU maxTxPower range?")
    assert result is not None
    assert result.route == Route.STRUCTURED_LOOKUP
    assert "NRCellDU" in result.entities.get("mo_name", "")
