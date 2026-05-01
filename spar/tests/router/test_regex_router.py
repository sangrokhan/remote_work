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


def test_ts_spec_dotted(router):
    result = router.route("TS 29.502 session management overview")
    assert result is not None
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.layer == "regex"
    assert result.confidence == 1.0
    assert result.entities.get("spec_number") == "29.502"


def test_ts_spec_dotted_with_3gpp_prefix(router):
    result = router.route("3GPP TS 38.300 NR architecture explained")
    assert result is not None
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.entities.get("spec_number") == "38.300"


def test_ts_spec_no_dot(router):
    result = router.route("TS29502 what is SMF?")
    assert result is not None
    assert result.route == Route.DEFINITION_EXPLAIN
    assert result.entities.get("spec_number") == "29.502"


def test_ts_spec_space_separated(router):
    result = router.route("TS 29 502 defines session management")
    assert result is not None
    assert result.entities.get("spec_number") == "29.502"


def test_ts_spec_no_conflict_with_alarm(router):
    """TS 패턴과 alarm 패턴이 같은 쿼리에 있을 때 alarm이 우선."""
    result = router.route("ALM-4012 related to TS 29.502")
    assert result is not None
    assert result.route == Route.STRUCTURED_LOOKUP
    assert result.entities.get("alarm_code") == "ALM-4012"
