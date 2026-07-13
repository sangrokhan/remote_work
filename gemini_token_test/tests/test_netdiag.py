"""Name the 403.

Private Google Access rewrites *.googleapis.com to a VIP. The restricted VIP
(199.36.153.4/30) only serves APIs that support VPC Service Controls --
aiplatform does, generativelanguage does not -- so from inside such a VPC the
Developer API is unreachable by construction, and it says so with a 403 that
looks exactly like a bad key. That misdiagnosis costs hours; the app should just
say which one it is.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import netdiag


def test_restricted_vip_range_is_named():
    assert netdiag.classify_ip("199.36.153.4") == "restricted"
    assert netdiag.classify_ip("199.36.153.7") == "restricted"


def test_private_vip_range_is_named():
    assert netdiag.classify_ip("199.36.153.8") == "private"
    assert netdiag.classify_ip("199.36.153.11") == "private"


def test_ordinary_addresses_are_public():
    assert netdiag.classify_ip("142.250.207.10") == "public"


def test_restricted_vip_diagnosis_explains_the_403():
    d = netdiag.diagnose("generativelanguage.googleapis.com", ips=["199.36.153.4"])
    assert d["vip"] == "restricted"
    assert d["reachable"] is False
    assert "restricted VIP" in d["explanation"]
    # The fix has to be actionable, not just a name for the failure.
    assert "egress" in d["explanation"].lower()


def test_public_route_is_reported_reachable():
    d = netdiag.diagnose("generativelanguage.googleapis.com", ips=["142.250.207.10"])
    assert d["vip"] == "public"
    assert d["reachable"] is True


def test_private_vip_still_cannot_serve_the_developer_api():
    # private.googleapis.com carries most APIs, but the Developer API is not
    # among them: only Vertex is reachable that way.
    d = netdiag.diagnose("generativelanguage.googleapis.com", ips=["199.36.153.8"])
    assert d["vip"] == "private"
    assert d["reachable"] is False


def test_a_restricted_vip_403_body_is_recognised():
    body = ('{"error":{"code":403,"message":"Access is not available on Google '
            'restricted VIPs.","status":"PERMISSION_DENIED"}}')
    assert netdiag.is_vip_block(body) is True


def test_an_ordinary_403_is_not_mistaken_for_a_vip_block():
    body = '{"error":{"code":403,"message":"API key not valid."}}'
    assert netdiag.is_vip_block(body) is False


def test_unresolvable_host_does_not_raise():
    d = netdiag.diagnose("generativelanguage.googleapis.com", ips=[])
    assert d["vip"] == "unknown"
    assert d["reachable"] is None
