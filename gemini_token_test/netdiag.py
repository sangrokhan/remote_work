"""Why the 403 happened: the key, or the route?

Private Google Access rewrites *.googleapis.com onto one of two VIP ranges. The
restricted VIP only carries APIs that support VPC Service Controls. Vertex
(aiplatform) does. The Gemini Developer API (generativelanguage) does not -- so
from inside a VPC that forces the restricted VIP, the Developer API is unreachable
by construction, and the 403 it returns reads exactly like a rejected API key.

That ambiguity is the whole point of this module: resolve where the host actually
points, and say which failure it is.
"""

from __future__ import annotations

import ipaddress
import socket

# https://cloud.google.com/vpc/docs/configure-private-google-access
RESTRICTED_VIP = ipaddress.ip_network("199.36.153.4/30")   # VPC-SC-supported APIs only
PRIVATE_VIP = ipaddress.ip_network("199.36.153.8/30")      # most APIs, but not this one

# Neither VIP carries the Developer API: the restricted one excludes it (not a
# VPC-SC service), and the private one is a Vertex/most-APIs path that does not
# route generativelanguage either.
_VIP_EXPLANATION = {
    "restricted": (
        "DNS points {host} at the restricted VIP ({ips}), which only carries APIs that "
        "support VPC Service Controls. The Gemini Developer API is not one of them, so "
        "every call is refused with a 403 that looks like a bad API key but isn't. "
        "Fix the route, not the key: set the service's egress to 'public IPs only' "
        "instead of routing all traffic through the VPC, or run the experiment from a "
        "host that reaches the public internet."
    ),
    "private": (
        "DNS points {host} at the private VIP ({ips}). It carries most Google APIs but "
        "not the Gemini Developer API, so calls are refused. Fix the route, not the key: "
        "set the service's egress to 'public IPs only', or run from a host with public "
        "internet access."
    ),
}


def classify_ip(ip: str) -> str:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "unknown"
    if addr in RESTRICTED_VIP:
        return "restricted"
    if addr in PRIVATE_VIP:
        return "private"
    return "public"


def resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        return sorted({i[4][0] for i in infos})
    except Exception:
        return []


def diagnose(host: str, ips: list[str] | None = None) -> dict:
    """Where does `host` actually resolve, and can the Developer API be reached
    from here? reachable is None when DNS gave us nothing to judge."""
    ips = resolve(host) if ips is None else list(ips)
    kinds = {classify_ip(ip) for ip in ips}
    if not ips:
        return {"host": host, "ips": [], "vip": "unknown", "reachable": None,
                "explanation": f"Could not resolve {host}."}

    for kind in ("restricted", "private"):
        if kind in kinds:
            return {
                "host": host, "ips": ips, "vip": kind, "reachable": False,
                "explanation": _VIP_EXPLANATION[kind].format(host=host, ips=", ".join(ips)),
            }
    return {"host": host, "ips": ips, "vip": "public", "reachable": True,
            "explanation": f"{host} resolves to public addresses ({', '.join(ips)})."}


def is_vip_block(body: str) -> bool:
    """Whether a 403 body is the restricted-VIP refusal rather than an auth failure.
    Matching on the phrase, not the exact JSON, since the wrapper varies."""
    return "restricted vip" in (body or "").lower()
