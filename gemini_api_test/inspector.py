"""Generic HTTP endpoint inspector.

Fire a request at any URL (a local MCP server, an A2A agent, the Gemini REST
endpoint, ...) and capture, app-layer (no packet capture / NET_RAW needed):

  - request: method, url, headers, optional body, body size
  - response: status, headers, optional body (capped), body size
  - real wire bytes (TLS socket counter), payload sizes, elapsed ms
  - protocol_hints: detected MCP / A2A / JSON-RPC / SSE markers in the headers

This is where protocol-specific headers show up in plaintext — a TLS pcap can't,
because the wire is encrypted.

SSRF guard: by default refuses targets that resolve to private / loopback /
reserved IPs. The link-local range (169.254.0.0/16, incl. the GCP metadata server
169.254.169.254) is ALWAYS refused. Set allow_private=True (UI toggle) to permit
localhost / RFC-1918 targets such as a local MCP server.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import secrets
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from gemini_client import _session, _active_counter

TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "data/transcripts"))
_SAFE_NAME = re.compile(r"^inspect_[0-9T\-]+_[0-9a-f]{16}\.json$")
MAX_BODY = int(os.environ.get("INSPECT_MAX_BODY", str(1 << 20)))  # 1 MiB cap

# Header-name / value substrings that hint at a protocol.
_HINTS = {
    "mcp": "MCP (Model Context Protocol)",
    "a2a": "A2A (Agent2Agent)",
    "jsonrpc": "JSON-RPC",
    "json-rpc": "JSON-RPC",
    "text/event-stream": "SSE (server-sent events)",
    "agent-card": "A2A agent card",
}


def _resolve_ips(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None)
    return sorted({i[4][0] for i in infos})


def _is_blocked(ip: str, allow_private: bool) -> bool:
    a = ipaddress.ip_address(ip)
    # Always block link-local (includes 169.254.169.254 metadata) + multicast.
    if a.is_link_local or a.is_multicast or a.is_unspecified:
        return True
    if allow_private:
        return False
    return a.is_private or a.is_loopback or a.is_reserved


def ssrf_check(url: str, allow_private: bool) -> tuple[bool, str]:
    """Returns (ok, reason_if_blocked)."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False, f"scheme '{p.scheme}' not allowed (http/https only)"
    if not p.hostname:
        return False, "no host in URL"
    try:
        ips = _resolve_ips(p.hostname)
    except Exception as exc:
        return False, f"dns_failed: {exc}"
    if not ips:
        return False, "host did not resolve"
    for ip in ips:
        if _is_blocked(ip, allow_private):
            return False, f"target {ip} blocked (private/link-local; enable allow_private for localhost)"
    return True, ""


def _protocol_hints(req_headers: dict, resp_headers: dict) -> list[str]:
    found = set()
    blob = " ".join(
        f"{k}:{v}" for k, v in
        list(req_headers.items()) + list(resp_headers.items())
    ).lower()
    for needle, label in _HINTS.items():
        if needle in blob:
            found.add(label)
    return sorted(found)


def _parse_headers(raw: str) -> dict:
    """Accept JSON object or 'Key: value' lines."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    headers = {}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers


def _safe_name(timestamp: str) -> str:
    return f"inspect_{timestamp.replace(':', '-')}_{secrets.token_hex(8)}.json"


def safe_transcript_path(name: str) -> Path | None:
    if not _SAFE_NAME.match(name):
        return None
    p = (TRANSCRIPT_DIR / name).resolve()
    if p.parent != TRANSCRIPT_DIR.resolve():
        return None
    return p if p.exists() else None


def inspect(method: str, url: str, headers_raw: str, body: str,
            include_bodies: bool, allow_private: bool, timestamp: str) -> dict:
    """Perform the request and build the inspection record. Never raises."""
    method = (method or "GET").upper()
    ok, reason = ssrf_check(url, allow_private)
    if not ok:
        return {"ok": False, "error": f"blocked: {reason}", "url": url}

    req_headers = _parse_headers(headers_raw)
    data = body.encode("utf-8") if (body and method in ("POST", "PUT", "PATCH")) else None
    req_body_bytes = len(data) if data else 0

    _active_counter["counter"] = None
    t0 = time.perf_counter()
    try:
        resp = _session().request(
            method, url, headers=req_headers, data=data, timeout=60,
            allow_redirects=False,  # do not follow -> avoids redirect-based SSRF
        )
    except Exception as exc:
        return {"ok": False, "error": f"request_failed: {exc}", "url": url}
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    counter = _active_counter["counter"]
    resp_headers = dict(resp.headers)
    content = resp.content or b""
    record = {
        "ok": True,
        "method": method,
        "url": url,
        "status": resp.status_code,
        "elapsed_ms": elapsed_ms,
        "request_headers": req_headers,
        "response_headers": resp_headers,
        "req_body_bytes": req_body_bytes,
        "resp_body_bytes": len(content),
        "wire_sent": counter.sent if counter else req_body_bytes,
        "wire_recv": counter.recv if counter else len(content),
        "protocol_hints": _protocol_hints(req_headers, resp_headers),
    }
    if include_bodies:
        record["request_body"] = body or ""
        text = content[:MAX_BODY].decode("utf-8", errors="replace")
        record["response_body"] = text
        record["response_body_truncated"] = len(content) > MAX_BODY
    return record


def save_transcript(timestamp: str, record: dict) -> str | None:
    if not record.get("ok"):
        return None
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_name(timestamp)
    (TRANSCRIPT_DIR / name).write_text(json.dumps(record, indent=2))
    return name
