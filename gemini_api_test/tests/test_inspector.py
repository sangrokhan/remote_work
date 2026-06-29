"""Unit tests for the endpoint inspector — no real network calls."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspector


class TestSSRFGuard(unittest.TestCase):
    def test_scheme_rejected(self):
        ok, reason = inspector.ssrf_check("file:///etc/passwd", allow_private=True)
        self.assertFalse(ok)
        self.assertIn("scheme", reason)

    def test_metadata_ip_always_blocked(self):
        # Link-local / metadata blocked even with allow_private.
        self.assertTrue(inspector._is_blocked("169.254.169.254", allow_private=True))

    def test_private_blocked_by_default(self):
        self.assertTrue(inspector._is_blocked("10.0.0.5", allow_private=False))
        self.assertTrue(inspector._is_blocked("127.0.0.1", allow_private=False))

    def test_private_allowed_when_opted_in(self):
        self.assertFalse(inspector._is_blocked("127.0.0.1", allow_private=True))
        self.assertFalse(inspector._is_blocked("192.168.1.10", allow_private=True))

    def test_public_ip_allowed(self):
        self.assertFalse(inspector._is_blocked("142.250.0.0", allow_private=False))


class TestHelpers(unittest.TestCase):
    def test_parse_headers_json(self):
        h = inspector._parse_headers('{"A": "1", "B": "2"}')
        self.assertEqual(h, {"A": "1", "B": "2"})

    def test_parse_headers_lines(self):
        h = inspector._parse_headers("X-Foo: bar\nMcp-Session-Id: abc")
        self.assertEqual(h["X-Foo"], "bar")
        self.assertEqual(h["Mcp-Session-Id"], "abc")

    def test_protocol_hints_detect_mcp_and_sse(self):
        hints = inspector._protocol_hints(
            {"Mcp-Protocol-Version": "2025-06-18"},
            {"Content-Type": "text/event-stream"},
        )
        self.assertIn("MCP (Model Context Protocol)", hints)
        self.assertIn("SSE (server-sent events)", hints)

    def test_protocol_hints_detect_a2a_jsonrpc(self):
        hints = inspector._protocol_hints(
            {}, {"Content-Type": "application/json", "X-A2A-Agent": "card"})
        self.assertIn("A2A (Agent2Agent)", hints)

    def test_safe_transcript_rejects_traversal(self):
        for bad in ["../x.json", "inspect_x.json", "run.json", "inspect_;rm.json"]:
            self.assertIsNone(inspector.safe_transcript_path(bad))

    def test_safe_name_matches_generated(self):
        name = inspector._safe_name("2026-06-29T00:00:00")
        self.assertIsNotNone(inspector._SAFE_NAME.match(name))


if __name__ == "__main__":
    unittest.main()
