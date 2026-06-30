"""Unit tests for capture helpers — no tcpdump / root / network needed."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import capture


class TestCapture(unittest.TestCase):
    def test_filter_with_ips(self):
        f = capture._filter_expr(["1.2.3.4", "5.6.7.8"])
        self.assertEqual(f, "tcp port 443 and (host 1.2.3.4 or host 5.6.7.8)")

    def test_filter_no_ips_falls_back(self):
        self.assertEqual(capture._filter_expr([]), "tcp port 443")

    def test_safe_name_accepts_valid(self):
        # Real generated form: mode + timestamp + 16-hex token.
        good = "capture_stateless_2026-06-29T00-00-00_0123456789abcdef.pcap"
        self.assertIsNotNone(capture._SAFE_NAME.match(good))
        self.assertIsNotNone(capture._SAFE_NAME.match(
            "capture_stateful_2026-06-29T00-00-00_0123456789abcdef.pcap"))

    def test_safe_name_rejects_modeless(self):
        # Old form without the mode segment must no longer validate.
        self.assertIsNone(capture._SAFE_NAME.match(
            "capture_2026-06-29T00-00-00_0123456789abcdef.pcap"))

    def test_safe_path_rejects_traversal(self):
        for bad in ["../etc/passwd", "capture_x.pcap/../y", "run.json",
                    "capture_; rm -rf.pcap"]:
            self.assertIsNone(capture.safe_pcap_path(bad))

    def test_available_reports_when_disabled(self):
        os.environ["PCAP_DISABLE"] = "1"
        try:
            ok, reason = capture.available()
            self.assertFalse(ok)
            self.assertIn("disabled", reason)
        finally:
            del os.environ["PCAP_DISABLE"]


if __name__ == "__main__":
    unittest.main()
