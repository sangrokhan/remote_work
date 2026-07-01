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

    def test_parse_tcpdump_stats_full(self):
        text = (
            "130 packets captured\n"
            "137 packets received by filter\n"
            "7 packets dropped by kernel\n"
            "0 packets dropped by interface\n"
        )
        self.assertEqual(capture._parse_tcpdump_stats(text), {
            "captured": 130,
            "received_by_filter": 137,
            "dropped_by_kernel": 7,
            "dropped_by_interface": 0,
        })

    def test_parse_tcpdump_stats_partial_and_empty(self):
        # Only some lines present (e.g. no "dropped by interface" on some platforms).
        self.assertEqual(
            capture._parse_tcpdump_stats("42 packets captured\n"),
            {"captured": 42},
        )
        self.assertEqual(capture._parse_tcpdump_stats(""), {})

    def test_snaplen_default_and_env(self):
        # Default snaplen is the header-only 100 bytes.
        c = capture.Capture("2026-06-29T00:00:00", "stateless")
        self.assertEqual(c.snaplen, capture.PCAP_SNAPLEN)
        self.assertEqual(capture.PCAP_SNAPLEN, 100)

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
