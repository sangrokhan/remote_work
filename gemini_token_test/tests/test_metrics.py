"""Unit tests for single-mode metric math — no network."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import summarize, _cumulative


def _rec(turn, prompt, resp, wire):
    return {
        "mode": "stateless", "turn": turn,
        "prompt_tokens": prompt, "resp_tokens": resp,
        "total_tokens": prompt + resp,
        "wire_sent": wire // 2, "wire_recv": wire - wire // 2,
        "req_payload_bytes": wire // 2, "resp_payload_bytes": wire - wire // 2,
        "error": "",
    }


def _exp(mode, records):
    return {"params": {"mode": mode}, "records": records}


class TestMetrics(unittest.TestCase):
    def test_cumulative(self):
        self.assertEqual(_cumulative([1, 2, 3]), [1, 3, 6])
        self.assertEqual(_cumulative([]), [])

    def test_single_mode_series_and_totals(self):
        recs = [_rec(k, 10 * k, 5, 100 * k) for k in range(1, 6)]
        s = summarize(_exp("stateless", recs))
        self.assertEqual(s["mode"], "stateless")
        self.assertEqual(s["totals"]["mode"], "stateless")
        # cum tokens = sum(10k+5) for k=1..5 = 150 + 25 = 175
        self.assertEqual(s["totals"]["tokens"], 175)
        self.assertEqual(s["series"]["cum_tokens"], [15, 40, 75, 120, 175])
        cw = s["series"]["cum_wire_bytes"]
        self.assertEqual(cw, sorted(cw))

    def test_mode_passthrough(self):
        s = summarize(_exp("stateful", [_rec(1, 10, 5, 100)]))
        self.assertEqual(s["mode"], "stateful")

    def test_cost_estimate(self):
        s = summarize(_exp("stateless", [_rec(1, 1000, 0, 50)]))
        price = s["totals"]["price_per_token"]
        self.assertAlmostEqual(s["totals"]["cost_usd"], round(1000 * price, 6))

    def test_empty_records(self):
        s = summarize(_exp("stateless", []))
        self.assertEqual(s["totals"]["tokens"], 0)
        self.assertEqual(s["series"]["turns"], [])


if __name__ == "__main__":
    unittest.main()
