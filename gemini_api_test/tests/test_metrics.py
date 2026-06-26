"""Unit tests for metric math — no network. Run: python -m pytest (or unittest)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import summarize, _cumulative


def _rec(mode, turn, prompt, resp, wire):
    return {
        "mode": mode, "turn": turn,
        "prompt_tokens": prompt, "resp_tokens": resp,
        "total_tokens": prompt + resp,
        "wire_sent": wire // 2, "wire_recv": wire - wire // 2,
        "req_payload_bytes": wire // 2, "resp_payload_bytes": wire - wire // 2,
        "error": "",
    }


class TestMetrics(unittest.TestCase):
    def test_cumulative(self):
        self.assertEqual(_cumulative([1, 2, 3]), [1, 3, 6])
        self.assertEqual(_cumulative([]), [])

    def test_stateless_grows_quadratic_vs_delta_linear(self):
        # stateless: prompt tokens grow with turn (k*base); delta: constant base.
        base = 10
        records = []
        for k in range(1, 6):
            records.append(_rec("stateless", k, base * k, 5, 100 * k))
            records.append(_rec("delta", k, base, 5, 100))
        s = summarize({"records": records})
        # delta cumulative tokens linear: 5 turns * (10+5) = 75
        self.assertEqual(s["totals"]["delta_tokens"], 75)
        # stateless cumulative prompt = base*(1+2+3+4+5)=150, +resp 25 = 175
        self.assertEqual(s["totals"]["stateless_tokens"], 175)
        # ratio > 1, proving stateless costs more
        self.assertGreater(s["totals"]["token_ratio"], 1)
        # cumulative series strictly non-decreasing
        cum = s["stateless"]["cum_tokens"]
        self.assertEqual(cum, sorted(cum))

    def test_cost_estimate(self):
        records = [_rec("stateless", 1, 1000, 0, 50), _rec("delta", 1, 10, 0, 50)]
        s = summarize({"records": records})
        price = s["totals"]["price_per_token"]
        self.assertAlmostEqual(s["totals"]["stateless_cost_usd"], round(1000 * price, 6))

    def test_ratio_none_when_delta_zero(self):
        records = [_rec("stateless", 1, 1000, 0, 50)]
        s = summarize({"records": records})
        self.assertIsNone(s["totals"]["token_ratio"])


if __name__ == "__main__":
    unittest.main()
