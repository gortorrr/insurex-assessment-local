import unittest

import pandas as pd

from insurex.analytics import acceptance_by_group, acceptance_metrics, response_filter_view


class DashboardAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "label": [0, 1, 2],
                "customer_segment": ["A", "A", "B"],
                "campaign_month": ["Jan", "Jan", "Jan"],
            }
        )

    def test_response_filter_does_not_change_base_acceptance_denominator(self):
        filtered = response_filter_view(self.frame, [1])
        self.assertEqual(len(filtered), 1)
        metrics = acceptance_metrics(self.frame)
        self.assertEqual(metrics["valid_label_records"], 3)
        self.assertEqual(metrics["accepted_records"], 2)
        self.assertAlmostEqual(metrics["rate"], 2 / 3)
        segment = acceptance_by_group(self.frame, "customer_segment")
        self.assertAlmostEqual(float(segment.loc[segment["customer_segment"] == "A", "acceptance_rate"].iloc[0]), 0.5)

    def test_empty_response_filter_has_no_rate(self):
        filtered = response_filter_view(self.frame, [99])
        self.assertTrue(filtered.empty)
        self.assertIsNone(acceptance_metrics(filtered)["rate"])


if __name__ == "__main__":
    unittest.main()
