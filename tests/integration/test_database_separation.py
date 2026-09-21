import tempfile
import unittest
from pathlib import Path

from insurex.auth import register, setup
from insurex.db import connect_analytics_local, connect_assistant_local


class DatabaseSeparationTests(unittest.TestCase):
    def test_local_test_cases_have_disjoint_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            assistant = connect_assistant_local(Path(directory) / "assistant.sqlite")
            analytics = connect_analytics_local(Path(directory) / "analytics.sqlite")
            try:
                setup(assistant)
                register(assistant, "customer", "Customer-2026!")
                assistant_tables = {
                    row["name"] for row in assistant.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                analytics_tables = {
                    row["name"] for row in analytics.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                self.assertTrue({"sessions", "messages", "leads", "handoff_cases", "user_accounts"} <= assistant_tables)
                self.assertFalse({"agents", "monthly_agent_performance", "monthly_kpi_evaluations"} & assistant_tables)
                self.assertTrue({"agents", "monthly_agent_performance", "monthly_kpi_evaluations"} <= analytics_tables)
                self.assertFalse({"sessions", "messages", "leads", "user_accounts"} & analytics_tables)
            finally:
                assistant.close()
                analytics.close()


if __name__ == "__main__":
    unittest.main()
