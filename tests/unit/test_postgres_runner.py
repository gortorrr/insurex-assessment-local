import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from insurex.config import Settings
from scripts.postgres_integration import RunOwnership, build_cleanup_plan, check_kpi_lead_session, cleanup_allowed


class PostgresRunnerContractTests(unittest.TestCase):
    def test_cleanup_uses_exact_ids_and_not_sql_wildcards(self):
        ownership = RunOwnership.create("run_%_previous")
        ownership.session_ids = ["session-%-owned"]
        ownership.message_ids = [7]
        ownership.lead_ids = [8]
        ownership.performance_ids = [9]
        ownership.evaluation_ids = [10]
        ownership.contract_ids = [11]
        plan = build_cleanup_plan(ownership)
        sql = " ".join(item[1] for item in plan).upper()
        self.assertNotIn("LIKE", sql)
        self.assertTrue(all("?" in statement for _, statement, _ in plan))
        self.assertTrue(any("run_%_previous" in str(value) for _, _, value in plan))
        self.assertIn("session-%-owned", [value for _, _, value in plan])

    def test_each_run_has_disjoint_ownership_ids(self):
        first = RunOwnership.create("run-a")
        second = RunOwnership.create("run-b")
        self.assertNotEqual(first.agent_id, second.agent_id)
        self.assertNotEqual(first.rule_id, second.rule_id)
        self.assertNotEqual(first.checkpoint_thread_id, second.checkpoint_thread_id)

    def test_cleanup_is_closed_when_preflight_or_mutation_did_not_pass(self):
        self.assertFalse(cleanup_allowed(preflight_passed=False, mutation_started=False))
        self.assertFalse(cleanup_allowed(preflight_passed=False, mutation_started=True))
        self.assertFalse(cleanup_allowed(preflight_passed=True, mutation_started=False))
        self.assertTrue(cleanup_allowed(preflight_passed=True, mutation_started=True))

    def test_business_flow_uses_typed_kpi_rule_on_temporary_sqlite_adapter(self):
        with TemporaryDirectory() as directory:
            settings = Settings(
                database_backend="local",
                business_db_path=Path(directory) / "business.sqlite",
            )
            ownership = RunOwnership.create("offline-kpi-flow")
            result = check_kpi_lead_session(settings, ownership)
            self.assertEqual(result["kpi"]["evaluations"], 3)
            self.assertTrue(result["kpi"]["all_results_pass"])
            self.assertTrue(result["lead"]["idempotency"])
            self.assertTrue(result["session"]["reconnect"])
            self.assertEqual(len(ownership.performance_ids), 3)
            self.assertEqual(len(ownership.session_ids), 2)


if __name__ == "__main__":
    unittest.main()
