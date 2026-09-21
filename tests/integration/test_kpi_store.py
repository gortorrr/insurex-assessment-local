from datetime import date
from pathlib import Path
import sqlite3
import tempfile
import unittest

from insurex.kpi import KpiRule, MonthlyPerformance, synthetic_fixture
from insurex.kpi_store import connect, persist_agent_series


class KpiStoreTests(unittest.TestCase):
    def test_schema_rejects_invalid_month_and_incomplete_complete_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "constraints.sqlite")
            try:
                connection.execute(
                    "INSERT INTO agents(agent_id, agent_name, joined_on, initial_contract_type) VALUES (?, ?, ?, ?)",
                    ("A", "Agent A", "2026-01-01", "salary"),
                )
                invalid_rows = [
                    ("A", "2026-01-15", 1_600_000, 6, "COMPLETE", "synthetic", 1, 1, "bad-month"),
                    ("A", "2026-02-01", None, 6, "COMPLETE", "synthetic", 1, 1, "missing-premium"),
                    ("A", "2026-03-01", -1, 6, "COMPLETE", "synthetic", 1, 1, "negative-premium"),
                ]
                for row in invalid_rows:
                    with self.subTest(month=row[1], digest=row[-1]), self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(
                            """
                            INSERT INTO monthly_agent_performance(
                                agent_id, month_start, total_premium_satang, new_policy_count,
                                data_status, source_kind, month_closed, input_revision, input_hash
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            row,
                        )
            finally:
                connection.close()

    def test_rerun_is_idempotent_and_keeps_audit_revision_on_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "kpi.sqlite")
            try:
                agents, rules, performances = synthetic_fixture()
                agent = agents[0]
                rows = [row for row in performances if row.agent_id == agent["agent_id"]]

                first_ids, first_events = persist_agent_series(
                    connection,
                    agent_id=agent["agent_id"],
                    agent_name=agent["agent_name"],
                    joined_on=date(2026, 1, 1),
                    initial_contract_type=agent["initial_contract_type"],
                    performances=rows,
                    rules=rules,
                )
                second_ids, second_events = persist_agent_series(
                    connection,
                    agent_id=agent["agent_id"],
                    agent_name=agent["agent_name"],
                    joined_on=date(2026, 1, 1),
                    initial_contract_type=agent["initial_contract_type"],
                    performances=rows,
                    rules=rules,
                )
                self.assertEqual(first_ids, second_ids)
                self.assertEqual(first_events, second_events)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM monthly_agent_performance").fetchone()[0], len(rows))
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM contract_history WHERE superseded_at IS NULL").fetchone()[0], 2)

                corrected = list(rows)
                corrected[2] = type(corrected[2])(
                    agent_id=corrected[2].agent_id,
                    month_start=corrected[2].month_start,
                    total_premium_satang=1500001,
                    new_policy_count=6,
                    data_status=corrected[2].data_status,
                    source_kind=corrected[2].source_kind,
                )
                persist_agent_series(
                    connection,
                    agent_id=agent["agent_id"],
                    agent_name=agent["agent_name"],
                    joined_on=date(2026, 1, 1),
                    initial_contract_type=agent["initial_contract_type"],
                    performances=corrected,
                    rules=rules,
                )
                self.assertGreater(
                    connection.execute("SELECT COUNT(*) FROM monthly_kpi_evaluations").fetchone()[0],
                    len(rows),
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM monthly_kpi_evaluations WHERE is_current=1").fetchone()[0],
                    len(rows),
                )
            finally:
                connection.close()

    def test_rule_version_change_creates_new_evaluation_and_preserves_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "version.sqlite")
            try:
                row = MonthlyPerformance("A", date(2026, 1, 1), 1_600_000, 6)
                v1 = KpiRule("kpi-v1", date(2026, 1, 1), premium_threshold_satang=1_500_000)
                v2 = KpiRule("kpi-v2", date(2026, 1, 1), premium_threshold_satang=1_550_000)
                persist_agent_series(
                    connection, agent_id="A", agent_name="A", joined_on=date(2026, 1, 1),
                    initial_contract_type="salary", performances=[row], rules=[v1]
                )
                persist_agent_series(
                    connection, agent_id="A", agent_name="A", joined_on=date(2026, 1, 1),
                    initial_contract_type="salary", performances=[row], rules=[v2]
                )
                evaluations = connection.execute(
                    "SELECT rule_id, result, snapshot_rule_premium_threshold_satang FROM monthly_kpi_evaluations ORDER BY evaluation_revision"
                ).fetchall()
                self.assertEqual([(row[0], row[1]) for row in evaluations], [("kpi-v1", "PASS"), ("kpi-v2", "PASS")])
                self.assertEqual(evaluations[0][2], 1_500_000)
                self.assertEqual(evaluations[1][2], 1_550_000)
            finally:
                connection.close()

    def test_existing_rule_id_cannot_change_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "immutable.sqlite")
            try:
                row = MonthlyPerformance("A", date(2026, 1, 1), 1_600_000, 6)
                persist_agent_series(
                    connection, agent_id="A", agent_name="A", joined_on=date(2026, 1, 1),
                    initial_contract_type="salary", performances=[row],
                    rules=[KpiRule("kpi-v1", date(2026, 1, 1), premium_threshold_satang=1_500_000)]
                )
                with self.assertRaisesRegex(ValueError, "immutable"):
                    persist_agent_series(
                        connection, agent_id="A", agent_name="A", joined_on=date(2026, 1, 1),
                        initial_contract_type="salary", performances=[row],
                        rules=[KpiRule("kpi-v1", date(2026, 1, 1), premium_threshold_satang=1_700_000)]
                    )
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
