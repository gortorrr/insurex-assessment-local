import sqlite3
from pathlib import Path
import tempfile
import unittest

from insurex.db import PostgresConnectionAdapter, connect_local


class FakeRawConnection:
    def __init__(self):
        self.calls = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def execute(self, query, params):
        self.calls.append((query, params))
        return object()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True


class DatabaseAdapterTests(unittest.TestCase):
    def test_postgres_adapter_translates_qmark_and_context_does_not_close(self):
        raw = FakeRawConnection()
        connection = PostgresConnectionAdapter(raw)
        connection.execute("SELECT * FROM leads WHERE session_id=? AND request_id=?", ("s", "r"))
        self.assertEqual(raw.calls[0][0], "SELECT * FROM leads WHERE session_id=%s AND request_id=%s")
        with connection:
            connection.execute("SELECT 1", ())
        self.assertEqual(raw.commits, 1)
        self.assertFalse(raw.closed)

    def test_postgres_adapter_escapes_literal_percent_for_psycopg(self):
        raw = FakeRawConnection()
        connection = PostgresConnectionAdapter(raw)
        connection.execute("SELECT * FROM leads WHERE request_id LIKE 'pg-lead-%'", ())
        self.assertEqual(
            raw.calls[0][0],
            "SELECT * FROM leads WHERE request_id LIKE 'pg-lead-%%'",
        )

    def test_local_schema_has_lead_fk_and_audit_snapshot_columns(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect_local(Path(temporary) / "adapter.sqlite")
            try:
                fk = connection.execute("PRAGMA foreign_key_list(leads)").fetchall()
                self.assertTrue(any(row[2] == "sessions" for row in fk))
                columns = {row[1] for row in connection.execute("PRAGMA table_info(monthly_kpi_evaluations)")}
                self.assertIn("performance_input_revision", columns)
                self.assertIn("snapshot_rule_premium_threshold_satang", columns)
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
