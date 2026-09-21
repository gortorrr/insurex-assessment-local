"""Database adapters for the local SQLite and Azure PostgreSQL profiles.

The application uses a small DB-API-shaped surface so the deterministic KPI,
lead, and session services can run against SQLite locally and PostgreSQL in
Azure.  The PostgreSQL dependency is optional until the Azure profile is used.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping


class DatabaseDependencyError(RuntimeError):
    """Raised when the selected runtime backend is not installed/configured."""


class PostgresConnectionAdapter:
    """Small DB-API adapter translating the repository's qmark SQL safely."""

    def __init__(self, raw: Any):
        self.raw = raw

    @staticmethod
    def _sql(query: str) -> str:
        # psycopg treats percent signs as interpolation markers even when the
        # application uses qmark SQL. Escape literal LIKE/wildcard signs first,
        # then translate the repository's qmark placeholders.
        return query.replace("%", "%%").replace("?", "%s")

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        return self.raw.execute(self._sql(query), params)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        self.raw.close()

    def __enter__(self) -> "PostgresConnectionAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        # Deliberately do not close here. A caller can reuse one connection for
        # multiple agents/transactions and close it explicitly at the boundary.


def row_value(row: Any, key: str, index: int = 0) -> Any:
    """Read a value from either sqlite3.Row or psycopg's dict_row."""

    if isinstance(row, Mapping):
        return row[key]
    return row[index]


def connect_local(path: Path) -> sqlite3.Connection:
    """Compatibility connector for migration/tests that need both test cases."""

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    schema = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "db" / "schema.legacy-combined.sql"
    connection.executescript(schema.read_text(encoding="utf-8"))
    _migrate_sqlite(connection)
    return connection


def _connect_scoped_local(path: Path, schema_name: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    root = Path(__file__).resolve().parents[1]
    schema = {
        "schema.assistant.sql": root / "rag" / "db" / "schema.assistant.sql",
        "schema.analytics.sql": root / "data_modeling" / "schema.sql",
    }.get(schema_name, root / "db" / schema_name)
    connection.executescript(schema.read_text(encoding="utf-8"))
    return connection


def connect_assistant_local(path: Path) -> sqlite3.Connection:
    """Open only the Test Case #2 account/chat/lead/handoff database."""

    return _connect_scoped_local(path, "schema.assistant.sql")


def connect_analytics_local(path: Path) -> sqlite3.Connection:
    """Open only the Test Case #1 KPI/data-modeling database."""

    return _connect_scoped_local(path, "schema.analytics.sql")


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}


def _migrate_sqlite(connection: sqlite3.Connection) -> None:
    """Add audit/idempotency columns to databases created by the earlier baseline."""

    for column, definition in {
        "purpose": "TEXT NOT NULL DEFAULT 'private'",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        "sender_type": "TEXT NOT NULL DEFAULT 'ai'",
        "sender_account_id": "TEXT",
        "sender_label": "TEXT",
    }.items():
        if column not in _sqlite_columns(connection, "messages"):
            connection.execute(f"ALTER TABLE messages ADD COLUMN {column} {definition}")
    connection.execute("UPDATE messages SET sender_type='customer' WHERE role='user' AND sender_type='ai'")
    connection.execute("UPDATE messages SET sender_type='system' WHERE role='system' AND sender_type='ai'")
    evaluation_defaults = {
        "performance_input_revision": "INTEGER NOT NULL DEFAULT 1",
        "snapshot_total_premium_satang": "INTEGER",
        "snapshot_new_policy_count": "INTEGER",
        "snapshot_data_status": "TEXT NOT NULL DEFAULT 'COMPLETE'",
        "snapshot_rule_premium_threshold_satang": "INTEGER NOT NULL DEFAULT 0",
        "snapshot_rule_policy_threshold": "INTEGER NOT NULL DEFAULT 0",
        "snapshot_rule_required_streak": "INTEGER NOT NULL DEFAULT 1",
        "snapshot_rule_comparison_operator": "TEXT NOT NULL DEFAULT '>'",
    }
    for column, definition in evaluation_defaults.items():
        if column not in _sqlite_columns(connection, "monthly_kpi_evaluations"):
            connection.execute(f"ALTER TABLE monthly_kpi_evaluations ADD COLUMN {column} {definition}")
    if "month_closed" not in _sqlite_columns(connection, "monthly_agent_performance"):
        connection.execute("ALTER TABLE monthly_agent_performance ADD COLUMN month_closed INTEGER NOT NULL DEFAULT 1")

    if "payload_hash" not in _sqlite_columns(connection, "leads"):
        connection.execute("ALTER TABLE leads ADD COLUMN payload_hash TEXT NOT NULL DEFAULT ''")
    for column in ("profile_verified_at", "refresh_requested_at"):
        if column not in _sqlite_columns(connection, "leads"):
            connection.execute(f"ALTER TABLE leads ADD COLUMN {column} TEXT")

    lead_columns = _sqlite_columns(connection, "leads")
    if "income_satang" in lead_columns and "income_thb" not in lead_columns:
        connection.execute("ALTER TABLE leads RENAME COLUMN income_satang TO income_thb")
        connection.execute("UPDATE leads SET income_thb = income_thb / 100.0")

    draft_columns = _sqlite_columns(connection, "lead_drafts")
    if "owner_id" not in draft_columns:
        connection.execute("ALTER TABLE lead_drafts RENAME TO lead_drafts_legacy")
        connection.execute(
            """
            CREATE TABLE lead_drafts (
                owner_id TEXT PRIMARY KEY,
                last_session_id TEXT NOT NULL REFERENCES sessions(session_id),
                product_id TEXT,
                name TEXT,
                occupation TEXT,
                income_value TEXT,
                income_min TEXT,
                income_max TEXT,
                income_currency TEXT,
                income_period TEXT,
                phone_raw TEXT,
                phone_normalized TEXT,
                status TEXT NOT NULL CHECK (status IN ('unconfirmed', 'confirmed', 'cancelled')),
                draft_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO lead_drafts(
                owner_id, last_session_id, product_id, name, occupation,
                income_value, income_min, income_max, income_currency,
                income_period, phone_raw, phone_normalized, status,
                draft_json, updated_at
            )
            SELECT s.owner_id, d.session_id, d.product_id, d.name, d.occupation,
                   d.income_value, d.income_min, d.income_max, d.income_currency,
                   d.income_period, d.phone_raw, d.phone_normalized, d.status,
                   d.draft_json, d.updated_at
            FROM lead_drafts_legacy AS d
            JOIN sessions AS s ON s.session_id=d.session_id
            WHERE d.rowid=(
                SELECT d2.rowid
                FROM lead_drafts_legacy AS d2
                JOIN sessions AS s2 ON s2.session_id=d2.session_id
                WHERE s2.owner_id=s.owner_id
                ORDER BY d2.updated_at DESC, d2.rowid DESC
                LIMIT 1
            )
            """
        )
        connection.execute("DROP TABLE lead_drafts_legacy")

    # Rebuild legacy session-scoped leads as one canonical customer profile per
    # account. Keep the most recently updated complete record for each owner.
    has_session_fk = bool(connection.execute("PRAGMA foreign_key_list(leads)").fetchall())
    lead_columns = _sqlite_columns(connection, "leads")
    if not has_session_fk or "owner_id" not in lead_columns:
        # schema.sql creates this table for fresh databases. On an existing
        # database SQLite retargets its FK when `leads` is renamed, so remove
        # the still-empty compatibility table before rebuilding the parent.
        connection.execute("DROP TABLE IF EXISTS lead_request_log")
        connection.execute("ALTER TABLE leads RENAME TO leads_legacy")
        connection.execute(
            """
            CREATE TABLE leads (
                lead_id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                product_id TEXT NOT NULL,
                name TEXT NOT NULL,
                occupation TEXT NOT NULL,
                income_thb NUMERIC NOT NULL CHECK (income_thb >= 0),
                income_period TEXT NOT NULL CHECK (income_period = 'monthly_thb'),
                phone_normalized TEXT NOT NULL,
                request_id TEXT NOT NULL UNIQUE,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO leads(
                lead_id, owner_id, session_id, product_id, name, occupation, income_thb,
                income_period, phone_normalized, request_id, payload_hash, created_at, updated_at
            )
            SELECT l.lead_id, s.owner_id, l.session_id, l.product_id, l.name,
                   l.occupation, l.income_thb, l.income_period,
                   l.phone_normalized, l.request_id, l.payload_hash,
                   l.created_at, l.updated_at
            FROM leads_legacy AS l
            JOIN sessions AS s ON s.session_id=l.session_id
            WHERE l.lead_id=(
                SELECT l2.lead_id
                FROM leads_legacy AS l2
                JOIN sessions AS s2 ON s2.session_id=l2.session_id
                WHERE s2.owner_id=s.owner_id
                ORDER BY l2.updated_at DESC, l2.lead_id DESC
                LIMIT 1
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS lead_request_log (
                request_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                lead_id INTEGER NOT NULL REFERENCES leads(lead_id),
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO lead_request_log(
                request_id, owner_id, lead_id, payload_hash, created_at
            )
            SELECT legacy.request_id, sessions.owner_id, canonical.lead_id,
                   legacy.payload_hash, legacy.created_at
            FROM leads_legacy AS legacy
            JOIN sessions ON sessions.session_id=legacy.session_id
            JOIN leads AS canonical ON canonical.owner_id=sessions.owner_id
            """
        )
        connection.execute("DROP TABLE leads_legacy")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS lead_request_log (
            request_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            lead_id INTEGER NOT NULL REFERENCES leads(lead_id),
            payload_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    # The product now has one durable customer conversation per account. Keep
    # the newest active conversation, preserve its transcript, and retire old
    # parallel chats after repointing the canonical profile/draft.
    owners = connection.execute(
        "SELECT owner_id FROM sessions WHERE status='active' GROUP BY owner_id HAVING COUNT(*) > 1"
    ).fetchall()
    for owner in owners:
        owner_id = owner["owner_id"]
        keep = connection.execute(
            "SELECT session_id FROM sessions WHERE owner_id=? AND status='active' "
            "ORDER BY CASE WHEN session_id=(SELECT session_id FROM leads WHERE owner_id=?) THEN 0 ELSE 1 END, "
            "last_active_at DESC, created_at DESC LIMIT 1",
            (owner_id, owner_id),
        ).fetchone()["session_id"]
        connection.execute("UPDATE leads SET session_id=? WHERE owner_id=?", (keep, owner_id))
        connection.execute("UPDATE lead_drafts SET last_session_id=? WHERE owner_id=?", (keep, owner_id))
        connection.execute(
            "DELETE FROM messages WHERE session_id IN (SELECT session_id FROM sessions WHERE owner_id=? AND status='active' AND session_id<>?)",
            (owner_id, keep),
        )
        connection.execute(
            "UPDATE sessions SET status='closed', owner_token_hash='' WHERE owner_id=? AND status='active' AND session_id<>?",
            (owner_id, keep),
        )
    connection.execute("DROP INDEX IF EXISTS one_active_conversation_per_customer")
    connection.commit()


def _execute_postgres_schema(connection: Any, schema_text: str) -> None:
    # The schema contains plain DDL statements without functions or procedural
    # blocks, so statement splitting is deterministic and dependency-free.
    for statement in schema_text.split(";"):
        statement = statement.strip()
        if statement:
            connection.execute(statement)
    connection.commit()


def connect_postgres(
    database_url: str,
    *,
    initialize_schema: bool = True,
    schema_name: str = "schema.legacy-combined.postgres.sql",
) -> Any:
    if not database_url:
        raise ValueError("DATABASE_URL is required for the postgresql backend")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise DatabaseDependencyError(
            "Install the Azure profile dependencies from requirements.azure.txt"
        ) from exc

    connection = PostgresConnectionAdapter(psycopg.connect(database_url, row_factory=dict_row))
    root = Path(__file__).resolve().parents[1]
    schema = {
        "schema.assistant.postgres.sql": root / "rag" / "db" / "schema.assistant.postgres.sql",
        "schema.analytics.postgres.sql": root / "data_modeling" / "schema.postgres.reference.sql",
        "schema.legacy-combined.postgres.sql": root / "tests" / "fixtures" / "db" / "schema.legacy-combined.postgres.sql",
    }.get(schema_name, root / "tests" / "fixtures" / "db" / schema_name)
    if initialize_schema:
        _execute_postgres_schema(connection, schema.read_text(encoding="utf-8"))
    if initialize_schema and schema_name in {
        "schema.assistant.postgres.sql", "schema.legacy-combined.postgres.sql"
    }:
        for statement in (
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sender_type TEXT NOT NULL DEFAULT 'ai'",
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sender_account_id TEXT",
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS sender_label TEXT",
            "ALTER TABLE leads ADD COLUMN IF NOT EXISTS profile_verified_at TEXT",
            "ALTER TABLE leads ADD COLUMN IF NOT EXISTS refresh_requested_at TEXT",
        ):
            connection.execute(statement)
        connection.execute("UPDATE messages SET sender_type='customer' WHERE role='user' AND sender_type='ai'")
        connection.execute("UPDATE messages SET sender_type='system' WHERE role='system' AND sender_type='ai'")
        connection.commit()
        columns = {
            row_value(row, "column_name")
            for row in connection.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='leads'"
            ).fetchall()
        }
        if "income_satang" in columns and "income_thb" not in columns:
            connection.execute("ALTER TABLE leads RENAME COLUMN income_satang TO income_thb")
            connection.execute(
                "ALTER TABLE leads ALTER COLUMN income_thb TYPE NUMERIC(14,2) USING income_thb / 100.0"
            )
            connection.commit()
    return connection


def connect_runtime(settings: Any, *, initialize_schema: bool = True) -> Any:
    if settings.database_backend == "local":
        return connect_local(settings.business_db_path)
    if settings.database_backend == "postgresql":
        return connect_postgres(settings.database_url, initialize_schema=initialize_schema)
    raise ValueError("DATABASE_BACKEND must be local or postgresql")


def connect_assistant_runtime(settings: Any, *, initialize_schema: bool = True) -> Any:
    if settings.database_backend == "local":
        path = settings.assistant_db_path or settings.business_db_path
        return connect_assistant_local(path)
    return connect_postgres(
        settings.database_url,
        initialize_schema=initialize_schema,
        schema_name="schema.assistant.postgres.sql",
    )


def connect_analytics_runtime(settings: Any, *, initialize_schema: bool = True) -> Any:
    if settings.database_backend == "local":
        return connect_analytics_local(settings.analytics_db_path)
    return connect_postgres(
        settings.database_url,
        initialize_schema=initialize_schema,
        schema_name="schema.analytics.postgres.sql",
    )


def query_rows(connection: Any, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Return query results as dictionaries for Streamlit and API adapters."""

    cursor = connection.execute(query, params)
    rows = cursor.fetchall()
    if not rows:
        return []
    if isinstance(rows[0], Mapping):
        return [dict(row) for row in rows]
    columns = [column.name if hasattr(column, "name") else column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in rows]
