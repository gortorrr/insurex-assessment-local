"""Split the legacy combined SQLite file into Test Case #1 and #2 stores."""

from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

from insurex.auth import setup as setup_auth
from insurex.db import connect_analytics_local, connect_assistant_local, connect_local


ANALYTICS_TABLES = (
    "agents", "kpi_rules", "monthly_agent_performance",
    "monthly_kpi_evaluations", "contract_history",
)
ASSISTANT_TABLES = (
    "user_accounts", "login_sessions", "sessions", "messages", "leads",
    "lead_drafts", "lead_request_log", "handoff_cases",
)


def _columns(connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()]


def _copy_table(source, destination, table: str) -> int:
    source_columns = _columns(source, table)
    destination_columns = _columns(destination, table)
    columns = [name for name in destination_columns if name in source_columns]
    if not columns:
        return 0
    rows = source.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall()
    if not rows:
        return 0
    placeholders = ",".join("?" for _ in columns)
    destination.executemany(
        f"INSERT OR REPLACE INTO {table}({','.join(columns)}) VALUES ({placeholders})",
        [tuple(row[name] for name in columns) for row in rows],
    )
    return len(rows)


def split(source_path: Path, assistant_path: Path, analytics_path: Path) -> dict:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = source_path.with_name(f"{source_path.stem}.before-split-{timestamp}{source_path.suffix}")
    shutil.copy2(source_path, backup)
    source = connect_local(source_path)
    assistant = connect_assistant_local(assistant_path)
    analytics = connect_analytics_local(analytics_path)
    try:
        setup_auth(source)
        setup_auth(assistant)
        assistant_counts = {table: _copy_table(source, assistant, table) for table in ASSISTANT_TABLES}
        analytics_counts = {table: _copy_table(source, analytics, table) for table in ANALYTICS_TABLES}
        assistant.commit()
        analytics.commit()
    finally:
        source.close()
        assistant.close()
        analytics.close()
    return {
        "source_backup": str(backup),
        "assistant_database": str(assistant_path),
        "analytics_database": str(analytics_path),
        "assistant_rows": assistant_counts,
        "analytics_rows": analytics_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("tmp/legacy_db/insurex.sqlite"))
    parser.add_argument("--assistant", type=Path, default=Path("rag/db/assistant.sqlite"))
    parser.add_argument("--analytics", type=Path, default=Path("data_modeling/db/analytics.sqlite"))
    args = parser.parse_args()
    print(split(args.source, args.assistant, args.analytics))


if __name__ == "__main__":
    main()
