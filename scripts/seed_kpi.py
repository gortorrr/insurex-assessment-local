"""Seed and evaluate the labelled synthetic KPI demonstration fixture."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insurex.config import Settings
from insurex.db import row_value
from insurex.kpi import evaluate_series, synthetic_fixture
from insurex.kpi_store import connect, connect_from_settings, persist_agent_series


def reset_local_database(path: Path) -> None:
    """Recreate the demo database, including when Windows holds the file open."""

    if not path.exists():
        return
    try:
        path.unlink()
        return
    except PermissionError:
        pass
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executescript(
            """
            DROP TABLE IF EXISTS contract_history;
            DROP TABLE IF EXISTS monthly_kpi_evaluations;
            DROP TABLE IF EXISTS monthly_agent_performance;
            DROP TABLE IF EXISTS kpi_rules;
            DROP TABLE IF EXISTS agents;
            """
        )
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="Use a local SQLite path instead of the configured profile")
    parser.add_argument("--reset", action="store_true", help="Delete and recreate the explicitly selected local SQLite database")
    args = parser.parse_args()
    if args.reset and args.db is None:
        parser.error("--reset requires --db so a configured remote database can never be deleted")
    if args.db:
        args.db.parent.mkdir(parents=True, exist_ok=True)
        if args.reset and args.db.exists():
            reset_local_database(args.db)
    agents, rules, performances = synthetic_fixture()
    settings = Settings.from_env()
    connection = connect(args.db) if args.db else connect_from_settings(settings)
    summary = []
    for agent in agents:
        rows = [row for row in performances if row.agent_id == agent["agent_id"]]
        evaluation_ids, event_ids = persist_agent_series(
            connection,
            agent_id=agent["agent_id"],
            agent_name=agent["agent_name"],
            joined_on=date(2026, 1, 1),
            initial_contract_type=agent["initial_contract_type"],
            performances=rows,
            rules=rules,
        )
        evaluations, events = evaluate_series(
            agent_id=agent["agent_id"],
            initial_contract_type=agent["initial_contract_type"],
            performances=rows,
            rules=rules,
        )
        summary.append(
            {
                "agent_id": agent["agent_id"],
                "source_kind": "synthetic",
                "evaluation_ids": evaluation_ids,
                "results": [evaluation.result for evaluation in evaluations],
                "streaks": [
                    {"month": evaluation.month_start.isoformat(), "pass": evaluation.pass_streak, "fail": evaluation.fail_streak}
                    for evaluation in evaluations
                ],
                "contract_events": [
                    {"from": event.from_type, "to": event.to_type, "effective_from": event.effective_from.isoformat(), "trigger_month": event.trigger_month.isoformat()}
                    for event in events
                ],
                "persisted_event_ids": event_ids,
            }
        )
    fixture_path = ROOT / "data_modeling" / "fixtures" / "kpi_seed_summary.json"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(
        json.dumps({"synthetic": True, "agents": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    counts = {
        "agents": row_value(connection.execute("SELECT COUNT(*) AS count FROM agents").fetchone(), "count"),
        "performances": row_value(connection.execute("SELECT COUNT(*) AS count FROM monthly_agent_performance").fetchone(), "count"),
        "current_evaluations": row_value(connection.execute("SELECT COUNT(*) AS count FROM monthly_kpi_evaluations WHERE is_current=1").fetchone(), "count"),
        "active_contract_events": row_value(connection.execute("SELECT COUNT(*) AS count FROM contract_history WHERE superseded_at IS NULL").fetchone(), "count"),
    }
    print(json.dumps({"database_backend": settings.database_backend if not args.db else "local", "db": str(args.db or settings.analytics_db_path), "synthetic": True, "counts": counts}, ensure_ascii=False, indent=2))
    connection.close()


if __name__ == "__main__":
    main()
