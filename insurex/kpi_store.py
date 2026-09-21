"""Persistence for the deterministic KPI engine."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable

from insurex.db import connect_analytics_local, connect_analytics_runtime, row_value
from insurex.kpi import KpiRule, MonthlyPerformance, evaluate_series, input_hash


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def connect(path: Path) -> sqlite3.Connection:
    return connect_analytics_local(path)


def connect_from_settings(settings: object | None = None):
    """Open the configured local SQLite or Azure PostgreSQL business database."""

    if settings is None:
        from insurex.config import Settings

        settings = Settings.from_env()
    return connect_analytics_runtime(settings)


def upsert_agent(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    agent_name: str,
    joined_on: date,
    initial_contract_type: str,
    inactive_on: date | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO agents(agent_id, agent_name, joined_on, inactive_on, initial_contract_type)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(agent_id) DO UPDATE SET
            agent_name=excluded.agent_name,
            joined_on=excluded.joined_on,
            inactive_on=excluded.inactive_on,
            initial_contract_type=excluded.initial_contract_type
        """,
        (agent_id, agent_name, joined_on.isoformat(), inactive_on.isoformat() if inactive_on else None, initial_contract_type),
    )


def upsert_rule(connection: sqlite3.Connection, rule: KpiRule) -> None:
    existing = connection.execute(
        """
        SELECT valid_from_month, valid_to_month, premium_threshold_satang,
               policy_threshold, required_streak, comparison_operator
        FROM kpi_rules WHERE rule_id=?
        """,
        (rule.rule_id,),
    ).fetchone()
    expected = (
        rule.valid_from_month.isoformat(),
        rule.valid_to_month.isoformat() if rule.valid_to_month else None,
        rule.premium_threshold_satang,
        rule.policy_threshold,
        rule.required_streak,
        rule.comparison_operator,
    )
    if existing:
        actual = tuple(existing[key] for key in (
            "valid_from_month", "valid_to_month", "premium_threshold_satang",
            "policy_threshold", "required_streak", "comparison_operator",
        ))
        if actual != expected:
            raise ValueError(f"KPI rule {rule.rule_id} is immutable; create a new rule_id for a new version")
        return
    connection.execute(
        """
        INSERT INTO kpi_rules(
            rule_id, valid_from_month, valid_to_month, premium_threshold_satang,
            policy_threshold, required_streak, comparison_operator
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rule.rule_id,
            rule.valid_from_month.isoformat(),
            rule.valid_to_month.isoformat() if rule.valid_to_month else None,
            rule.premium_threshold_satang,
            rule.policy_threshold,
            rule.required_streak,
            rule.comparison_operator,
        ),
    )


def _upsert_performance(connection: sqlite3.Connection, row: MonthlyPerformance) -> int:
    digest = input_hash(row)
    existing = connection.execute(
        "SELECT performance_id, input_hash, input_revision FROM monthly_agent_performance WHERE agent_id = ? AND month_start = ?",
        (row.agent_id, row.month_start.isoformat()),
    ).fetchone()
    if existing and existing["input_hash"] == digest:
        return int(existing["performance_id"])
    revision = int(existing["input_revision"] + 1) if existing else 1
    if existing:
        connection.execute(
            """
            UPDATE monthly_agent_performance
            SET total_premium_satang=?, new_policy_count=?, data_status=?, source_kind=?, month_closed=?,
                input_revision=?, input_hash=?
            WHERE performance_id=?
            """,
            (
                row.total_premium_satang,
                row.new_policy_count,
                row.data_status,
                row.source_kind,
                int(row.month_closed),
                revision,
                digest,
                existing["performance_id"],
            ),
        )
        return int(existing["performance_id"])
    cursor = connection.execute(
        """
        INSERT INTO monthly_agent_performance(
            agent_id, month_start, total_premium_satang, new_policy_count,
            data_status, source_kind, month_closed, input_revision, input_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING performance_id
        """,
        (
            row.agent_id,
            row.month_start.isoformat(),
            row.total_premium_satang,
            row.new_policy_count,
            row.data_status,
            row.source_kind,
            int(row.month_closed),
            revision,
            digest,
        ),
    )
    return int(row_value(cursor.fetchone(), "performance_id"))


def persist_agent_series(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    agent_name: str,
    joined_on: date,
    initial_contract_type: str,
    performances: Iterable[MonthlyPerformance],
    rules: Iterable[KpiRule],
    inactive_on: date | None = None,
    closed_through: date | None = None,
) -> tuple[list[int], list[int]]:
    """Persist current evaluations and reconcile transition events idempotently.

    Returns evaluation IDs and active contract event IDs in chronological order.
    """

    rows = list(performances)
    rule_list = list(rules)
    with connection:
        upsert_agent(
            connection,
            agent_id=agent_id,
            agent_name=agent_name,
            joined_on=joined_on,
            initial_contract_type=initial_contract_type,
            inactive_on=inactive_on,
        )
        for rule in rule_list:
            upsert_rule(connection, rule)
        performance_ids = {row.month_start: _upsert_performance(connection, row) for row in rows}
        evaluations, events = evaluate_series(
            agent_id=agent_id,
            initial_contract_type=initial_contract_type,
            performances=rows,
            rules=rule_list,
            joined_on=joined_on,
            inactive_on=inactive_on,
            closed_through=closed_through,
        )

        evaluation_ids: list[int] = []
        for evaluation in evaluations:
            performance_id = performance_ids[evaluation.month_start]
            current = connection.execute(
                """
                SELECT evaluation_id, evaluation_revision, input_hash, rule_id, result, pass_streak, fail_streak
                FROM monthly_kpi_evaluations
                WHERE performance_id=? AND is_current=1
                """,
                (performance_id,),
            ).fetchone()
            same = current and (
                current["input_hash"] == evaluation.input_hash
                and current["rule_id"] == evaluation.rule_id
                and current["result"] == evaluation.result
                and current["pass_streak"] == evaluation.pass_streak
                and current["fail_streak"] == evaluation.fail_streak
            )
            if same:
                evaluation_ids.append(int(current["evaluation_id"]))
                continue
            revision = int(current["evaluation_revision"] + 1) if current else 1
            if current:
                connection.execute(
                    "UPDATE monthly_kpi_evaluations SET is_current=0 WHERE evaluation_id=?",
                    (current["evaluation_id"],),
                )
            stored_performance = connection.execute(
                "SELECT input_revision FROM monthly_agent_performance WHERE performance_id=?",
                (performance_id,),
            ).fetchone()
            performance_row = next(row for row in rows if row.month_start == evaluation.month_start)
            rule = next(rule for rule in rule_list if rule.rule_id == evaluation.rule_id)
            cursor = connection.execute(
                """
                INSERT INTO monthly_kpi_evaluations(
                    performance_id, rule_id, evaluation_revision, result,
                    pass_streak, fail_streak, evaluated_at, input_hash,
                    performance_input_revision, snapshot_total_premium_satang,
                    snapshot_new_policy_count, snapshot_data_status,
                    snapshot_rule_premium_threshold_satang, snapshot_rule_policy_threshold,
                    snapshot_rule_required_streak, snapshot_rule_comparison_operator, is_current
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                RETURNING evaluation_id
                """,
                (
                    performance_id,
                    evaluation.rule_id,
                    revision,
                    evaluation.result,
                    evaluation.pass_streak,
                    evaluation.fail_streak,
                    utc_now(),
                    evaluation.input_hash,
                    int(row_value(stored_performance, "input_revision")),
                    performance_row.total_premium_satang,
                    performance_row.new_policy_count,
                    performance_row.data_status,
                    rule.premium_threshold_satang,
                    rule.policy_threshold,
                    rule.required_streak,
                    rule.comparison_operator,
                ),
            )
            evaluation_ids.append(int(row_value(cursor.fetchone(), "evaluation_id")))

        expected_keys = {(event.effective_from.isoformat(), event.to_type) for event in events}
        active = connection.execute(
            "SELECT contract_event_id, effective_from, to_type FROM contract_history WHERE agent_id=? AND superseded_at IS NULL",
            (agent_id,),
        ).fetchall()
        for existing in active:
            if (existing["effective_from"], existing["to_type"]) not in expected_keys:
                connection.execute(
                    "UPDATE contract_history SET superseded_at=? WHERE contract_event_id=?",
                    (utc_now(), existing["contract_event_id"]),
                )

        event_ids: list[int] = []
        eval_by_month = {evaluation.month_start: evaluation_ids[index] for index, evaluation in enumerate(evaluations)}
        for event in events:
            trigger_id = eval_by_month[event.trigger_month]
            existing = connection.execute(
                """
                SELECT contract_event_id FROM contract_history
                WHERE agent_id=? AND effective_from=? AND to_type=?
                """,
                (agent_id, event.effective_from.isoformat(), event.to_type),
            ).fetchone()
            if existing:
                connection.execute(
                    """
                    UPDATE contract_history
                    SET from_type=?, trigger_evaluation_id=?, reason=?, superseded_at=NULL
                    WHERE contract_event_id=?
                    """,
                    (event.from_type, trigger_id, event.reason, existing["contract_event_id"]),
                )
                event_ids.append(int(existing["contract_event_id"]))
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO contract_history(
                        agent_id, from_type, to_type, effective_from,
                        trigger_evaluation_id, reason
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    RETURNING contract_event_id
                    """,
                    (
                        agent_id,
                        event.from_type,
                        event.to_type,
                        event.effective_from.isoformat(),
                        trigger_id,
                        event.reason,
                    ),
                )
                event_ids.append(int(row_value(cursor.fetchone(), "contract_event_id")))
        return evaluation_ids, event_ids
