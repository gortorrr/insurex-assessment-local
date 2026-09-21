"""Run PostgreSQL integration checks with run-owned, exact cleanup.

The script is fail-closed and is intentionally not part of the local test
suite. Use it only with an explicitly selected PostgreSQL environment.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def redact(value: object) -> str:
    text = str(value)
    text = re.sub(r"(?i)(postgres(?:ql)?://)[^\s]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(password|passwd|pwd)=?[^\s,;]+", r"\1=[REDACTED]", text)
    database_url = os.environ.get("DATABASE_URL", "")
    return text.replace(database_url, "[REDACTED]") if database_url else text


def require_dependencies() -> None:
    missing: list[str] = []
    for module in ("psycopg", "langgraph.checkpoint.postgres"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError("PostgreSQL integration dependencies are missing: " + ", ".join(missing))


@dataclass
class RunOwnership:
    """Exact identifiers created by one run; no prefix cleanup is allowed."""

    run_id: str
    agent_id: str
    rule_id: str
    owner_a: str
    owner_b: str
    session_ids: list[str] = field(default_factory=list)
    message_ids: list[int] = field(default_factory=list)
    lead_ids: list[int] = field(default_factory=list)
    performance_ids: list[int] = field(default_factory=list)
    evaluation_ids: list[int] = field(default_factory=list)
    contract_ids: list[int] = field(default_factory=list)
    checkpoint_thread_id: str = ""

    @classmethod
    def create(cls, run_id: str | None = None) -> "RunOwnership":
        run = run_id or uuid.uuid4().hex[:16]
        return cls(
            run_id=run,
            agent_id=f"PG_INTEGRATION_{run}",
            rule_id=f"pg-integration-rule-{run}",
            owner_a=f"pg-integration-owner-a-{run}",
            owner_b=f"pg-integration-owner-b-{run}",
            checkpoint_thread_id=f"pg-checkpoint-{run}",
        )


def build_cleanup_plan(ownership: RunOwnership) -> list[tuple[str, str, Any]]:
    """Return exact-id deletion statements; values are never SQL patterns."""

    plan: list[tuple[str, str, Any]] = []
    for value in ownership.message_ids:
        plan.append(("messages", "DELETE FROM messages WHERE message_id=?", value))
    for value in ownership.lead_ids:
        plan.append(("leads", "DELETE FROM leads WHERE lead_id=?", value))
    for value in ownership.session_ids:
        plan.append(("sessions", "DELETE FROM sessions WHERE session_id=?", value))
    for value in ownership.contract_ids:
        plan.append(("contracts", "DELETE FROM contract_history WHERE contract_event_id=?", value))
    for value in ownership.evaluation_ids:
        plan.append(("evaluations", "DELETE FROM monthly_kpi_evaluations WHERE evaluation_id=?", value))
    for value in ownership.performance_ids:
        plan.append(("performances", "DELETE FROM monthly_agent_performance WHERE performance_id=?", value))
    plan.append(("rules", "DELETE FROM kpi_rules WHERE rule_id=?", ownership.rule_id))
    plan.append(("agents", "DELETE FROM agents WHERE agent_id=?", ownership.agent_id))
    if ownership.checkpoint_thread_id:
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            plan.append((table, f"DELETE FROM {table} WHERE thread_id=?", ownership.checkpoint_thread_id))
    return plan


def cleanup_allowed(*, preflight_passed: bool, mutation_started: bool) -> bool:
    """Cleanup is safe only after the target passed checks and writes began."""

    return preflight_passed and mutation_started


def _count_owned_rows(connection: Any, ownership: RunOwnership) -> dict[str, int]:
    checks = {
        "agent": ("SELECT COUNT(*) AS count FROM agents WHERE agent_id=?", (ownership.agent_id,)),
        "rule": ("SELECT COUNT(*) AS count FROM kpi_rules WHERE rule_id=?", (ownership.rule_id,)),
        "owner_a": ("SELECT COUNT(*) AS count FROM sessions WHERE owner_id=?", (ownership.owner_a,)),
        "owner_b": ("SELECT COUNT(*) AS count FROM sessions WHERE owner_id=?", (ownership.owner_b,)),
    }
    return {name: int(connection.execute(query, params).fetchone()["count"]) for name, (query, params) in checks.items()}


def cleanup_test_rows(settings: object, ownership: RunOwnership) -> dict[str, object]:
    """Delete only identifiers captured by this run and verify they are gone."""

    from insurex.db import connect_runtime

    deleted: dict[str, int] = {}
    connection = connect_runtime(settings, initialize_schema=False)
    try:
        with connection:
            for name, query, value in build_cleanup_plan(ownership):
                cursor = connection.execute(query, (value,))
                deleted[name] = deleted.get(name, 0) + int(cursor.rowcount)
    finally:
        connection.close()
    verify = connect_runtime(settings, initialize_schema=False)
    try:
        columns = {
            "messages": ("messages", "message_id"), "leads": ("leads", "lead_id"),
            "sessions": ("sessions", "session_id"), "contracts": ("contract_history", "contract_event_id"),
            "evaluations": ("monthly_kpi_evaluations", "evaluation_id"),
            "performances": ("monthly_agent_performance", "performance_id"),
            "rules": ("kpi_rules", "rule_id"), "agents": ("agents", "agent_id"),
            "checkpoint_writes": ("checkpoint_writes", "thread_id"),
            "checkpoint_blobs": ("checkpoint_blobs", "thread_id"), "checkpoints": ("checkpoints", "thread_id"),
        }
        remaining = 0
        for name, query, value in build_cleanup_plan(ownership):
            table, column = columns[name]
            row = verify.execute(f"SELECT COUNT(*) AS count FROM {table} WHERE {column}=?", (value,)).fetchone()
            remaining += int(row["count"])
    finally:
        verify.close()
    if remaining:
        raise RuntimeError(f"cleanup verification found {remaining} owned rows")
    return {
        "performed": True,
        "deleted": deleted,
        "exact_ids_only": True,
        "post_cleanup_counts": {"remaining_owned_rows": remaining},
        "cleanup_verified": True,
    }


def check_connection_and_schema(settings: object, *, initialize_schema: bool, require_checkpoint_tables: bool = True) -> dict[str, object]:
    """Verify target/TLS first, then optionally initialize and inspect schema."""

    from insurex.db import connect_runtime, row_value

    connection = connect_runtime(settings, initialize_schema=initialize_schema)
    try:
        target = connection.raw.execute(
            "SELECT current_database() AS database_name, ssl, version() AS server_version "
            "FROM pg_stat_ssl WHERE pid=pg_backend_pid()"
        ).fetchone()
        if not target or not bool(row_value(target, "ssl")):
            raise RuntimeError("PostgreSQL connection is not using TLS")
        expected_database = urlparse(settings.database_url).path.lstrip("/").split("?", 1)[0]
        actual_database = str(row_value(target, "database_name"))
        if expected_database and actual_database != expected_database:
            raise RuntimeError("connected database does not match the database in DATABASE_URL")
        result: dict[str, object] = {
            "tls": True,
            "target_database_verified": bool(expected_database and actual_database == expected_database),
            "server_version_observed": bool(row_value(target, "server_version")),
        }
        if not initialize_schema and not require_checkpoint_tables:
            return result
        required_tables = {
            "agents", "kpi_rules", "monthly_agent_performance", "monthly_kpi_evaluations",
            "contract_history", "sessions", "messages", "leads",
        }
        if require_checkpoint_tables:
            required_tables.update({"checkpoints", "checkpoint_blobs", "checkpoint_writes"})
        actual_tables = {
            row["table_name"] for row in connection.raw.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
            ).fetchall()
        }
        missing_tables = sorted(required_tables - actual_tables)
        if missing_tables:
            raise RuntimeError("schema compatibility missing tables: " + ", ".join(missing_tables))
        required_columns = {
            "leads": {"session_id", "product_id", "income_thb", "income_period", "request_id", "payload_hash"},
            "sessions": {"session_id", "owner_id", "thread_id", "owner_token_hash"},
            "monthly_agent_performance": {"agent_id", "month_start", "input_revision", "input_hash", "month_closed"},
            "monthly_kpi_evaluations": {"performance_id", "rule_id", "evaluation_revision", "input_hash", "performance_input_revision"},
            "contract_history": {"agent_id", "trigger_evaluation_id", "effective_from"},
        }
        missing_columns: dict[str, list[str]] = {}
        checked_columns = 0
        for table, expected in required_columns.items():
            actual = {
                row["column_name"] for row in connection.raw.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name=%s",
                    (table,),
                ).fetchall()
            }
            checked_columns += len(expected)
            if expected - actual:
                missing_columns[table] = sorted(expected - actual)
        if missing_columns:
            raise RuntimeError("schema compatibility missing columns: " + json.dumps(missing_columns, sort_keys=True))
        foreign_keys = connection.raw.execute(
            "SELECT tc.table_name, kcu.column_name, ccu.table_name AS foreign_table_name, ccu.column_name AS foreign_column_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu ON tc.constraint_name=kcu.constraint_name AND tc.table_schema=kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu ON ccu.constraint_name=tc.constraint_name AND ccu.table_schema=tc.table_schema "
            "WHERE tc.constraint_type='FOREIGN KEY' AND tc.table_schema='public'"
        ).fetchall()
        lead_fk_ok = any(
            row["table_name"] == "leads" and row["column_name"] == "session_id"
            and row["foreign_table_name"] == "sessions" and row["foreign_column_name"] == "session_id"
            for row in foreign_keys
        )
        indexes = {row["indexname"] for row in connection.raw.execute("SELECT indexname FROM pg_indexes WHERE schemaname='public'").fetchall()}
        required_indexes = {"one_current_evaluation_performance", "message_request_unique"}
        constraints = " ".join(
            row["definition"] for row in connection.raw.execute(
                "SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint "
                "WHERE conrelid IN ('leads'::regclass, 'kpi_rules'::regclass)"
            ).fetchall()
        )
        if not lead_fk_ok or not required_indexes.issubset(indexes) or "income_period = 'monthly_thb'" not in constraints or "comparison_operator = '>'" not in constraints:
            raise RuntimeError("schema compatibility constraints, FK, or indexes are incomplete")
        result["schema_compatibility"] = {
            "tables_checked": len(required_tables),
            "columns_checked": checked_columns,
            "lead_session_fk": True,
            "indexes_checked": sorted(required_indexes),
            "check_constraints_verified": True,
        }
        return result
    finally:
        connection.close()


def check_kpi_lead_session(settings: object, ownership: RunOwnership) -> dict[str, object]:
    from decimal import Decimal
    from insurex.kpi import KpiRule, MonthlyPerformance
    from insurex.kpi_store import connect_from_settings, persist_agent_series
    from insurex.db import connect_local
    from insurex.leads import LeadDraft, ValidatedLead, save_lead, validate_lead
    from insurex.session import append_message, create_session, load_history, resume_session

    # PostgreSQL integration intentionally exercises the shared deployment
    # schema. Its local contract test uses the legacy combined adapter only as
    # an in-memory stand-in; production local runtime uses split databases.
    connection = (
        connect_local(settings.business_db_path)
        if getattr(settings, "database_backend", "local") == "local"
        else connect_from_settings(settings)
    )
    try:
        rule = KpiRule(
            rule_id=ownership.rule_id,
            valid_from_month=date(2026, 1, 1),
            premium_threshold_satang=1_500_000,
            policy_threshold=5,
            required_streak=3,
        )
        performances = [MonthlyPerformance(ownership.agent_id, date(2026, month, 1), 1_500_001, 6, source_kind="synthetic") for month in (1, 2, 3)]
        evaluations, events = persist_agent_series(
            connection, agent_id=ownership.agent_id, agent_name="Synthetic PostgreSQL integration agent",
            joined_on=date(2026, 1, 1), initial_contract_type="commission", performances=performances, rules=[rule],
        )
        ownership.evaluation_ids.extend(evaluations)
        ownership.contract_ids.extend(events)
        ownership.performance_ids.extend([
            int(row["performance_id"]) for row in connection.execute(
                "SELECT performance_id FROM monthly_agent_performance WHERE agent_id=? ORDER BY month_start", (ownership.agent_id,)
            ).fetchall()
        ])
        if len(evaluations) != 3 or len(events) != 1 or len(ownership.performance_ids) != 3:
            raise AssertionError("KPI evaluation, performance, or contract event result is incorrect")
        evaluation_rows = [connection.execute("SELECT result, rule_id, snapshot_total_premium_satang, snapshot_new_policy_count FROM monthly_kpi_evaluations WHERE evaluation_id=?", (item,)).fetchone() for item in evaluations]
        if any(row["result"] != "PASS" or row["rule_id"] != ownership.rule_id or row["snapshot_total_premium_satang"] != 1_500_001 or row["snapshot_new_policy_count"] != 6 for row in evaluation_rows):
            raise AssertionError("KPI persisted result does not match the tested strict boundary")
        session_a = create_session(connection, ownership.owner_a)
        ownership.session_ids.append(session_a.session_id)
        session_b = create_session(connection, ownership.owner_b)
        ownership.session_ids.append(session_b.session_id)
        ownership.message_ids.append(append_message(
            connection, session_id=session_a.session_id, access_token=session_a.access_token,
            role="user", content="synthetic session A", request_id=f"{ownership.run_id}-message-a",
        ))
        ownership.message_ids.append(append_message(
            connection, session_id=session_b.session_id, access_token=session_b.access_token,
            role="user", content="synthetic session B", request_id=f"{ownership.run_id}-message-b",
        ))
        valid = validate_lead(LeadDraft(name="Synthetic PostgreSQL lead", occupation="Tester", income=Decimal("45000.00"), income_currency="THB", income_period="monthly_thb", phone="0812345678", product_id="khum-talodcheep-ci-plus", confirmation_status="confirmed"))
        request_id = f"{ownership.run_id}-lead"
        lead_id = save_lead(connection, session_id=session_a.session_id, access_token=session_a.access_token, request_id=request_id, lead=valid)
        ownership.lead_ids.append(lead_id)
        if save_lead(connection, session_id=session_a.session_id, access_token=session_a.access_token, request_id=request_id, lead=valid) != lead_id:
            raise AssertionError("repeated request_id did not remain idempotent")
        try:
            save_lead(connection, session_id=session_a.session_id, access_token=session_a.access_token, request_id=request_id, lead=valid.model_copy(update={"phone": "0812345679"}))
        except ValueError:
            pass
        else:
            raise AssertionError("changed payload reused the same request_id")
        try:
            save_lead(connection, session_id=session_a.session_id, access_token=session_b.access_token, request_id=f"{ownership.run_id}-wrong-owner", lead=valid)
        except PermissionError:
            pass
        else:
            raise AssertionError("wrong owner was allowed to save a lead")
        invalid = ValidatedLead.model_construct(name="Bad", occupation="Tester", income=Decimal("45000"), income_currency="THB", income_period="monthly_thb", phone="garbage", product_id="khum-talodcheep-ci-plus", confirmation_status="confirmed")
        before = int(connection.execute("SELECT COUNT(*) AS count FROM leads").fetchone()["count"])
        try:
            save_lead(connection, session_id=session_a.session_id, access_token=session_a.access_token, request_id=f"{ownership.run_id}-invalid", lead=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid lead was accepted")
        after = int(connection.execute("SELECT COUNT(*) AS count FROM leads").fetchone()["count"])
        if before != after:
            raise AssertionError("invalid lead created a database row")
        connection.close()
        reopened = (
            connect_local(settings.business_db_path)
            if getattr(settings, "database_backend", "local") == "local"
            else connect_from_settings(settings)
        )
        try:
            resumed_a = resume_session(reopened, session_a.session_id, session_a.access_token, expected_owner_id=ownership.owner_a)
            history_a = load_history(reopened, session_id=resumed_a.session_id, access_token=resumed_a.access_token)
            history_b = load_history(reopened, session_id=session_b.session_id, access_token=session_b.access_token)
            if [row["content"] for row in history_a] != ["synthetic session A"] or [row["content"] for row in history_b] != ["synthetic session B"]:
                raise AssertionError("session history did not survive reconnect or isolation failed")
            try:
                resume_session(reopened, session_a.session_id, session_b.access_token, expected_owner_id=ownership.owner_a)
            except PermissionError:
                pass
            else:
                raise AssertionError("wrong owner token resumed the session")
        finally:
            reopened.close()
        return {
            "kpi": {"status": "passed_live", "evaluations": len(evaluations), "contract_events": len(events), "all_results_pass": True, "rule_id_bound": ownership.rule_id},
            "lead": {"status": "passed_live", "validation": True, "ownership": True, "idempotency": True, "payload_conflict_rejected": True},
            "session": {"status": "passed_live", "history_a": 1, "history_b": 1, "reconnect": True, "isolated": True, "wrong_owner_rejected": True},
        }
    finally:
        try:
            connection.close()
        except Exception:
            pass


class CheckpointState(TypedDict):
    value: str


def check_postgres_checkpointer(settings: object, ownership: RunOwnership) -> dict[str, object]:
    from langgraph.graph import END, START, StateGraph
    from insurex.rag.checkpoint import persistent_checkpointer

    builder = StateGraph(CheckpointState)
    builder.add_node("mark", lambda state: {"value": state["value"] + ":ok"})
    builder.add_edge(START, "mark")
    builder.add_edge("mark", END)
    config = {"configurable": {"thread_id": ownership.checkpoint_thread_id}}
    with persistent_checkpointer(settings) as saver:
        graph = builder.compile(checkpointer=saver)
        result = graph.invoke({"value": "synthetic"}, config)
    with persistent_checkpointer(settings) as saver:
        reopened_graph = builder.compile(checkpointer=saver)
        persisted = reopened_graph.get_state(config).values
        if persisted.get("value") != "synthetic:ok":
            raise AssertionError("checkpoint did not survive saver/connection restart")
        reopened_graph.update_state(config, {"value": "synthetic:resumed"})
        resumed = reopened_graph.get_state(config).values
    if result.get("value") != "synthetic:ok" or resumed.get("value") != "synthetic:resumed":
        raise AssertionError("checkpoint resume result is incorrect")
    return {"status": "passed_live", "persisted_after_reopen": True, "resumed_after_reopen": True}


def main() -> int:
    settings = None
    ownership = RunOwnership.create()
    cleanup: dict[str, object] = {"performed": False, "exact_ids_only": True}
    preflight_passed = False
    mutation_started = False
    try:
        require_dependencies()
        if not os.environ.get("INSUREX_ENV_FILE") and (ROOT / ".env.azure").exists():
            os.environ["INSUREX_ENV_FILE"] = str(ROOT / ".env.azure")
        from insurex.config import Settings

        settings = Settings.from_env()
        errors = settings.validate()
        if errors or settings.database_backend != "postgresql":
            raise RuntimeError("Azure PostgreSQL profile is not configured: " + "; ".join(errors or ["DATABASE_BACKEND is not postgresql"]))
        preflight = check_connection_and_schema(settings, initialize_schema=False, require_checkpoint_tables=False)
        # Only after TLS/target verification is DDL allowed.
        schema_base = check_connection_and_schema(settings, initialize_schema=True, require_checkpoint_tables=False)
        from insurex.rag.checkpoint import persistent_checkpointer

        with persistent_checkpointer(settings):
            pass
        schema = check_connection_and_schema(settings, initialize_schema=False, require_checkpoint_tables=True)
        preflight_passed = True
        schema["business_schema"] = schema_base["schema_compatibility"]
        from insurex.db import connect_runtime

        baseline_connection = connect_runtime(settings, initialize_schema=False)
        try:
            baseline = _count_owned_rows(baseline_connection, ownership)
            if any(baseline.values()):
                raise RuntimeError("run-owned identifiers unexpectedly existed before test data creation")
        finally:
            baseline_connection.close()
        mutation_started = True
        business = check_kpi_lead_session(settings, ownership)
        checkpointer = check_postgres_checkpointer(settings, ownership)
        evidence = {
            "status": "tested_live",
            "run_id": ownership.run_id,
            "target": {"database_backend": "postgresql", "tls_and_target": preflight, "schema": schema},
            "business_checks": business,
            "checkpointer": checkpointer,
            "existing_rows_touched": False,
            "existing_rows_evidence": {"run_owned_identifiers_absent_before_insert": True},
            "synthetic_test_rows_created": True,
        }
        output = ROOT / "rag" / "reports" / "demo" / "postgres_integration.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            cleanup = cleanup_test_rows(settings, ownership)
        finally:
            evidence["test_data_cleanup"] = cleanup
        output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "tested_live", "output": str(output), "tls": True}, ensure_ascii=True))
        return 0
    except Exception as exc:
        original_error = redact(exc)
        if settings is not None and cleanup_allowed(
            preflight_passed=preflight_passed,
            mutation_started=mutation_started,
        ):
            try:
                cleanup = cleanup_test_rows(settings, ownership)
            except Exception as cleanup_exc:
                cleanup = {"performed": False, "exact_ids_only": True, "cleanup_error": redact(cleanup_exc)}
        print(json.dumps({"status": "blocked_or_failed_live", "run_id": ownership.run_id, "error_type": type(exc).__name__, "error": original_error[:500], "cleanup": cleanup}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
