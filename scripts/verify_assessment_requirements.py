"""Run reproducible requirement checks against isolated local application stores.

The script intentionally uses synthetic lead data. It writes the two requested
demo leads to temporary SQLite databases, while the JSON report contains only
validation flags and identifiers (no names, phone numbers, or income). Existing
demo or user databases are never read or modified.
"""

from __future__ import annotations

import importlib.metadata
import json
import sys
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insurex.auth import register, setup
from insurex.config import Settings
from insurex.db import connect_assistant_local
from insurex.rag.checkpoint import persistent_checkpointer
from insurex.rag.graph import build_rag_graph
from insurex.rag.index import load_index_metadata, retrieve_from_chroma, verify_index_metadata
from insurex.rag.runtime import build_offline_services
from insurex.session import append_message, get_or_create_customer_session, load_history, resume_session
from rag.ui.app import _run_turn_job


OUT = ROOT / "rag" / "reports" / "demo" / "assessment_requirements_current.json"


def run_turn(settings: Settings, session, catalog: set[str], text: str, request_id: str):
    connection = connect_assistant_local(settings.assistant_db_path or settings.business_db_path)
    try:
        history = load_history(
            connection,
            session_id=session.session_id,
            access_token=session.access_token,
            limit=12,
            product_only=True,
        )
        append_message(
            connection,
            session_id=session.session_id,
            access_token=session.access_token,
            role="user",
            content=text,
            request_id=request_id,
        )
    finally:
        connection.close()
    events: list[dict[str, str]] = []
    result = _run_turn_job(
        settings,
        session_id=session.session_id,
        access_token=session.access_token,
        owner_id=session.owner_id,
        question=text,
        request_id=request_id,
        history=history,
        product_ids=catalog,
        mode="Offline evidence extract",
        thread_id=session.thread_id,
        cancel_event=Event(),
        runtime_events=events,
    )
    if result.get("error_code"):
        raise AssertionError(f"turn failed: {result['error_code']}")
    return result, events


def run_checks(settings: Settings) -> int:
    if settings.database_backend != "local":
        raise SystemExit("This local assessment evidence requires DATABASE_BACKEND=local")
    manifest = json.loads(settings.kb_manifest_path.read_text(encoding="utf-8"))
    catalog = {item["product_id"] for item in manifest["documents"]}
    docs = {item["product_id"]: item["doc_id"] for item in manifest["documents"]}

    services = build_offline_services(settings)
    graph = build_rag_graph(services)
    drawable = graph.get_graph()
    node_names = sorted(drawable.nodes)
    edges = sorted({(edge.source, edge.target) for edge in drawable.edges})
    expected_nodes = {
        "prepare_turn", "route_intent", "resolve_query", "retrieve",
        "assess_evidence", "rewrite_query_once", "generate_grounded_answer",
        "validate_answer_citations", "collect_lead", "fallback",
    }

    recorded_index_meta = load_index_metadata(settings.chroma_path)
    verified_index = verify_index_metadata(
        settings.chroma_path,
        expected_fingerprint=recorded_index_meta["index_fingerprint"],
        expected_count=recorded_index_meta["chunk_count"],
    )
    retrieval_cases = [
        ("khum-talodcheep-ci-plus", "ซีไอ พลัสคุ้มครองถึงอายุเท่าไร"),
        ("khum-talodcheep-plus", "คุ้มตลอดชีพ พลัสคุ้มครองถึงอายุเท่าไร"),
        ("khum-aomsook-25-15", "คุ้มออมสุข 25/15 ออมกี่ปี"),
        ("khum-manjai-sure-term", "คุ้มมั่นใจชัวร์มีระยะเวลาคุ้มครองแบบใด"),
        ("whole-life-90-20", "ตลอดชีพ 90/20 จ่ายเบี้ยกี่ปี"),
    ]
    retrieval_results = []
    for product_id, query in retrieval_cases:
        chunks = retrieve_from_chroma(
            query,
            chroma_path=settings.chroma_path,
            product_id=product_id,
            top_k=5,
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
            retrieval_mode=settings.retrieval_mode,
        )
        ids = sorted({chunk.doc_id for chunk in chunks})
        retrieval_results.append({
            "product_id": product_id,
            "expected_doc_id": docs[product_id],
            "retrieved_doc_ids": ids,
            "chunk_count": len(chunks),
            "passed": docs[product_id] in ids,
        })

    connection = connect_assistant_local(settings.assistant_db_path or settings.business_db_path)
    try:
        accounts = {
            row["username"]: row["owner_id"]
            for row in connection.execute(
                "SELECT username, owner_id FROM user_accounts WHERE username IN (?, ?)",
                ("demo-agent-a", "demo-agent-b"),
            ).fetchall()
        }
        if set(accounts) != {"demo-agent-a", "demo-agent-b"}:
            raise AssertionError("demo-agent-a and demo-agent-b must exist")
        sessions = {
            username: get_or_create_customer_session(connection, owner)
            for username, owner in accounts.items()
        }
    finally:
        connection.close()

    run_tag = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    synthetic = {
        "demo-agent-a": ("ทดสอบ เอ", "นักวิเคราะห์", "0819000001", "คุ้มออมสุข 25/15", "45000"),
        "demo-agent-b": ("ทดสอบ บี", "ที่ปรึกษาการขาย", "0819000002", "ตลอดชีพ 90/20", "52000"),
    }
    lead_results = []
    all_runtime_events = []
    for username, session in sessions.items():
        name, occupation, phone, product, income = synthetic[username]
        steps = [
            f"สนใจสมัคร {product}",
            f"ชื่อ {name} อาชีพ {occupation} รายได้ {income} บาทต่อเดือน เบอร์ {phone}",
        ]
        last = None
        for index, text in enumerate(steps, 1):
            last, events = run_turn(
                settings,
                session,
                catalog,
                text,
                f"assessment-{run_tag}-{username}-{index}-{uuid.uuid4().hex[:8]}",
            )
            all_runtime_events.extend(events)
        if not last or last.get("lead_saved_id") is None:
            raise AssertionError(
                f"lead was not saved for {username}; missing={last.get('lead_missing') if last else None}; "
                f"error={last.get('error_code') if last else None}; answer={last.get('answer') if last else None}; "
                f"stages={[event['stage'] for event in events]}"
            )
        lead_results.append({
            "username": username,
            "session_id": session.session_id,
            "thread_id": session.thread_id,
            "lead_id": last["lead_saved_id"],
            "structured_fields_complete": last.get("lead_missing") == [],
            "saved": True,
        })

    # Exercise the graph's controlled no-answer branch through the same worker.
    no_answer_session = sessions["demo-agent-a"]
    no_answer, no_answer_events = run_turn(
        settings,
        no_answer_session,
        catalog,
        "วิธีซ่อมเครื่องยนต์รถยนต์หลังน้ำท่วม",
        f"assessment-{run_tag}-no-answer",
    )
    all_runtime_events.extend(no_answer_events)

    connection = connect_assistant_local(settings.assistant_db_path or settings.business_db_path)
    try:
        persisted = []
        for item in lead_results:
            row = connection.execute(
                """
                SELECT l.lead_id, l.owner_id, l.name, l.occupation, l.income_thb,
                       l.income_period, l.phone_normalized
                FROM leads l
                WHERE l.lead_id=?
                """,
                (item["lead_id"],),
            ).fetchone()
            expected = synthetic[item["username"]]
            persisted.append({
                "username": item["username"],
                "lead_id": item["lead_id"],
                "owner_matches": row["owner_id"] == accounts[item["username"]],
                "name_matches": row["name"] == expected[0],
                "occupation_matches": row["occupation"] == expected[1],
                "income_matches": row["income_thb"] == int(expected[4]),
                "income_period_matches": row["income_period"] == "monthly_thb",
                "phone_matches": row["phone_normalized"] == expected[2],
            })
        cross_owner_blocked = False
        try:
            resume_session(
                connection,
                sessions["demo-agent-a"].session_id,
                sessions["demo-agent-b"].access_token,
                expected_owner_id=accounts["demo-agent-b"],
            )
        except PermissionError:
            cross_owner_blocked = True
    finally:
        connection.close()

    with persistent_checkpointer(settings) as saver:
        checkpoint_a = saver.get({"configurable": {"thread_id": sessions["demo-agent-a"].thread_id}})
        checkpoint_b = saver.get({"configurable": {"thread_id": sessions["demo-agent-b"].thread_id}})

    # Ensure the operational UI log has no synthetic PII.
    event_text = json.dumps(all_runtime_events, ensure_ascii=False)
    pii_safe = all(value not in event_text for values in synthetic.values() for value in values[:3])
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "environment": {
            "langchain": importlib.metadata.version("langchain"),
            "langgraph": importlib.metadata.version("langgraph"),
            "chromadb": importlib.metadata.version("chromadb"),
            "assistant_database": "isolated_temporary_sqlite",
            "analytics_database": str(settings.analytics_db_path),
        },
        "workflow": {
            "uses_state_graph": True,
            "nodes": node_names,
            "edges": edges,
            "all_expected_nodes_present": expected_nodes.issubset(node_names),
            "bounded_retrieval_cycle_present": ("rewrite_query_once", "retrieve") in edges,
            "persistent_sqlite_checkpointer": True,
        },
        "vector_database": {
            "backend": "ChromaDB",
            "retrieval_mode": settings.retrieval_mode,
            "collection": recorded_index_meta["collection_name"],
            "chunk_count": verified_index["actual_count"],
            "document_count": len(manifest["documents"]),
            "all_five_documents_retrieved": all(item["passed"] for item in retrieval_results),
            "cases": retrieval_results,
        },
        "no_answer": {
            "passed": no_answer.get("citations") == [] and bool(no_answer.get("answer")),
            "citations": no_answer.get("citations"),
            "error_code": no_answer.get("error_code"),
            "fallback_node_seen": any(event["stage"] == "fallback" for event in no_answer_events),
        },
        "lead_collection": {
            "implementation": "LangGraph callable Tooling path; not an MCP protocol server",
            "schema": ["name", "occupation", "income", "phone", "product_id"],
            "accounts": lead_results,
            "sqlite_integrity": persisted,
            "all_passed": all(all(value for key, value in row.items() if key.endswith("_matches")) for row in persisted),
        },
        "session_management": {
            "distinct_session_ids": sessions["demo-agent-a"].session_id != sessions["demo-agent-b"].session_id,
            "distinct_thread_ids": sessions["demo-agent-a"].thread_id != sessions["demo-agent-b"].thread_id,
            "cross_owner_access_blocked": cross_owner_blocked,
            "checkpoint_a_exists": checkpoint_a is not None,
            "checkpoint_b_exists": checkpoint_b is not None,
        },
        "runtime_log": {
            "event_count": len(all_runtime_events),
            "pii_safe": pii_safe,
            "stages_seen": sorted({event["stage"] for event in all_runtime_events}),
        },
    }
    report["passed"] = all([
        report["workflow"]["all_expected_nodes_present"],
        report["workflow"]["bounded_retrieval_cycle_present"],
        report["vector_database"]["all_five_documents_retrieved"],
        report["no_answer"]["passed"],
        report["lead_collection"]["all_passed"],
        all(report["session_management"].values()),
        report["runtime_log"]["pii_safe"],
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "chroma_chunks": report["vector_database"]["chunk_count"],
        "five_documents": report["vector_database"]["all_five_documents_retrieved"],
        "lead_ids": [item["lead_id"] for item in lead_results],
        "sessions_isolated": report["session_management"]["cross_owner_access_blocked"],
        "runtime_events": report["runtime_log"]["event_count"],
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


def main() -> int:
    base = Settings.from_env()
    if base.database_backend != "local":
        raise SystemExit("This local assessment evidence requires DATABASE_BACKEND=local")
    with TemporaryDirectory(prefix="insurex-assessment-") as temporary:
        runtime = Path(temporary)
        assistant_db = runtime / "assistant.sqlite"
        settings = replace(
            base,
            business_db_path=assistant_db,
            assistant_db_path=assistant_db,
            checkpoint_db_path=runtime / "checkpoints.sqlite",
            trace_path=runtime / "traces",
        )
        connection = connect_assistant_local(assistant_db)
        try:
            setup(connection)
            register(connection, "demo-agent-a", "Demo-Agent-A-2026!", account_role="customer")
            register(connection, "demo-agent-b", "Demo-Agent-B-2026!", account_role="customer")
        finally:
            connection.close()
        return run_checks(settings)


if __name__ == "__main__":
    raise SystemExit(main())

