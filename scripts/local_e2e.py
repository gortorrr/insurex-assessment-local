"""Run the local deterministic end-to-end workflow with synthetic leads only."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from insurex.config import Settings
from insurex.db import connect_local
from insurex.rag.checkpoint import persistent_checkpointer
from insurex.rag.graph import build_rag_graph
from insurex.rag.runtime import LeadTurnHandler, build_offline_services
from insurex.session import create_session, resume_session


ROOT = Path(".")
EVIDENCE = ROOT / "rag" / "reports" / "demo" / "local_e2e_evidence.json"
PRODUCT_ID = "khum-talodcheep-ci-plus"


def masked_phone(value: str) -> str:
    return value[:3] + "•••••" + value[-2:]


def main() -> None:
    runtime_dir = ROOT / "tmp" / "local_e2e"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for path in (runtime_dir / "local_e2e_shared.sqlite", runtime_dir / "local_e2e_checkpoints.sqlite"):
        if path.exists():
            path.unlink()
    settings = Settings.from_env()
    settings = Settings(**{**settings.__dict__, "checkpoint_db_path": runtime_dir / "local_e2e_checkpoints.sqlite"})
    catalog = {d["product_id"] for d in json.loads((ROOT / "rag" / "knowledge_base" / "manifest.json").read_text(encoding="utf-8"))["documents"]}
    conn_a = connect_local(runtime_dir / "local_e2e_shared.sqlite")
    conn_b = connect_local(runtime_dir / "local_e2e_shared.sqlite")
    session_a = create_session(conn_a, "synthetic-owner-a")
    session_b = create_session(conn_b, "synthetic-owner-b")
    handler_a = LeadTurnHandler(conn_a, session_a.session_id, session_a.access_token, catalog, "local-e2e-a")
    handler_b = LeadTurnHandler(conn_b, session_b.session_id, session_b.access_token, catalog, "local-e2e-b")
    evidence: dict[str, object] = {"status": "tested_offline", "synthetic_only": True, "turns": [], "session_isolation": {}}

    with persistent_checkpointer(settings) as saver:
        graph_rag = build_rag_graph(build_offline_services(settings), checkpointer=saver)
        rag_cfg = {"configurable": {"thread_id": "offline-rag-examples"}}
        answerable = graph_rag.invoke({"query": "Khum Talodcheep CI Plus coverage", "active_product_id": PRODUCT_ID}, rag_cfg)
        unanswerable = graph_rag.invoke({"query": "สูตรทำผัดไทย", "active_product_id": PRODUCT_ID}, rag_cfg)
        evidence["rag_examples"] = {
            "answerable": {
                "answer_preview": str(answerable.get("answer", ""))[:240],
                "citations": [{"doc_id": c.doc_id, "page": c.page_number, "chunk_id": c.chunk_id, "source_url": c.source_url} for c in answerable.get("citations", [])],
                "status": "offline_evidence_extract",
            },
            "unanswerable": {
                "answer": unanswerable.get("answer"),
                "citations": [{"doc_id": c.doc_id, "page": c.page_number, "chunk_id": c.chunk_id, "source_url": c.source_url} for c in unanswerable.get("citations", [])],
                "status": "offline_no_answer_gate",
                "evidence_status": unanswerable.get("evidence_status"),
                "error_code": unanswerable.get("error_code"),
            },
        }
        graph_a = build_rag_graph(build_offline_services(settings, lead_turn=handler_a), checkpointer=saver)
        graph_b = build_rag_graph(build_offline_services(settings, lead_turn=handler_b), checkpointer=saver)
        cfg_a = {"configurable": {"thread_id": session_a.thread_id}}
        cfg_b = {"configurable": {"thread_id": session_b.thread_id}}

        def ask(graph, cfg, sid: str, text: str) -> dict[str, object]:
            state = graph.invoke({"session_id": sid, "query": text, "active_product_id": PRODUCT_ID}, cfg)
            row = {
                "session": "A" if sid == session_a.session_id else "B",
                "query": text,
                "answer": state.get("answer"),
                "intent": state.get("intent"),
                "missing": state.get("lead_missing"),
                "saved_id": state.get("lead_saved_id"),
                "citation_count": len(state.get("citations", [])),
                "citation_preview": [
                    {"doc_id": c.doc_id, "page": c.page_number, "source_url": c.source_url}
                    for c in state.get("citations", [])[:1]
                ],
            }
            evidence["turns"].append(row)  # type: ignore[index]
            return row

        # A/B are interleaved on the same business DB with separate threads.
        ask(graph_a, cfg_a, session_a.session_id, "สนใจสมัครประกัน Khum Talodcheep CI Plus")
        ask(graph_b, cfg_b, session_b.session_id, "สนใจสมัครประกัน Khum Talodcheep CI Plus")
        ask(graph_a, cfg_a, session_a.session_id, "ชื่อ เอก อาชีพ โปรแกรมเมอร์ รายได้ 30000 บาทต่อเดือน")
        ask(graph_b, cfg_b, session_b.session_id, "ชื่อ บี อาชีพ ครู รายได้ 45000 บาทต่อเดือน")

    # Re-open the same persisted checkpointer and database handles.
    conn_a.close()
    conn_b.close()
    conn_a = connect_local(ROOT / "db" / "local_e2e_shared.sqlite")
    conn_b = connect_local(ROOT / "db" / "local_e2e_shared.sqlite")
    session_a = resume_session(conn_a, session_a.session_id, session_a.access_token)
    session_b = resume_session(conn_b, session_b.session_id, session_b.access_token)
    handler_a = LeadTurnHandler(conn_a, session_a.session_id, session_a.access_token, catalog, "local-e2e-a")
    handler_b = LeadTurnHandler(conn_b, session_b.session_id, session_b.access_token, catalog, "local-e2e-b")
    with persistent_checkpointer(settings) as saver:
        graph_a = build_rag_graph(build_offline_services(settings, lead_turn=handler_a), checkpointer=saver)
        graph_b = build_rag_graph(build_offline_services(settings, lead_turn=handler_b), checkpointer=saver)
        after_restart_a = graph_a.invoke({"session_id": session_a.session_id, "query": "เบอร์โทร 0812345678", "active_product_id": PRODUCT_ID}, {"configurable": {"thread_id": session_a.thread_id}})
        after_restart_b = graph_b.invoke({"session_id": session_b.session_id, "query": "ยังไม่มีเบอร์โทร", "active_product_id": PRODUCT_ID}, {"configurable": {"thread_id": session_b.thread_id}})
        evidence["restart"] = {
            "A_missing_after_restart": after_restart_a.get("lead_missing"),
            "B_missing_after_restart": after_restart_b.get("lead_missing"),
            "A_saved_id": after_restart_a.get("lead_saved_id"),
            "B_saved_id": after_restart_b.get("lead_saved_id"),
        }

    query = "SELECT lead_id, name, occupation, income_thb, income_period, product_id, phone_normalized FROM leads WHERE owner_id=?"
    leads_a = [dict(row) for row in conn_a.execute(query, (session_a.owner_id,)).fetchall()]
    leads_b = [dict(row) for row in conn_b.execute(query, (session_b.owner_id,)).fetchall()]
    assert len(leads_a) == 1 and not leads_b, "account profiles leaked or did not persist the completed lead"
    assert leads_a[0]["name"] == "เอก" and "phone" in after_restart_b["lead_missing"]
    assert answerable.get("citations") and not answerable.get("error_code")
    assert unanswerable.get("evidence_status") == "insufficient" and not unanswerable.get("citations")
    evidence["shared_business_database"] = True
    for row in leads_a + leads_b:
        row["phone_normalized"] = masked_phone(row["phone_normalized"])
    evidence["session_isolation"] = {"A_leads": leads_a, "B_leads": leads_b, "A_thread": session_a.thread_id, "B_thread": session_b.thread_id, "threads_differ": session_a.thread_id != session_b.thread_id}
    evidence["chroma_count"] = chromadb.PersistentClient(path=str(settings.chroma_path)).get_collection("insurex_kb").count()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_path": str(EVIDENCE), "chroma_count": evidence["chroma_count"], "A_leads": len(leads_a), "B_leads": len(leads_b)}, ensure_ascii=False))
    conn_a.close()
    conn_b.close()


if __name__ == "__main__":
    main()
