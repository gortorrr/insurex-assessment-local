"""Call Gemini through the real LangGraph RAG path for every PDF in the corpus."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from insurex.config import Settings
from insurex.rag.checkpoint import persistent_checkpointer
from insurex.rag.contracts import NO_ANSWER_MESSAGE
from insurex.rag.graph import build_rag_graph
from insurex.rag.runtime import RequestBudget, build_gemini_services


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rag" / "reports" / "demo" / "live_five_pdf_answers.json"


def citation_rows(state):
    return [
        {"doc_id": item.doc_id, "page": item.page_number, "chunk_id": item.chunk_id,
         "source_url": item.source_url}
        for item in state.get("citations", [])
    ]


def main() -> int:
    settings = Settings.from_env()
    if settings.llm_provider != "google_ai_studio" or not settings.google_api_key:
        raise SystemExit("Google AI Studio is not configured")
    budget = RequestBudget(max_requests=24)
    gemini = build_gemini_services(settings, budget=budget)
    cases = [
        ("insurex-khum-talodcheep-ci-plus", "khum-talodcheep-ci-plus", "ซีไอ พลัสคุ้มครองผู้เอาประกันถึงอายุเท่าไร", ["90"]),
        ("insurex-khum-talodcheep-plus", "khum-talodcheep-plus", "คุ้มตลอดชีพ พลัสมีระยะเวลาที่ไม่คุ้มครองกี่วันบ้าง", ["90", "120"]),
        ("insurex-khum-aomsook", "khum-aomsook-25-15", "คุ้มออมสุข 25/15 กำหนดระยะเวลาออมกี่ปี", ["15"]),
        ("insurex-khum-manjai-sure-term", "khum-manjai-sure-term", "คุ้มมั่นใจชัวร์มีระยะเวลาคุ้มครองแบบใด", ["5", "10", "15"]),
        ("insurex-whole-life-90-20", "whole-life-90-20", "ตลอดชีพ 90/20 จ่ายเบี้ยแบบรายเดือนได้หรือไม่", ["รายเดือน"]),
    ]
    results = []
    with persistent_checkpointer(settings) as saver:
        graph = build_rag_graph(gemini.services(), checkpointer=saver)
        for expected_doc, product_id, question, expected_facts in cases:
            state = graph.invoke(
                {"query": question, "active_product_id": product_id, "conversation_history": []},
                {"configurable": {"thread_id": f"five-pdf-{uuid.uuid4().hex}"}},
            )
            citations = citation_rows(state)
            answer = state.get("answer", "")
            facts_present = all(fact in answer for fact in expected_facts)
            used_raw_fallback = answer.startswith("### ข้อมูลที่พบในเอกสารผลิตภัณฑ์")
            results.append({
                "product_id": product_id,
                "question": question,
                "answer": answer,
                "expected_facts": expected_facts,
                "expected_facts_present": facts_present,
                "used_raw_fallback": used_raw_fallback,
                "citations": citations,
                "evidence_status": state.get("evidence_status"),
                "error_code": state.get("error_code"),
                "passed": bool(answer)
                and expected_doc in {row["doc_id"] for row in citations}
                and state.get("error_code") is None
                and facts_present
                and not used_raw_fallback,
            })
        no_answer = graph.invoke(
            {"query": "วิธีเปลี่ยนน้ำมันเครื่องรถยนต์", "active_product_id": None, "conversation_history": []},
            {"configurable": {"thread_id": f"five-pdf-no-answer-{uuid.uuid4().hex}"}},
        )
    no_answer_result = {
        "answer": no_answer.get("answer"),
        "citations": citation_rows(no_answer),
        "error_code": no_answer.get("error_code"),
        "passed": no_answer.get("answer") == NO_ANSWER_MESSAGE
        and citation_rows(no_answer) == []
        and no_answer.get("error_code") is None,
    }
    report = {
        "executed_at": datetime.now(UTC).isoformat(),
        "model": settings.llm_model,
        "provider": "Google AI Studio through LangChain",
        "cases": results,
        "no_answer": no_answer_result,
        "request_count": budget.requests,
        "retry_requests": budget.retry_requests,
        "usage": budget.usage,
        "passed": all(item["passed"] for item in results) and no_answer_result["passed"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "pdf_cases_passed": sum(item["passed"] for item in results),
        "pdf_cases_total": len(results),
        "no_answer_passed": no_answer_result["passed"],
        "requests": budget.requests,
    }, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
