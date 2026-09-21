"""Finite Gemini smoke test with assertions and reproducible corpus evidence."""

from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from insurex.config import Settings
from insurex.rag.checkpoint import persistent_checkpointer
from insurex.rag.contracts import NO_ANSWER_MESSAGE
from insurex.rag.graph import build_rag_graph
from insurex.rag.manifest import file_sha256
from insurex.rag.runtime import RequestBudget, build_gemini_services


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "rag/reports/demo/live_gemini_smoke.json"
PRODUCT_ID = "khum-talodcheep-ci-plus"


def corpus_fingerprint(settings: Settings) -> dict[str, str | None]:
    manifest = settings.kb_manifest_path
    model_manifest = ROOT / "rag/knowledge_base/embedding_model_manifest.json"
    config = {
        "manifest_sha256": file_sha256(manifest),
        "embedding_manifest_sha256": file_sha256(model_manifest),
        "embedding_model": settings.embedding_model,
        "chroma_path": str(settings.chroma_path),
        "product_id": PRODUCT_ID,
        "retrieval_top_k": "5",
        "chunking": "max_characters=1800,overlap=240,max_tokens=400,revision=paragraph-boundary-v4",
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    return {**config, "fingerprint": digest}


def citation_payload(state: dict) -> list[dict[str, object]]:
    return [
        {"doc_id": item.doc_id, "page": item.page_number, "chunk_id": item.chunk_id, "source_url": item.source_url}
        for item in state.get("citations", [])
    ]


def write_evidence(payload: dict[str, object]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    run_id = uuid.uuid4().hex[:16]
    started_at = datetime.now(UTC).isoformat()
    settings = Settings.from_env()
    base: dict[str, object] = {
        "run_id": run_id,
        "started_at": started_at,
        "model": settings.llm_model,
        "executed": False,
        "passed": False,
        "failed": False,
        "skipped": False,
        "requests": 0,
        "failed_attempts": 0,
        "usage": [],
        "corpus": corpus_fingerprint(settings),
    }
    if not settings.google_api_key or not settings.llm_model:
        base.update({"status": "skipped", "skipped": True, "reason": "GOOGLE_API_KEY or LLM_MODEL is missing"})
        write_evidence(base)
        print(json.dumps({"status": "skipped", "requests": 0}, ensure_ascii=True))
        return 3
    # The structured response includes answer text plus a full PDF citation
    # payload. 256 tokens can truncate the JSON before it can be parsed.
    # Keep the smoke finite while allowing the contract response to close.
    settings = Settings(**{**settings.__dict__, "max_retrieval_retries": 0, "llm_max_output_tokens": min(settings.llm_max_output_tokens, 512)})
    budget = RequestBudget(max_requests=10)
    try:
        gemini = build_gemini_services(settings, budget=budget)
        with persistent_checkpointer(settings) as saver:
            graph = build_rag_graph(gemini.services(), checkpointer=saver)
            config = {"configurable": {"thread_id": f"live-gemini-smoke-{run_id}"}}
            answerable = graph.invoke({"query": "อายุ 90 ปี", "active_product_id": PRODUCT_ID}, config)
            no_answer = graph.invoke({"query": "สูตรทำผัดไทย", "active_product_id": PRODUCT_ID}, config)
        answerable_citations = citation_payload(answerable)
        no_answer_citations = citation_payload(no_answer)
        assertions = {
            "answerable_has_text": bool(str(answerable.get("answer", "")).strip()),
            "answerable_has_citations": bool(answerable_citations),
            "answerable_citations_have_pdf_urls": all(str(item["source_url"]).lower().endswith(".pdf") for item in answerable_citations),
            "no_answer_is_controlled": no_answer.get("answer") == NO_ANSWER_MESSAGE,
            "no_answer_has_no_citations": no_answer_citations == [],
            "no_answer_is_not_service_error": no_answer.get("error_code") is None,
        }
        passed = all(assertions.values())
        base.update({
            "status": "passed" if passed else "failed",
            "executed": True,
            "passed": passed,
            "failed": not passed,
            "assertions": assertions,
            "answerable": {"answer": str(answerable.get("answer", ""))[:1200], "citations": answerable_citations, "evidence_status": answerable.get("evidence_status")},
            "no_answer": {"answer": no_answer.get("answer"), "citations": no_answer_citations, "evidence_status": no_answer.get("evidence_status"), "error_code": no_answer.get("error_code")},
        })
        base["requests"] = budget.requests
        base["failed_attempts"] = budget.retry_requests
        base["usage"] = budget.usage
        write_evidence(base)
        print(json.dumps({"status": base["status"], "model": settings.llm_model, "requests": budget.requests, "failed_attempts": budget.retry_requests, "usage": budget.usage}, ensure_ascii=False))
        return 0 if passed else 1
    except Exception as exc:
        message = str(exc).replace(settings.google_api_key, "[REDACTED]")
        lowered = message.lower()
        status = "blocked_quota_or_billing" if any(term in lowered for term in ("quota", "billing", "resource exhausted", "payment required")) else "failed"
        base.update({"status": status, "executed": True, "failed": True, "error_type": type(exc).__name__, "error": message[:300], "requests": budget.requests, "failed_attempts": budget.retry_requests, "usage": budget.usage})
        write_evidence(base)
        print(json.dumps({"status": status, "error_type": type(exc).__name__, "requests": budget.requests, "failed_attempts": budget.retry_requests}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
