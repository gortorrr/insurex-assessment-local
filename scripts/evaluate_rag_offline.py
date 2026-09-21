"""Evaluate retrieval/gating against a fixed 10-dev/20-held-out case set."""

from __future__ import annotations

import json
import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insurex.config import Settings
from insurex.rag.contracts import assess_candidates
from insurex.rag.graph import build_standalone_query
from insurex.rag.index import index_fingerprint, verify_index_metadata
from insurex.rag.manifest import file_sha256
from insurex.rag.runtime import _assess_local, _local_embedding_path, _retrieve_factory


CASES = ROOT / "rag/eval/rag_cases.jsonl"
OUT = ROOT / "rag/reports/demo/rag_retrieval_evaluation.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    cases_path = args.cases if args.cases.is_absolute() else ROOT / args.cases
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    settings = Settings.from_env()
    retrieve = _retrieve_factory(settings)
    embedding_path = _local_embedding_path(settings)
    fingerprint, embedding_revision, index_config = index_fingerprint(
        settings.kb_manifest_path,
        root=ROOT / "rag",
        embedding_model=embedding_path,
        max_characters=1800,
        overlap=240,
        max_tokens=400,
    )
    actual_index = verify_index_metadata(
        settings.chroma_path,
        expected_fingerprint=fingerprint,
    )
    rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in rows:
        query = build_standalone_query(case["query"], case.get("history"))
        chunks = retrieve(query, case.get("product_id"))
        assessment = _assess_local(query, chunks)
        structural = assess_candidates(chunks, evidence_status=assessment, query=query, active_product_id=case.get("product_id"))
        expected_doc = case.get("expected_doc_id")
        doc_hit = expected_doc is None or any(chunk.doc_id == expected_doc for chunk in chunks)
        page_hit = expected_doc is None or any(chunk.doc_id == expected_doc and chunk.page_number in case.get("expected_pages", []) for chunk in chunks)
        if case["expected_type"] == "answerable":
            passed = doc_hit and page_hit and assessment == "sufficient"
        elif case["expected_type"] == "ambiguous":
            passed = structural.status == "ambiguous"
        elif case["expected_type"] == "no_answer":
            # Ambiguity is a separate outcome and must never count as a
            # successful no-answer result.
            passed = assessment == "insufficient" and structural.status == "insufficient"
        else:
            passed = False
        results.append({
            "case_id": case["case_id"], "split": case["split"], "expected_type": case["expected_type"],
            "retrieved_doc_ids": [chunk.doc_id for chunk in chunks],
            "retrieved_pages": [{"doc_id": chunk.doc_id, "page": chunk.page_number, "score": chunk.score} for chunk in chunks],
            "assessment": assessment, "structural_status": structural.status,
            "expected_doc_hit": doc_hit, "expected_page_hit": page_hit, "passed": passed,
        })
    dev = [row for row in results if row["split"] == "dev"]
    heldout = [row for row in results if row["split"].startswith("held_out")]
    payload = {
        "status": "tested_offline",
        "executed_at": datetime.now(UTC).isoformat(),
        "case_manifest": str(cases_path.relative_to(ROOT)),
        "case_manifest_sha256": file_sha256(cases_path),
        "corpus_manifest_sha256": file_sha256(settings.kb_manifest_path),
        "embedding_model": embedding_path,
        "embedding_revision": embedding_revision,
        "embedding_manifest_sha256": index_config["embedding_manifest_sha256"],
        "extraction_revision": index_config["extraction_revision"],
        "extraction_verification_sha256": index_config["extraction_verification_sha256"],
        "index_fingerprint": fingerprint,
        "chunk_config": {key: index_config[key] for key in ("max_characters", "overlap", "max_tokens", "chunking_revision")},
        "chroma_count": actual_index["actual_count"],
        "actual_index_ids_sha256": actual_index["actual_ids_sha256"],
        "actual_index_metadata": actual_index["metadata"],
        "total_cases": len(results), "dev_cases": len(dev), "held_out_cases": len(heldout),
        "dev_passed": sum(row["passed"] for row in dev), "held_out_passed": sum(row["passed"] for row in heldout),
        "results": results,
        "live_llm_calls": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "total_cases", "dev_cases", "held_out_cases", "dev_passed", "held_out_passed", "live_llm_calls")}, ensure_ascii=False))
    complete = bool(payload["total_cases"]) and bool(payload["held_out_cases"] or payload["dev_cases"])
    return 0 if complete and payload["dev_passed"] == payload["dev_cases"] and payload["held_out_passed"] == payload["held_out_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
