"""Manifest validation for the five-document PDF knowledge base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "doc_id",
    "product_id",
    "product_name",
    "source_page_url",
    "pdf_url",
    "publisher",
    "retrieved_at",
    "sha256",
    "page_count",
    "language",
    "extraction_status",
    "local_path",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(
    manifest: dict[str, Any],
    *,
    root: Path,
    require_exactly_five: bool = True,
) -> list[str]:
    errors: list[str] = []
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        return ["manifest.documents must be a list"]
    if require_exactly_five and len(documents) != 5:
        errors.append(f"expected exactly 5 documents, found {len(documents)}")
    ids: set[str] = set()
    hashes: set[str] = set()
    for index, document in enumerate(documents, start=1):
        missing = REQUIRED_FIELDS - set(document)
        if missing:
            errors.append(f"document {index} missing fields: {sorted(missing)}")
            continue
        if document["doc_id"] in ids:
            errors.append(f"duplicate doc_id: {document['doc_id']}")
        ids.add(document["doc_id"])
        local_path = root / document["local_path"]
        if not local_path.exists():
            errors.append(f"document {document['doc_id']} local file does not exist: {local_path}")
            continue
        if local_path.read_bytes()[:5] != b"%PDF-":
            errors.append(f"document {document['doc_id']} is not a PDF file")
        actual_hash = file_sha256(local_path)
        if actual_hash != document["sha256"]:
            errors.append(f"document {document['doc_id']} sha256 mismatch")
        if actual_hash in hashes:
            errors.append(f"duplicate PDF content hash: {actual_hash}")
        hashes.add(actual_hash)
        if not isinstance(document["page_count"], int) or document["page_count"] < 1:
            errors.append(f"document {document['doc_id']} page_count is not a positive integer")
        corrected_path = document.get("corrected_text_path")
        if not corrected_path:
            errors.append(f"document {document['doc_id']} has no corrected_text_path")
        elif not (root / corrected_path).exists():
            errors.append(f"document {document['doc_id']} corrected text does not exist: {corrected_path}")
    return errors
