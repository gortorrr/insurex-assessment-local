from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
RAG_ROOT = ROOT / "rag"
PRODUCTS_PAGE = "https://www.insurex.co.th/products"

DOCUMENTS = [
    {
        "doc_id": "insurex-khum-talodcheep-ci-plus",
        "product_id": "khum-talodcheep-ci-plus",
        "product_name": "Khum Talodcheep CI Plus",
        "category": "critical_illness",
        "source_page_url": PRODUCTS_PAGE,
        "pdf_url": "https://cdn.scbprotect.co.th/11_SCB-InsureX_Khum-Talodcheep-CI-Plus_90-5_90-10_90-20_180669.pdf",
        "filename": "insurex-khum-talodcheep-ci-plus.pdf",
        "publisher": "InsureX / SCB Protect CDN",
        "source_page_note": "Current InsureX products page; PDF is served from the official SCB Protect CDN hostname.",
    },
    {
        "doc_id": "insurex-khum-talodcheep-plus",
        "product_id": "khum-talodcheep-plus",
        "product_name": "Khum Talodcheep Plus",
        "category": "life",
        "source_page_url": PRODUCTS_PAGE,
        "pdf_url": "https://cdn.scbprotect.co.th/04_SCB-InsureX_Khum-Talodcheep-Plus_180669.pdf",
        "filename": "insurex-khum-talodcheep-plus.pdf",
        "publisher": "InsureX / SCB Protect CDN",
        "source_page_note": "Current InsureX products page; PDF is served from the official SCB Protect CDN hostname.",
    },
    {
        "doc_id": "insurex-khum-aomsook",
        "product_id": "khum-aomsook-25-15",
        "product_name": "Khum Aomsook 25/15",
        "category": "savings",
        "source_page_url": PRODUCTS_PAGE,
        "pdf_url": "https://cdn.scbprotect.co.th/09_SCB-InsureX_Khum-Aomsook_25-15_180669.pdf",
        "filename": "insurex-khum-aomsook-25-15.pdf",
        "publisher": "InsureX / SCB Protect CDN",
        "source_page_note": "Current InsureX products page; PDF is served from the official SCB Protect CDN hostname.",
    },
    {
        "doc_id": "insurex-khum-manjai-sure-term",
        "product_id": "khum-manjai-sure-term",
        "product_name": "Khum Manjai Sure Term",
        "category": "life",
        "source_page_url": PRODUCTS_PAGE,
        "pdf_url": "https://cdn.scbprotect.co.th/06_SCB-InsureX_Khum-Manjai-Sure_Term-5-10-15_180669.pdf",
        "relative_path": "knowledge_base/pdfs/candidates/04-term.pdf",
        "publisher": "InsureX / SCB Protect CDN",
        "source_page_note": "Current InsureX products page; PDF is served from the official SCB Protect CDN hostname.",
    },
    {
        "doc_id": "insurex-whole-life-90-20",
        "product_id": "whole-life-90-20",
        "product_name": "Whole Life 90/20",
        "category": "life",
        "source_page_url": PRODUCTS_PAGE,
        "pdf_url": "https://cdn.scbprotect.co.th/14_SCB-InsureX_Whole-Life_90-20_180669.pdf",
        "relative_path": "knowledge_base/pdfs/candidates/05-wholelife90-20.pdf",
        "publisher": "InsureX / SCB Protect CDN",
        "source_page_note": "Current InsureX products page; PDF is served from the official SCB Protect CDN hostname.",
    },
]


def metadata_date(reader: PdfReader) -> str | None:
    raw = str((reader.metadata or {}).get("/CreationDate", ""))
    if raw.startswith("D:") and len(raw) >= 10:
        return f"{raw[2:6]}-{raw[6:8]}-{raw[8:10]}"
    return None


def build() -> dict:
    retrieved = datetime.now(timezone.utc).isoformat()
    documents = []
    for definition in DOCUMENTS:
        relative_path = definition.get("relative_path")
        if not relative_path:
            relative_path = f"knowledge_base/pdfs/{definition['filename']}"
        path = RAG_ROOT / relative_path
        reader = PdfReader(str(path))
        texts = [page.extract_text() or "" for page in reader.pages]
        extracted = "\n".join(texts)
        private_use = sum(0xE000 <= ord(char) <= 0xF8FF for char in extracted)
        metadata = reader.metadata or {}
        is_historical = "2022" in relative_path
        document = {
            **definition,
            "retrieved_at": retrieved,
            "downloaded_at": retrieved,
            "document_date": metadata_date(reader),
            "document_date_status": "metadata_creation_date_only; printed_date_or_version_not_found",
            "version_note": (
                "Historical brochure created in 2022; do not present as a current offer."
                if is_historical
                else "PDF metadata creation date and filename marker recorded; printed issue/version not verified."
            ),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
            "page_count": len(reader.pages),
            "language": "th",
            "extraction_status": "limited_text_encoding" if private_use else "passed",
            "extraction_quality": {
                "empty_pages": [index + 1 for index, text in enumerate(texts) if not text.strip()],
                "pages_with_text": sum(bool(text.strip()) for text in texts),
                "total_extracted_characters": len(extracted),
                "average_characters_per_nonempty_page": round(
                    len(extracted) / max(1, sum(bool(text.strip()) for text in texts)), 1
                ),
                "text_extraction_method": "pypdf.PdfReader.page.extract_text",
                "metadata": {str(key): str(value) for key, value in metadata.items()},
                "private_use_area_characters": private_use,
                "quality_note": (
                    "Text is available on every page, but some Thai glyphs map to private-use characters; "
                    "treat answers as evidence requiring source-page verification."
                    if private_use
                    else "Text is extractable on every page with no private-use characters observed."
                ),
            },
            "local_path": relative_path,
            "redistribution_note": "Downloaded for local assessment evidence; verify source terms before redistribution.",
            "current_offer_warning": (
                "This corpus contains current InsureX product documents. Do not present any PDF alone "
                "as a current sales offer; confirm current availability, terms and date on the official source page."
            ),
        }
        document.pop("filename", None)
        document.pop("relative_path", None)
        documents.append(document)
    manifest = {
        "status": "verified_insurex_only_corpus",
        "required_document_count": 5,
        "corpus_notes": (
            "Exactly five current InsureX/SCB Protect product documents linked from the official products page are "
            "included. Printed document dates/versions were not verified; metadata creation dates are recorded "
            "separately. All five PDFs have private-use Thai font "
            "mappings, so extraction is usable for local evidence but requires source-page verification before production use."
        ),
        "documents": documents,
    }
    (RAG_ROOT / "knowledge_base" / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps({"documents": len(result["documents"]), "status": result["status"]}))
