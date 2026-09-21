"""Scrape official InsureX product PDF candidates from the public products page."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PRODUCTS_URL = "https://www.insurex.co.th/products?tag_ins="
OUTPUT = ROOT / "rag" / "knowledge_base" / "insurex_product_candidates.json"
DOWNLOAD_ROOT = ROOT / "rag" / "knowledge_base" / "pdfs" / "candidates"
USER_AGENT = "InsureX assessment research; contact via official site"


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def extract_candidates(html: bytes) -> list[dict[str, str]]:
    page = html.decode("utf-8")
    match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page, re.S)
    if not match:
        raise RuntimeError("InsureX page did not contain __NEXT_DATA__")
    data = json.loads(match.group(1))
    candidates: list[dict[str, str]] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            url = value.get("brochureDownloadLink")
            if isinstance(url, str) and url.lower().endswith(".pdf"):
                parsed = urlsplit(url)
                if parsed.hostname == "cdn.scbprotect.co.th" and "insurex" in url.lower():
                    encoded_url = urlunsplit(
                        (parsed.scheme, parsed.netloc, quote(parsed.path), parsed.query, parsed.fragment)
                    )
                    candidates.append(
                        {
                            "product_name": str(value.get("name") or ""),
                            "product_path": str(value.get("path") or ""),
                            "category": str(value.get("type") or ""),
                            "pdf_url": encoded_url,
                            "source_page_url": PRODUCTS_URL,
                        }
                    )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    unique: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        unique[candidate["pdf_url"]] = candidate
    return list(unique.values())


def metadata_date(reader: PdfReader) -> str | None:
    raw = str((reader.metadata or {}).get("/CreationDate", ""))
    if raw.startswith("D:") and len(raw) >= 10:
        return f"{raw[2:6]}-{raw[6:8]}-{raw[8:10]}"
    return None


def inspect_pdf(path: Path) -> dict[str, object]:
    reader = PdfReader(str(path))
    texts = [page.extract_text() or "" for page in reader.pages]
    extracted = "\n".join(texts)
    private_use = sum(0xE000 <= ord(char) <= 0xF8FF for char in extracted)
    if not any(text.strip() for text in texts):
        status = "image_only"
    elif private_use:
        status = "limited_text_encoding"
    else:
        status = "passed"
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "page_count": len(reader.pages),
        "document_date": metadata_date(reader),
        "document_date_status": "metadata_creation_date_only; printed_date_or_version_not_found",
        "extraction_status": status,
        "extraction_quality": {
            "empty_pages": [index + 1 for index, text in enumerate(texts) if not text.strip()],
            "pages_with_text": sum(bool(text.strip()) for text in texts),
            "total_extracted_characters": len(extracted),
            "private_use_area_characters": private_use,
            "text_extraction_method": "pypdf.PdfReader.page.extract_text",
            "metadata": {str(key): str(value) for key, value in (reader.metadata or {}).items()},
        },
    }


def safe_filename(index: int, candidate: dict[str, str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", candidate["product_path"].lower()).strip("-")
    return f"{index:02d}-{slug or 'product'}.pdf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download and inspect every official candidate")
    args = parser.parse_args()
    observed_at = datetime.now(timezone.utc).isoformat()
    candidates = extract_candidates(fetch(PRODUCTS_URL))
    records: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        record: dict[str, object] = {**candidate, "observed_at": observed_at, "publisher": "InsureX / SCB Protect CDN"}
        if args.download:
            DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
            local = DOWNLOAD_ROOT / safe_filename(index, candidate)
            try:
                local.write_bytes(fetch(candidate["pdf_url"]))
                record.update(inspect_pdf(local))
                record["downloaded_at"] = datetime.now(timezone.utc).isoformat()
                record["local_path"] = str(local.relative_to(ROOT)).replace("\\", "/")
            except Exception as exc:
                record.update({"extraction_status": "download_or_extract_failed", "error_type": type(exc).__name__, "error": str(exc)[:300]})
        records.append(record)
    manifest = {
        "status": "scraped_official_insurex_candidates",
        "source_page_url": PRODUCTS_URL,
        "observed_at": observed_at,
        "selection_policy": "Only .pdf links on cdn.scbprotect.co.th containing InsureX in the URL are included; active corpus selection remains separate.",
        "candidate_count": len(records),
        "documents": records,
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "candidate_count": len(records), "downloaded": args.download}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
