"""Page-aware PDF extraction and deterministic chunk metadata.

The pypdf import is lazy. No PDF is read in the offline baseline because the
five-document corpus has not been approved or downloaded.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from insurex.rag.manifest import file_sha256


CHUNKING_REVISION = "paragraph-boundary-v4"


class PdfDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PageText:
    doc_id: str
    page_number: int
    text: str
    content_hash: str


def load_verified_pages(text_path: Path, *, doc_id: str, expected_pdf_sha256: str) -> list[PageText]:
    """Load page text repaired from a verified PDF mapping artifact.

    Generation/indexing must use this path for the active corpus. Raw pypdf
    extraction remains available through ``extract_pages`` for audit only.
    """

    import json

    if not text_path.exists():
        raise ValueError(f"verified extraction artifact is missing: {text_path}")
    payload = json.loads(text_path.read_text(encoding="utf-8"))
    if payload.get("doc_id") != doc_id or payload.get("source_pdf_sha256") != expected_pdf_sha256:
        raise ValueError(f"verified extraction provenance mismatch for {doc_id}")
    if payload.get("mapping_revision") != "sukhumvitset-pua-thai-v1":
        raise ValueError(f"unknown extraction mapping revision for {doc_id}")
    pages: list[PageText] = []
    for item in payload.get("pages", []):
        text = str(item.get("text", "")).replace("\u00a0", " ").strip()
        if not text or not item.get("verified"):
            raise ValueError(f"unverified or empty corrected text for {doc_id} p{item.get('page_number')}")
        if any(0xE000 <= ord(char) <= 0xF8FF for char in text):
            raise ValueError(f"corrected text still contains private-use glyphs for {doc_id} p{item.get('page_number')}")
        import hashlib

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash != item.get("sha256"):
            raise ValueError(f"corrected text hash mismatch for {doc_id} p{item.get('page_number')}")
        pages.append(PageText(doc_id, int(item["page_number"]), text, content_hash))
    if not pages:
        raise ValueError(f"no verified pages in extraction artifact for {doc_id}")
    return pages


def extract_pages(pdf_path: Path, *, doc_id: str) -> list[PageText]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PdfDependencyError("pypdf is not installed") from exc
    if not pdf_path.exists() or pdf_path.read_bytes()[:5] != b"%PDF-":
        raise ValueError(f"not a readable PDF: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    pages: list[PageText] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\u00a0", " ").strip()
        if not text:
            raise ValueError(f"empty extracted text on page {page_number} of {doc_id}")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        pages.append(PageText(doc_id, page_number, text, content_hash))
    return pages


def _token_count(text: str, tokenizer: Callable[[str], Any] | Any) -> int:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        encoded = backend.encode(text)
        ids = getattr(encoded, "ids", encoded)
        return len(ids)
    if callable(tokenizer):
        encoded = tokenizer(text)
    elif hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids", [])
    shape = getattr(encoded, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    if isinstance(encoded, (list, tuple)) and encoded and isinstance(encoded[0], (list, tuple)):
        return len(encoded[0])
    return len(encoded)


def chunk_page_text(
    page: PageText,
    *,
    max_characters: int = 1800,
    overlap: int = 240,
    tokenizer: Callable[[str], Any] | Any | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, object]]:
    """Create deterministic page chunks with an optional token hard limit."""

    if max_characters <= overlap or overlap < 0:
        raise ValueError("max_characters must be greater than overlap >= 0")
    chunks: list[dict[str, object]] = []
    start = 0
    order = 0
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    while start < len(page.text):
        stop = min(start + max_characters, len(page.text))
        if tokenizer is not None and max_tokens is not None:
            low, high = start + 1, stop
            best = start + 1
            while low <= high:
                middle = (low + high) // 2
                if _token_count(page.text[start:middle], tokenizer) <= max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            stop = best
        if stop < len(page.text):
            # Keep chunks readable at extraction/layout boundaries. The old
            # character-only split could start in the middle of a Thai line or
            # numbered list, which made the offline evidence extract look like
            # a broken answer even though the source text was verified.
            boundary_floor = start + max(80, min(overlap, max_characters // 4))
            candidates = [
                page.text.rfind("\n", start, stop),
                page.text.rfind(".", start, stop),
                page.text.rfind("?", start, stop),
                page.text.rfind("!", start, stop),
                page.text.rfind(" ", start, stop),
            ]
            boundary = max((item for item in candidates if item >= boundary_floor), default=-1)
            if boundary >= 0:
                stop = boundary + 1
        text = page.text[start:stop].strip()
        if text:
            chunk_id = f"{page.doc_id}-p{page.page_number}-c{order}"
            chunks.append(
                {
                    "doc_id": page.doc_id,
                    "page_number": page.page_number,
                    "chunk_id": chunk_id,
                    "text": text,
                    "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
        if stop == len(page.text):
            break
        start = max(start + 1, stop - overlap)
        if start < len(page.text):
            # Prefer the next numbered-list item when a PDF line wrap splits
            # an item (for example, the Thai text may wrap before item 15).
            # This is layout normalization only; it does not add or infer text.
            limit = min(len(page.text), start + max(80, overlap))
            next_line = page.text.find("\n", start, limit)
            marker = re.search(r"(?<!\d)\d{1,2}\.\s+", page.text[start:limit])
            marker_pos = start + marker.start() if marker else -1
            section = re.search(r"(?m)^\d{1,2}\s+[^\n]{0,100}(?:ได้แก่|โรค)[^\n]*", page.text[start:limit])
            section_pos = start + section.start() if section else -1
            if section_pos >= 0:
                start = section_pos
            elif marker_pos >= 0 and (next_line < 0 or marker_pos < next_line):
                start = marker_pos
            elif next_line >= 0:
                start = next_line + 1
        order += 1
    return chunks
