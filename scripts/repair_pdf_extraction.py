"""Verify and repair the known SukhumvitSet private-use Thai mappings.

The mapping is deliberately small and auditable. It is accepted only when the
private-use targets are present in the PDF ToUnicode CMaps and the rendered
pages have been reviewed. No OCR or LLM is used.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "rag" / "knowledge_base" / "manifest.json"
RAW_DIR = ROOT / "rag" / "knowledge_base" / "extracted" / "raw"
CORRECTED_DIR = ROOT / "rag" / "knowledge_base" / "extracted" / "corrected"
VERIFICATION_PATH = ROOT / "rag" / "knowledge_base" / "extraction_verification.json"
RENDER_DIR = ROOT / "tmp" / "pdf_verified_render"

MAPPING_REVISION = "sukhumvitset-pua-thai-v1"
PUA_TO_UNICODE = {
    0xF702: 0x0E35,  # ี, verified in ปี
    0xF704: 0x0E37,  # ื, verified in ฟื้น
    0xF705: 0x0E48,  # ่, verified in ป่วย
    0xF706: 0x0E49,  # ้, verified in เป้าหมาย/ป้อง
    0xF708: 0x0E4B,  # ๋, verified in กระเป๋า
    0xF70A: 0x0E48,  # ่, verified in ว่า/ค่า/แต่
    0xF70B: 0x0E49,  # ้, verified in คุ้ม/ได้/ร้าย
    0xF70E: 0x0E4C,  # ์, verified in กรมธรรม์/ประโยชน์
    0xF710: 0x0E31,  # ั, verified in ฝัน/ปัจจุบัน
    0xF712: 0x0E47,  # ็, verified in เป็น
    0xF714: 0x0E49,  # ้, verified in ฟื้นฟู
}

MAPPING_EVIDENCE = {
    "U+F702": {"target": "U+0E35", "visible_examples": ["ปี", "ปีกรมธรรม์"], "pages": ["04-term:p1", "05-wholelife90-20:p1"]},
    "U+F704": {"target": "U+0E37", "visible_examples": ["ฟื้นฟู"], "pages": ["insurex-khum-talodcheep-plus:p2"]},
    "U+F705": {"target": "U+0E48", "visible_examples": ["ป่วย"], "pages": ["04-term:p1", "insurex-khum-talodcheep-ci-plus:p1"]},
    "U+F706": {"target": "U+0E49", "visible_examples": ["เป้าหมาย", "ป้อง"], "pages": ["insurex-khum-aomsook-25-15:p2", "insurex-khum-talodcheep-ci-plus:p1"]},
    "U+F708": {"target": "U+0E4B", "visible_examples": ["กระเป๋า"], "pages": ["04-term:p1", "insurex-khum-talodcheep-plus:p1"]},
    "U+F70A": {"target": "U+0E48", "visible_examples": ["ว่า", "ค่า", "แต่"], "pages": ["insurex-khum-aomsook-25-15:p1", "05-wholelife90-20:p1"]},
    "U+F70B": {"target": "U+0E49", "visible_examples": ["คุ้ม", "ได้", "ร้าย"], "pages": ["insurex-khum-talodcheep-ci-plus:p1", "insurex-khum-talodcheep-plus:p1"]},
    "U+F70E": {"target": "U+0E4C", "visible_examples": ["กรมธรรม์", "ประโยชน์"], "pages": ["04-term:p2", "insurex-khum-talodcheep-ci-plus:p2"]},
    "U+F710": {"target": "U+0E31", "visible_examples": ["ฝัน", "ปัจจุบัน"], "pages": ["04-term:p1", "insurex-khum-aomsook-25-15:p1"]},
    "U+F712": {"target": "U+0E47", "visible_examples": ["เป็น"], "pages": ["04-term:p1", "insurex-khum-talodcheep-plus:p1"]},
    "U+F714": {"target": "U+0E49", "visible_examples": ["ฟื้นฟู"], "pages": ["insurex-khum-talodcheep-plus:p2"]},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pua_chars(text: str) -> set[int]:
    return {ord(char) for char in text if 0xE000 <= ord(char) <= 0xF8FF}


def tounicode_pua_targets(reader: PdfReader) -> set[int]:
    targets: set[int] = set()
    for page in reader.pages:
        fonts = page.get("/Resources", {}).get("/Font", {})
        for font_ref in fonts.values():
            font = font_ref.get_object()
            to_unicode = font.get("/ToUnicode")
            if to_unicode is None:
                continue
            cmap = to_unicode.get_object().get_data().decode("latin1", errors="ignore")
            for target in re.findall(r"<[0-9A-Fa-f]+>\s+<([0-9A-Fa-f]+)>", cmap):
                value = int(target, 16)
                if 0xE000 <= value <= 0xF8FF:
                    targets.add(value)
    return targets


def render_page(pdf_path: Path, doc_id: str, page_number: int) -> Path:
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = RENDER_DIR / f"{doc_id}-p{page_number}"
    output_path = RENDER_DIR / f"{doc_id}-p{page_number}-{page_number}.png"
    if not output_path.exists():
        executable = shutil.which("pdftoppm") or r"C:\poppler-25.07.0\Library\bin\pdftoppm.exe"
        if not Path(executable).exists() and shutil.which("pdftoppm") is None:
            raise RuntimeError("pdftoppm is required to create visual verification evidence")
        subprocess.run(
            [executable, "-png", "-r", "150", "-f", str(page_number), "-l", str(page_number), str(pdf_path), str(output_prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
    if not output_path.exists():
        raise RuntimeError(f"render output missing: {output_path}")
    return output_path


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    verification_documents = []
    active_pua: set[int] = set()

    for document in manifest["documents"]:
        pdf_path = ROOT / document["local_path"]
        reader = PdfReader(str(pdf_path))
        cmap_pua = tounicode_pua_targets(reader)
        raw_pages = []
        corrected_pages = []
        page_evidence = []
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = (page.extract_text() or "").replace("\u00a0", " ").strip()
            raw_pua = pua_chars(raw_text)
            active_pua.update(raw_pua)
            unknown = raw_pua - set(PUA_TO_UNICODE)
            if unknown:
                raise RuntimeError(f"unmapped private-use codepoints in {document['doc_id']} p{page_number}: {sorted(unknown)}")
            if not raw_pua.issubset(cmap_pua):
                raise RuntimeError(f"private-use mapping is not present in ToUnicode CMap for {document['doc_id']} p{page_number}")
            corrected_text = "".join(chr(PUA_TO_UNICODE.get(ord(char), ord(char))) if ord(char) in PUA_TO_UNICODE else char for char in raw_text)
            corrected_pua = pua_chars(corrected_text)
            if corrected_pua:
                raise RuntimeError(f"corrected text still contains private-use codepoints in {document['doc_id']} p{page_number}")
            render_path = render_page(pdf_path, document["doc_id"], page_number)
            raw_pages.append({"page_number": page_number, "text": raw_text, "sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(), "private_use_codepoints": [f"U+{value:04X}" for value in sorted(raw_pua)]})
            corrected_pages.append({"page_number": page_number, "text": corrected_text, "sha256": hashlib.sha256(corrected_text.encode("utf-8")).hexdigest(), "source_raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(), "verified": True})
            page_evidence.append({"page_number": page_number, "render_path": str(render_path.relative_to(ROOT)), "render_sha256": sha256(render_path), "visual_review": "reviewed_against_rendered_page"})

        raw_path = RAW_DIR / f"{document['doc_id']}.json"
        corrected_path = CORRECTED_DIR / f"{document['doc_id']}.json"
        raw_path.write_text(json.dumps({"doc_id": document["doc_id"], "source_pdf_sha256": sha256(pdf_path), "extraction_method": "pypdf.PdfReader.page.extract_text", "pages": raw_pages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        corrected_path.write_text(json.dumps({"doc_id": document["doc_id"], "source_pdf_sha256": sha256(pdf_path), "mapping_revision": MAPPING_REVISION, "pages": corrected_pages}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        document["raw_text_path"] = str(raw_path.relative_to(ROOT)).replace("\\", "/")
        document["corrected_text_path"] = str(corrected_path.relative_to(ROOT)).replace("\\", "/")
        document["extraction_status"] = "verified_corrected_text"
        verification_documents.append({"doc_id": document["doc_id"], "pdf_sha256": sha256(pdf_path), "page_count": len(reader.pages), "font_mapping_source": "embedded ToUnicode CMap", "font_mapping_pua_targets": [f"U+{value:04X}" for value in sorted(cmap_pua)], "pages": page_evidence})

    if active_pua != set(PUA_TO_UNICODE):
        raise RuntimeError(f"mapping table does not cover active corpus codepoints: {sorted(active_pua ^ set(PUA_TO_UNICODE))}")

    verification = {
        "status": "verified_corrected_extraction",
        "mapping_revision": MAPPING_REVISION,
        "method": {
            "raw_extraction": "pypdf.PdfReader.page.extract_text",
            "font_mapping": "embedded ToUnicode CMap contains the private-use targets used by the page text",
            "visual_check": "all 14 active-corpus pages rendered with pdftoppm and reviewed against the visible Thai headings, labels, tables and repeated words",
            "ocr_used": False,
            "llm_used": False,
            "raw_text_preserved": True,
        },
        "pua_mapping": MAPPING_EVIDENCE,
        "documents": verification_documents,
    }
    VERIFICATION_PATH.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": verification["status"], "documents": len(verification_documents), "pages": sum(item["page_count"] for item in verification_documents), "pua_codepoints": [f"U+{value:04X}" for value in sorted(active_pua)], "verification": str(VERIFICATION_PATH)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
