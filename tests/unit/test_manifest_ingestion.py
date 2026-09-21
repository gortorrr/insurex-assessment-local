import json
from pathlib import Path
import tempfile
import unittest

from insurex.rag.ingestion import CHUNKING_REVISION, PageText, chunk_page_text, load_verified_pages
from insurex.rag.index import (
    _chunk_ids_hash,
    find_product_matches,
    infer_product_id_from_query,
    stale_chunk_ids,
    verify_index_metadata,
)
from insurex.rag.manifest import validate_manifest
from unittest.mock import patch


class ManifestIngestionTests(unittest.TestCase):
    def test_active_corrected_extraction_has_no_private_use_glyphs(self):
        root = Path(__file__).resolve().parents[2]
        corpus_root = root / "rag"
        manifest = json.loads((corpus_root / "knowledge_base/manifest.json").read_text(encoding="utf-8"))
        for document in manifest["documents"]:
            pages = load_verified_pages(
                corpus_root / document["corrected_text_path"],
                doc_id=document["doc_id"],
                expected_pdf_sha256=document["sha256"],
            )
            self.assertEqual(len(pages), document["page_count"])
            self.assertTrue(all(not any(0xE000 <= ord(char) <= 0xF8FF for char in page.text) for page in pages))

    def test_verified_extraction_rejects_wrong_pdf_provenance(self):
        root = Path(__file__).resolve().parents[2]
        corpus_root = root / "rag"
        document = json.loads((corpus_root / "knowledge_base/manifest.json").read_text(encoding="utf-8"))["documents"][0]
        with self.assertRaisesRegex(ValueError, "provenance"):
            load_verified_pages(
                corpus_root / document["corrected_text_path"],
                doc_id=document["doc_id"],
                expected_pdf_sha256="wrong-pdf-hash",
            )

    def test_stale_chunk_ids_are_removed_by_exact_id(self):
        self.assertEqual(stale_chunk_ids(["current", "old_%_chunk"], {"current"}), ["old_%_chunk"])

    def test_query_product_alias_scopes_retrieval_only_when_unique(self):
        products = [
            ("khum-talodcheep-ci-plus", "Khum Talodcheep CI Plus"),
            ("khum-talodcheep-plus", "Khum Talodcheep Plus"),
        ]
        self.assertEqual(
            infer_product_id_from_query("Khum Talodcheep CI Plus คุ้มครองอะไร", products),
            "khum-talodcheep-ci-plus",
        )
        self.assertIsNone(infer_product_id_from_query("คุ้มครองอะไรบ้าง", products))

    def test_partial_thai_name_returns_every_tied_product_for_clarification(self):
        products = [
            ("khum-talodcheep-ci-plus", "Khum Talodcheep CI Plus"),
            ("khum-talodcheep-plus", "Khum Talodcheep Plus"),
        ]
        aliases = {
            "khum-talodcheep-ci-plus": ["คุ้มตลอดชีพ ซีไอ พลัส"],
            "khum-talodcheep-plus": ["คุ้มตลอดชีพ พลัส"],
        }
        self.assertEqual(
            find_product_matches("สนใจคุ้มตลอดชีพอะ", products, aliases),
            ["khum-talodcheep-ci-plus", "khum-talodcheep-plus"],
        )
        self.assertEqual(
            infer_product_id_from_query("สนใจคุ้มตลอดชีพ ซีไอ พลัส", products, aliases),
            "khum-talodcheep-ci-plus",
        )

    def test_manifest_template_is_blocked_until_five_real_files_exist(self):
        root = Path(__file__).resolve().parents[2]
        corpus_root = root / "rag"
        manifest = json.loads((corpus_root / "knowledge_base/manifest.template.json").read_text(encoding="utf-8"))
        errors = validate_manifest(manifest, root=corpus_root)
        self.assertTrue(any("expected exactly 5" in error for error in errors))

    def test_chunks_keep_page_and_deterministic_ids(self):
        page = PageText("doc-a", 2, "ก" * 100, "page-hash")
        chunks = chunk_page_text(page, max_characters=40, overlap=5)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["page_number"] == 2 for chunk in chunks))
        self.assertEqual(chunks[0]["chunk_id"], "doc-a-p2-c0")
        self.assertEqual(chunks, chunk_page_text(page, max_characters=40, overlap=5))

    def test_token_aware_chunks_respect_hard_limit(self):
        page = PageText("doc-token", 1, "abcdefghij" * 10, "page-hash")
        tokenizer = lambda text: list(text)
        chunks = chunk_page_text(page, max_characters=80, overlap=5, tokenizer=tokenizer, max_tokens=20)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk["text"]) <= 20 for chunk in chunks))

    def test_chunks_prefer_line_boundaries_for_readable_evidence(self):
        page = PageText("doc-lines", 1, "หัวข้อแรก\n" + "ข้อความยาว " * 20 + "\nหัวข้อถัดไป\nรายละเอียด", "page-hash")
        chunks = chunk_page_text(page, max_characters=80, overlap=10)
        self.assertEqual(CHUNKING_REVISION, "paragraph-boundary-v4")
        self.assertTrue(all(not chunk["text"].startswith("ข้อความยาว ข้อ") for chunk in chunks))

    def test_numbered_section_does_not_start_with_tail_of_previous_item(self):
        page = PageText(
            "doc-section",
            1,
            "15 โรคร้ายแรงระดับรุนแรง ได้แก่ 1. รายการหนึ่ง\n13. รายการสิบสาม\n"
            "4 โรคร้ายแรงระดับเริ่มต้น ได้แก่ 1. รายการใหม่\nรายละเอียด",
            "page-hash",
        )
        chunks = chunk_page_text(page, max_characters=65, overlap=10)
        self.assertTrue(any(chunk["text"].startswith("4 โรคร้ายแรงระดับเริ่มต้น") for chunk in chunks))

    def test_index_verification_fails_closed_on_actual_metadata_mismatch(self):
        class FakeCollection:
            def get(self):
                return {"ids": ["a", "b"]}

            def count(self):
                return 2

        class FakeClient:
            def get_collection(self, name):
                self.asserted_name = name
                return FakeCollection()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {
                "collection_name": "insurex_kb",
                "chunk_count": 2,
                "chunk_ids_sha256": _chunk_ids_hash(["a", "b"]),
                "index_fingerprint": "fingerprint-a",
            }
            (root / "index_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with patch("insurex.rag.index._client", return_value=FakeClient()):
                self.assertEqual(verify_index_metadata(root, expected_fingerprint="fingerprint-a")["actual_count"], 2)
                with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                    verify_index_metadata(root, expected_fingerprint="fingerprint-b")


if __name__ == "__main__":
    unittest.main()
