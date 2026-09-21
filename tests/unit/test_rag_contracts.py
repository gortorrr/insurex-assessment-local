import unittest

from insurex.rag.contracts import (
    AMBIGUOUS_PRODUCT_MESSAGE,
    NO_ANSWER_MESSAGE,
    Citation,
    RetrievedChunk,
    assess_candidates,
    validate_answer_support,
    validate_citations,
)
from insurex.rag.graph import _route_intent, build_rag_graph, build_standalone_query, prepare_turn, RagServices
from rag.ui.app import should_handoff
from insurex.rag.runtime import _assess_local, _llm_evidence_text, format_offline_evidence_extract


CHUNK = RetrievedChunk(
    doc_id="doc-1",
    product_id="product-a",
    product_name="Product A",
    page_number=3,
    chunk_id="doc-1-p3-c1",
    text="Coverage text",
    source_url="https://example.test/a.pdf",
    score=0.21,
)


class RagContractTests(unittest.TestCase):
    def test_age_range_before_next_numeric_table_row_is_supported(self):
        chunk = RetrievedChunk(**{**CHUNK.__dict__, "text": "อายุรับประกันภัย\n20 - 55 ปี\n300,000 บาท"})
        citation = Citation(chunk.doc_id, chunk.page_number, chunk.chunk_id, chunk.source_url)
        self.assertTrue(validate_answer_support("อายุรับประกันภัย 20 - 55 ปี", [citation], [chunk])[0])
        self.assertFalse(validate_answer_support("อายุรับประกันภัย 20 - 56 ปี", [citation], [chunk])[0])

    def test_no_candidates_is_insufficient(self):
        assessment = assess_candidates([])
        self.assertEqual(assessment.status, "insufficient")
        self.assertEqual(NO_ANSWER_MESSAGE, "ไม่พบข้อมูลนี้ในเอกสารผลิตภัณฑ์ที่ระบบมี")

    def test_multi_product_candidates_are_ambiguous_without_product(self):
        other = RetrievedChunk(**{**CHUNK.__dict__, "doc_id": "doc-2", "product_id": "product-b", "chunk_id": "doc-2-p1-c1"})
        self.assertEqual(assess_candidates([CHUNK, other]).status, "ambiguous")

    def test_generic_age_question_does_not_mix_conditions_across_products(self):
        a = RetrievedChunk(**{**CHUNK.__dict__, "text": "อายุรับประกันภัย 20 ปี"})
        b = RetrievedChunk(**{**a.__dict__, "product_id": "product-b", "product_name": "Product B"})
        result = assess_candidates([a, b], query="อายุรับประกันภัยเท่าไร", evidence_status="sufficient")
        self.assertEqual(result.status, "ambiguous")

    def test_citation_must_exist_in_current_context(self):
        valid = Citation(CHUNK.doc_id, CHUNK.page_number, CHUNK.chunk_id, CHUNK.source_url)
        self.assertEqual(validate_citations([valid], [CHUNK])[0], True)
        invalid = Citation("doc-unknown", 9, "missing", "https://example.test/missing.pdf")
        self.assertEqual(validate_citations([invalid], [CHUNK])[0], False)

    def test_unrelated_chunk_is_not_sufficient_for_out_of_corpus_question(self):
        unrelated = RetrievedChunk(**{**CHUNK.__dict__, "text": "ข้อมูลการรักษาพยาบาลทั่วไป"})
        self.assertEqual(_assess_local("สูตรทำผัดไทย", [unrelated]), "insufficient")

    def test_generic_insurance_terms_do_not_ground_another_domain(self):
        unrelated = RetrievedChunk(
            **{
                **CHUNK.__dict__,
                "product_name": "คุ้มตลอดชีพ พลัส",
                "text": "ผลประโยชน์และความคุ้มครองชีวิตและโรคร้ายแรงตามกรมธรรม์",
            }
        )
        self.assertEqual(_assess_local("ประกันรถยนต์ชั้น 1 คุ้มครองอะไร", [unrelated]), "insufficient")
        self.assertEqual(_assess_local("ดอกเบี้ยเงินฝากธนาคารเท่าไร", [unrelated]), "insufficient")

    def test_product_routed_plan_price_question_uses_numeric_evidence(self):
        product_chunk = RetrievedChunk(
            **{
                **CHUNK.__dict__,
                "product_id": "khum-aomsook-25-15",
                "product_name": "Khum Aomsook 25/15",
                "text": "คุ้มออมสุข แผน 1 แผน 2 แผน 3 จำนวนเงินเอาประกันภัย 150,000 450,000 750,000 บาท",
                "score": 0.63,
            }
        )
        self.assertEqual(
            _assess_local("คุ้มออมสุข มีกี่แผน และแต่ละแผนมีราคาเท่าไรบ้าง", [product_chunk]),
            "sufficient",
        )

    def test_numeric_answer_claim_must_exist_in_cited_text(self):
        citation = Citation(CHUNK.doc_id, CHUNK.page_number, CHUNK.chunk_id, CHUNK.source_url)
        supported = RetrievedChunk(**{**CHUNK.__dict__, "text": "คุ้มครอง 80 ปี"})
        self.assertEqual(validate_answer_support("คุ้มครอง 90 ปี", [citation], [supported])[0], False)

    def test_numeric_range_from_pdf_table_is_validated_as_a_range(self):
        evidence = RetrievedChunk(
            **{
                **CHUNK.__dict__,
                "text": "ระยะเวลาชำระเบี้ยประกันภัย 5 ปี อายุรับประกันภัย 16 - 60 ปี ถึงอายุ 90 ปี",
            }
        )
        citation = Citation(evidence.doc_id, evidence.page_number, evidence.chunk_id, evidence.source_url)
        answer = "อายุรับประกันภัย 16 - 60 ปี และคุ้มครองถึงอายุ 90 ปี"
        self.assertEqual(validate_answer_support(answer, [citation], [evidence])[0], True)

    def test_answer_support_rejects_negation_and_numeric_field_swap(self):
        evidence = RetrievedChunk(
            **{
                **CHUNK.__dict__,
                "text": "ไม่คุ้มครองโรคมะเร็ง ระยะเวลาคุ้มครอง 90 ปี ชำระเบี้ย 20 ปี",
            }
        )
        citation = Citation(evidence.doc_id, evidence.page_number, evidence.chunk_id, evidence.source_url)
        self.assertFalse(validate_answer_support("คุ้มครองโรคมะเร็ง", [citation], [evidence])[0])
        self.assertFalse(validate_answer_support("ชำระเบี้ย 90 ปี", [citation], [evidence])[0])
        self.assertFalse(validate_answer_support("คุ้มครอง 9 ปี", [citation], [evidence])[0])

    def test_answer_support_accepts_currency_values_from_pdf_table_layout(self):
        evidence = RetrievedChunk(
            **{
                **CHUNK.__dict__,
                "text": (
                    "ตัวอย่างเบี้ยประกันภัย แผน 1 แผน 2 แผน 3 "
                    "รายเดือน 878 2,635 4,346 รายปี 9,758 29,274 48,290"
                ),
            }
        )
        citation = Citation(evidence.doc_id, evidence.page_number, evidence.chunk_id, evidence.source_url)
        answer = "แผน 1 มีเบี้ยรายเดือน 878 บาท และรายปี 9,758 บาท"
        self.assertEqual(validate_answer_support(answer, [citation], [evidence])[0], True)

    def test_answer_support_accepts_duration_values_from_pdf_table_layout(self):
        evidence = RetrievedChunk(
            **{
                **CHUNK.__dict__,
                "text": (
                    "ระยะเวลาเอาประกันภัย ระยะเวลาชําระเบี้ยประกันภัย "
                    "จํานวนเงินเอาประกันภัยขั้นตํ่า 25 ปี 15 ปี 100,000 บาท"
                ),
            }
        )
        citation = Citation(evidence.doc_id, evidence.page_number, evidence.chunk_id, evidence.source_url)
        answer = "ระยะเวลาเอาประกันภัย 25 ปี และชำระเบี้ยประกันภัย 15 ปี"
        self.assertTrue(validate_answer_support(answer, [citation], [evidence])[0])

    def test_unrelated_new_question_does_not_inherit_product_history(self):
        query = build_standalone_query(
            "สูตรทำผัดไทย",
            [
                {"role": "user", "content": "ชื่อ สมชาย รายได้ 45000 บาทต่อเดือน เบอร์ 0812345678"},
                {"role": "assistant", "content": "คุ้มครองชีวิตถึงอายุ 90 ปี"},
            ],
        )
        self.assertEqual(query, "สูตรทำผัดไทย")
        self.assertNotIn("45000", query)
        self.assertNotIn("0812345678", query)

    def test_unverified_private_use_text_is_preserved_for_quarantine(self):
        raw = "coverage \ue000 text"
        self.assertEqual(_llm_evidence_text(raw), raw)

    def test_offline_evidence_extract_is_marked_and_markdown_formatted(self):
        output = format_offline_evidence_extract(
            "4 โรคร้ายแรงระดับเริ่มต้น ได้แก่ 1. โรคมะเร็งระยะไม่ลุกลาม",
            doc_id="doc-1",
            page_number=1,
        )
        self.assertIn("### ข้อความหลักฐานจากเอกสาร", output)
        self.assertIn("ไม่ใช่คำตอบที่สร้างโดย LLM", output)
        self.assertIn("**เอกสาร:** `doc-1`", output)
        self.assertIn("> 4 โรคร้ายแรงระดับเริ่มต้น", output)

    def test_follow_up_query_uses_bounded_history(self):
        query = build_standalone_query(
            "แล้วแบบนั้นจ่ายกี่ปี",
            [{"role": "user", "content": "สนใจ Product A"}, {"role": "assistant", "content": "Product A คุ้มครองตามเอกสาร"}],
        )
        self.assertIn("Product A", query)
        self.assertIn("แล้วแบบนั้นจ่ายกี่ปี", query)
        self.assertLessEqual(len(query), 2400)

    def test_graph_passes_history_into_retrieval_query(self):
        seen = []
        def retrieve(query, product_id):
            seen.append(query)
            return [CHUNK]
        services = RagServices(
            retrieve=retrieve,
            assess=lambda query, chunks: "sufficient",
            generate=lambda query, chunks: ("Coverage text", [Citation(CHUNK.doc_id, CHUNK.page_number, CHUNK.chunk_id, CHUNK.source_url)]),
            rewrite=lambda query: query + " details",
        )
        graph = build_rag_graph(services)
        graph.invoke({
            "query": "follow up", "active_product_id": "product-a",
            "conversation_history": [{"role": "user", "content": "Product A"}],
        })
        self.assertIn("Product A", seen[0])

    def test_graph_searches_all_products_when_ui_does_not_select_one(self):
        seen = []

        def retrieve(query, product_id):
            seen.append(product_id)
            return [CHUNK]

        services = RagServices(
            retrieve=retrieve,
            assess=lambda query, chunks: "sufficient",
            generate=lambda query, chunks: (
                "Coverage text",
                [Citation(CHUNK.doc_id, CHUNK.page_number, CHUNK.chunk_id, CHUNK.source_url)],
            ),
            rewrite=lambda query: query,
        )
        graph = build_rag_graph(services)
        graph.invoke({"query": "คุ้มครองอะไร", "active_product_id": None})
        self.assertTrue(seen)
        self.assertTrue(all(product_id is None for product_id in seen))

    def test_sufficient_multi_product_evidence_reaches_grounded_generator(self):
        product_a = RetrievedChunk(
            **{**CHUNK.__dict__, "text": "Product A Coverage text"}
        )
        product_b = RetrievedChunk(
            **{
                **CHUNK.__dict__,
                "product_id": "product-b",
                "product_name": "Product B",
                "doc_id": "doc-2",
                "chunk_id": "doc-2-p1-c1",
                "text": "Product B Coverage text",
            }
        )
        generated = []
        services = RagServices(
            retrieve=lambda query, product_id: [product_a, product_b],
            assess=lambda query, chunks: "sufficient",
            generate=lambda query, chunks: (
                generated.append(True) or "Product A Product B",
                [
                    Citation(product_a.doc_id, product_a.page_number, product_a.chunk_id, product_a.source_url),
                    Citation(product_b.doc_id, product_b.page_number, product_b.chunk_id, product_b.source_url),
                ],
            ),
            rewrite=lambda query: query,
        )
        result = build_rag_graph(services).invoke({"query": "Product A Coverage", "active_product_id": None})
        self.assertEqual(result["evidence_status"], "sufficient")
        self.assertTrue(generated)
        self.assertEqual(result["answer"], "Product A Product B")

    def test_multi_product_clarification_with_no_citations_uses_safe_router_fallback(self):
        product_a = RetrievedChunk(**{**CHUNK.__dict__, "text": "Product A Coverage text"})
        product_b = RetrievedChunk(
            **{**CHUNK.__dict__, "product_id": "product-b", "product_name": "Product B", "doc_id": "doc-2", "chunk_id": "doc-2-p1-c1", "text": "Product B Coverage text"}
        )
        services = RagServices(
            retrieve=lambda query, product_id: [product_a, product_b],
            assess=lambda query, chunks: "sufficient",
            generate=lambda query, chunks: ("??????????? ?????????????? Product B", [], "clarification"),
            rewrite=lambda query: query,
        )
        result = build_rag_graph(services).invoke({"query": "Product A Coverage", "active_product_id": None})
        self.assertEqual(result["response_type"], "answer")
        self.assertEqual(result["evidence_status"], "ambiguous")
        self.assertIn("Product A", result["answer"])
        self.assertIn("Product B", result["answer"])

    def test_single_product_clarification_without_claim_is_preserved(self):
        product = RetrievedChunk(**{**CHUNK.__dict__, "text": "Product A Coverage text"})
        services = RagServices(
            retrieve=lambda query, product_id: [product],
            assess=lambda query, chunks: "sufficient",
            generate=lambda query, chunks: ("?????????????????????????????", [], "clarification"),
            rewrite=lambda query: query,
        )
        result = build_rag_graph(services).invoke({"query": "Product A Coverage", "active_product_id": "product-a"})
        self.assertEqual(result["response_type"], "clarification")
        self.assertEqual(result["answer"], "กรุณาระบุรายละเอียดที่ต้องการทราบเกี่ยวกับผลิตภัณฑ์นี้เพิ่มเติม")

    def test_factual_claim_disguised_as_clarification_is_not_shown(self):
        services = RagServices(
            retrieve=lambda query, product_id: [CHUNK],
            assess=lambda query, chunks: "sufficient",
            generate=lambda query, chunks: ("คุ้มครองทุกโรค 999 ปี", [], "clarification"),
            rewrite=lambda query: query,
        )
        result = build_rag_graph(services).invoke({"query": "Coverage", "active_product_id": "product-a"})
        self.assertNotIn("999", result["answer"])
        self.assertEqual(result["citations"], [])

    def test_no_answer_type_cannot_return_product_claims_with_citations(self):
        citation = Citation(CHUNK.doc_id, CHUNK.page_number, CHUNK.chunk_id, CHUNK.source_url)
        services = RagServices(
            retrieve=lambda query, product_id: [CHUNK],
            assess=lambda query, chunks: "sufficient",
            generate=lambda query, chunks: ("Coverage text", [citation], "no_answer"),
            rewrite=lambda query: query,
        )
        result = build_rag_graph(services).invoke({"query": "Coverage", "active_product_id": "product-a"})
        self.assertEqual(result["answer"], NO_ANSWER_MESSAGE)
        self.assertEqual(result["citations"], [])

    def test_active_lead_collection_allows_product_question_to_reach_rag(self):
        self.assertEqual(
            _route_intent({"lead_collection_active": True, "query": "คุ้มครองกี่ปี"})["intent"],
            "question",
        )
        self.assertEqual(
            _route_intent({"lead_collection_active": True, "query": "อาชีพ โปรแกรมเมอร์"})["intent"],
            "lead",
        )

    def test_conversation_intents_bypass_rag_even_when_lead_refresh_is_due(self):
        cases = {
            "สวัสดีครับ": "greeting",
            "ขอบคุณมากครับ": "thanks",
            "ไว้คุยใหม่": "goodbye",
            "ช่วยอะไรได้บ้าง": "help",
            "ช่วยบอกสูตรทำแกงเขียวหวาน": "out_of_scope",
            "ขอคุยกับพนักงาน": "human_handoff",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                state = _route_intent({"query": text, "lead_refresh_required": True})
                self.assertEqual(state["intent"], expected)

    def test_direct_conversation_nodes_do_not_call_retrieval_or_generation(self):
        calls = []
        services = RagServices(
            retrieve=lambda query, product_id: calls.append("retrieve") or [],
            assess=lambda query, chunks: calls.append("assess") or "insufficient",
            generate=lambda query, chunks: calls.append("generate") or ("unexpected", []),
            rewrite=lambda query: calls.append("rewrite") or query,
        )
        graph = build_rag_graph(services)
        for text in ("สวัสดีครับ", "ขอบคุณครับ", "ช่วยอะไรได้บ้าง", "สูตรทำผัดไทย"):
            result = graph.invoke({"query": text})
            self.assertEqual(result["response_type"], "direct")
            self.assertFalse(result["handoff_requested"])
        self.assertEqual(calls, [])

    def test_only_explicit_human_or_failed_product_question_is_handed_off(self):
        self.assertTrue(should_handoff({"intent": "human_handoff", "handoff_requested": True}, []))
        self.assertTrue(should_handoff(
            {"intent": "question", "evidence_status": "insufficient", "response_type": "answer"}, []
        ))
        self.assertFalse(should_handoff(
            {"intent": "out_of_scope", "evidence_status": "insufficient", "response_type": "direct"}, []
        ))
        self.assertFalse(should_handoff(
            {"intent": "greeting", "evidence_status": "", "response_type": "direct"}, []
        ))

    def test_new_turn_clears_retrieval_state_but_keeps_context_outside_transients(self):
        state = {
            "query": "new question",
            "session_id": "session-a",
            "active_product_id": "product-a",
            "standalone_query": "old question",
            "retrieved_chunks": [CHUNK],
            "citations": [Citation(CHUNK.doc_id, CHUNK.page_number, CHUNK.chunk_id, CHUNK.source_url)],
            "retrieval_attempts": 1,
            "error_code": "OLD_ERROR",
        }
        cleared = prepare_turn(state)
        self.assertEqual(cleared["standalone_query"], "")
        self.assertEqual(cleared["retrieved_chunks"], [])
        self.assertEqual(cleared["citations"], [])
        self.assertEqual(cleared["retrieval_attempts"], 0)
        self.assertIsNone(cleared["error_code"])
        self.assertNotIn("session_id", cleared)

    def test_ambiguous_message_is_a_distinct_contract(self):
        self.assertIn("หลายผลิตภัณฑ์", AMBIGUOUS_PRODUCT_MESSAGE)


if __name__ == "__main__":
    unittest.main()

