import unittest

from insurex.config import Settings
from insurex.rag.contracts import RetrievedChunk
from insurex.rag.runtime import (
    GeneratedAnswer,
    GeminiServices,
    RequestBudget,
    _contains_no_answer_framing,
    _flatten_json_schema,
    format_grounded_evidence_fallback,
)


class _FakeStructured:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return {"parsed": response}


class _FakeModel:
    def __init__(self, responses):
        self.structured = _FakeStructured(responses)

    def with_structured_output(self, schema, method, include_raw):
        return self.structured


class StructuredSchemaTests(unittest.TestCase):
    def test_grounded_fallback_is_scannable_markdown(self):
        rendered = format_grounded_evidence_fallback(
            "ความคุ้มครอง • เสียชีวิตรับผลประโยชน์ • ชำระเบี้ย 15 ปี"
        )
        self.assertTrue(rendered.startswith("### "))
        self.assertIn("\n\n- เสียชีวิตรับผลประโยชน์", rendered)
        self.assertIn("\n\n- ชำระเบี้ย 15 ปี", rendered)

    def test_thai_no_answer_framing_is_detected(self):
        self.assertFalse(_contains_no_answer_framing("หลักฐานระบุว่าคุ้มครองถึงอายุ 90 ปี"))
        self.assertTrue(
            _contains_no_answer_framing(
                "ไม่พบข้อมูลเงื่อนไขการรับประกันภัยของผลิตภัณฑ์คุ้มตลอดชีพในหลักฐานที่กำหนด"
            )
        )
        self.assertTrue(_contains_no_answer_framing("ไม่พบข้อมูลนี้ในเอกสารผลิตภัณฑ์ที่ระบบมี"))
        self.assertFalse(
            _contains_no_answer_framing(
                "จากเอกสารหลักฐาน เงื่อนไขการรับประกันภัยมีระยะเวลา 90 วัน"
            )
        )

    def test_response_type_has_safe_backward_compatible_default(self):
        self.assertEqual(GeneratedAnswer(answer="grounded").response_type, "answer")

    def test_schema_is_flattened_for_gemini_without_dropping_required_fields(self):
        schema = _flatten_json_schema(GeneratedAnswer.model_json_schema())
        self.assertNotIn("$defs", schema)
        self.assertNotIn("$schema", schema)
        citation = schema["properties"]["citations"]["items"]
        self.assertEqual(citation["type"], "object")
        self.assertEqual(
            citation["required"],
            ["doc_id", "page_number", "chunk_id", "source_url"],
        )

    def test_schema_filter_does_not_remove_semantic_constraints(self):
        schema = _flatten_json_schema(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "minLength": 1},
                    "page_number": {"type": "integer", "minimum": 1},
                },
                "required": ["answer", "page_number"],
                "additionalProperties": False,
            }
        )
        self.assertEqual(schema["properties"]["answer"]["minLength"], 1)
        self.assertEqual(schema["properties"]["page_number"]["minimum"], 1)
        self.assertNotIn("additionalProperties", schema)

    def test_semantic_validation_retry_repairs_unsupported_claim(self):
        chunk = RetrievedChunk(
            doc_id="doc-1",
            product_id="product-a",
            product_name="Product A",
            page_number=1,
            chunk_id="doc-1-p1-c1",
            text="Product A coverage 20 years",
            source_url="https://example.test/a.pdf",
        )
        citation = {
            "doc_id": chunk.doc_id,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "source_url": chunk.source_url,
        }
        model = _FakeModel(
            [
                {"answer": "Product A covers 90 years", "citations": [citation]},
                {"answer": "Product A coverage is 20 years", "citations": [citation]},
            ]
        )
        budget = RequestBudget(max_requests=2)
        services = GeminiServices(Settings(), model, budget).services()
        answer, citations, response_type = services.generate("Product A coverage", [chunk])
        self.assertEqual(answer, "Product A coverage is 20 years")
        self.assertEqual(response_type, "answer")
        self.assertEqual(len(citations), 1)
        self.assertEqual(budget.requests, 2)
        self.assertEqual(len(model.structured.prompts), 2)
        self.assertIn("Current date and time in Asia/Bangkok:", model.structured.prompts[0])
        self.assertIn("VALIDATION REPAIR", model.structured.prompts[1])

    def test_no_answer_with_factual_explanation_is_repaired(self):
        chunk = RetrievedChunk(
            doc_id="doc-1",
            product_id="product-a",
            product_name="Product A",
            page_number=1,
            chunk_id="doc-1-p1-c1",
            text="Product A accepts applications under the company's underwriting rules.",
            source_url="https://example.test/a.pdf",
        )
        citation = {
            "doc_id": chunk.doc_id,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "source_url": chunk.source_url,
        }
        model = _FakeModel(
            [
                {
                    "answer": "ไม่พบข้อมูลเงื่อนไข แต่บริษัทใช้หลักเกณฑ์การรับประกันของตนเอง",
                    "citations": [],
                    "response_type": "no_answer",
                },
                {
                    "answer": "Product A accepts applications under the company\'s underwriting rules.",
                    "citations": [citation],
                    "response_type": "answer",
                },
            ]
        )
        budget = RequestBudget(max_requests=2)
        services = GeminiServices(Settings(), model, budget).services()
        answer, citations, response_type = services.generate("Product A underwriting", [chunk])
        self.assertEqual(response_type, "answer")
        self.assertIn("underwriting", answer)
        self.assertEqual(len(citations), 1)
        self.assertEqual(budget.requests, 2)
        self.assertIn("VALIDATION REPAIR", model.structured.prompts[1])

    def test_no_answer_with_citation_and_negative_framing_is_repaired(self):
        chunk = RetrievedChunk(
            doc_id="doc-1",
            product_id="product-a",
            product_name="Product A",
            page_number=1,
            chunk_id="doc-1-p1-c1",
            text="Product A underwriting follows the company's criteria.",
            source_url="https://example.test/a.pdf",
        )
        citation = {
            "doc_id": chunk.doc_id,
            "page_number": chunk.page_number,
            "chunk_id": chunk.chunk_id,
            "source_url": chunk.source_url,
        }
        model = _FakeModel(
            [
                {
                    "answer": "Product A accepts applications under the company\'s underwriting rules.",
                    "citations": [citation],
                    "response_type": "no_answer",
                },
                {
                    "answer": "Product A underwriting follows the company's criteria.",
                    "citations": [citation],
                    "response_type": "answer",
                },
            ]
        )
        budget = RequestBudget(max_requests=3)
        services = GeminiServices(Settings(), model, budget).services()
        answer, citations, response_type = services.generate("Product A underwriting", [chunk])
        self.assertEqual(response_type, "answer")
        self.assertIn("underwriting", answer)
        self.assertEqual(len(citations), 1)
        self.assertEqual(budget.requests, 2)

    def test_exhausted_model_repairs_return_verified_evidence_extract(self):
        chunk = RetrievedChunk(
            doc_id="doc-1",
            product_id="product-a",
            product_name="Product A",
            page_number=4,
            chunk_id="doc-1-p4-c1",
            text="Product A provides coverage for 20 years.",
            source_url="https://example.test/a.pdf",
        )
        model = _FakeModel(
            [
                {"answer": "ไม่พบข้อมูลนี้ในเอกสาร", "citations": [], "response_type": "no_answer"},
                {"answer": "Product A covers 90 years", "citations": [], "response_type": "answer"},
                {"answer": "ไม่พบข้อมูลนี้ในเอกสาร", "citations": [], "response_type": "no_answer"},
            ]
        )
        budget = RequestBudget(max_requests=3)
        services = GeminiServices(Settings(), model, budget).services()

        answer, citations, response_type = services.generate("Product A coverage", [chunk])

        self.assertEqual(response_type, "answer")
        self.assertIn("Product A provides coverage for 20 years.", answer)
        self.assertNotIn("ไม่พบข้อมูล", answer)
        self.assertEqual([citation.chunk_id for citation in citations], [chunk.chunk_id])
        self.assertEqual(budget.requests, 3)

    def test_provider_failure_returns_verified_evidence_extract(self):
        chunk = RetrievedChunk(
            doc_id="doc-1",
            product_id="product-a",
            product_name="Product A",
            page_number=2,
            chunk_id="doc-1-p2-c1",
            text="Product A follows the company's underwriting criteria.",
            source_url="https://example.test/a.pdf",
        )
        model = _FakeModel([RuntimeError("provider unavailable"), RuntimeError("provider unavailable")])
        settings = Settings(max_retrieval_retries=1)
        budget = RequestBudget(max_requests=2)
        services = GeminiServices(settings, model, budget).services()

        answer, citations, response_type = services.generate("Product A underwriting", [chunk])

        self.assertEqual(response_type, "answer")
        self.assertIn("underwriting criteria", answer)
        self.assertEqual([citation.chunk_id for citation in citations], [chunk.chunk_id])
        self.assertEqual(budget.requests, 2)


if __name__ == "__main__":
    unittest.main()
