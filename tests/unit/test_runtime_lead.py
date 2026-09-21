import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from insurex.db import connect_local as connect
from insurex.leads import LeadDraft, draft_fingerprint, lead_summary
from insurex.rag.runtime import LeadTurnHandler
from insurex.session import create_session


def complete_draft(product_id: str = "product-a", name: str = "เอก") -> LeadDraft:
    return LeadDraft(
        name=name,
        occupation="Engineer",
        income=Decimal("30000"),
        income_currency="THB",
        income_period="monthly_thb",
        phone="0812345678",
        product_id=product_id,
    )


class LeadTurnHandlerTests(unittest.TestCase):
    def test_ui_without_product_selector_collects_and_saves_automatically(self):
        self.handler.product_aliases = {"product-a": ["คุ้มตลอดชีพ พลัส"]}
        state = {"active_product_id": None}
        state.update(self.handler("สนใจสมัคร คุ้มตลอดชีพ พลัส", state))
        self.assertEqual(state["lead_draft"]["product_id"], "product-a")
        state.update(self.handler("ชื่อ เอก อาชีพ ครู รายได้ 30000 บาทต่อเดือน เบอร์โทร 0812345678", state))
        self.assertEqual(state["lead_missing"], [])
        self.assertFalse(state["lead_confirmation_pending"])
        self.assertIsNotNone(state["lead_saved_id"])
        self.assertEqual(self.connection.execute("SELECT product_id FROM leads").fetchone()[0], "product-a")

    def test_english_product_name_is_inferred_without_selector(self):
        result = self.handler("interested in Product A", {"active_product_id": None})
        self.assertEqual(result["lead_draft"]["product_id"], "product-a")

    def test_shared_alias_does_not_choose_arbitrary_product(self):
        self.handler.product_aliases = {"product-a": ["ประกันชีวิต"], "product-b": ["ประกันชีวิต"]}
        result = self.handler("สนใจประกันชีวิต", {"active_product_id": None})
        self.assertIn("product_id", result["lead_missing"])
        self.assertEqual(result["lead_product_candidates"], ["product-a", "product-b"])
        self.assertIn("1. **ประกันชีวิต**", result["answer"])
        self.assertIn("2. **ประกันชีวิต**", result["answer"])

    def test_ambiguous_product_is_resolved_by_number_on_the_next_turn(self):
        self.handler.product_aliases = {
            "product-a": ["คุ้มตลอดชีพ ซีไอ พลัส"],
            "product-b": ["คุ้มตลอดชีพ พลัส"],
        }
        state = {"active_product_id": None}
        state.update(self.handler(
            "สนใจคุ้มตลอดชีพอะ ชื่อ เอก อาชีพ ครู รายได้ 30000 บาทต่อเดือน เบอร์ 0812345678",
            state,
        ))
        self.assertEqual(state["lead_product_candidates"], ["product-a", "product-b"])
        self.assertIn("พบผลิตภัณฑ์ชื่อใกล้กัน", state["answer"])
        self.assertIsNone(state["lead_saved_id"])

        state.update(self.handler("1", state))
        self.assertEqual(state["lead_product_candidates"], [])
        self.assertEqual(state["lead_draft"]["product_id"], "product-a")
        self.assertIsNotNone(state["lead_saved_id"])
        self.assertEqual(
            self.connection.execute("SELECT product_id FROM leads").fetchone()[0],
            "product-a",
        )

    def test_full_similar_product_name_does_not_ask_for_clarification(self):
        self.handler.product_aliases = {
            "product-a": ["คุ้มตลอดชีพ ซีไอ พลัส"],
            "product-b": ["คุ้มตลอดชีพ พลัส"],
        }
        result = self.handler("สนใจคุ้มตลอดชีพ พลัส", {"active_product_id": None})
        self.assertEqual(result["lead_draft"]["product_id"], "product-b")
        self.assertEqual(result["lead_product_candidates"], [])
        self.assertNotIn("พบผลิตภัณฑ์ชื่อใกล้กัน", result["answer"])

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.connection = connect(Path(self.temporary.name) / "lead.sqlite")
        self.session = create_session(self.connection, "owner-a")
        self.handler = LeadTurnHandler(
            self.connection,
            self.session.session_id,
            self.session.access_token,
            {"product-a", "product-b"},
            "request-1",
        )

    def tearDown(self):
        self.connection.close()
        self.temporary.cleanup()

    def state(self, draft: LeadDraft | None = None, *, pending: bool = False, saved_id=None):
        draft = draft or LeadDraft()
        return {
            "active_product_id": draft.product_id or "product-a",
            "lead_draft": draft.model_dump(mode="json"),
            "lead_confirmation_pending": pending,
            "lead_confirmation_fingerprint": draft_fingerprint(draft) if pending else None,
            "lead_saved_id": saved_id,
        }

    def test_complete_draft_is_saved_without_confirmation(self):
        result = self.handler("ขอดูข้อมูลก่อนยืนยัน", self.state(complete_draft()))
        self.assertIsNotNone(result["lead_saved_id"])
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 1)
        self.assertIn("สอบถามเกี่ยวกับ", result["answer"])
        self.assertNotIn("lead_id", result["answer"])
        self.assertIn("รายได้: 30000 THB / monthly_thb", result["lead_summary"])

    def test_partial_slots_are_persisted_immediately(self):
        result = self.handler("ชื่อ เอก", self.state())
        self.assertIsNone(result["lead_saved_id"])
        row = self.connection.execute(
            "SELECT name, occupation, status FROM lead_drafts WHERE owner_id=?",
            (self.session.owner_id,),
        ).fetchone()
        self.assertEqual(tuple(row), ("เอก", None, "unconfirmed"))
        self.assertEqual(
            result["answer"],
            "กรุณาแจ้ง อาชีพ, รายได้พร้อมสกุลเงินและช่วงเวลา, เบอร์โทรศัพท์ เพิ่มเติม",
        )
        self.assertEqual(result["lead_summary"], "ชื่อ: เอก | ผลิตภัณฑ์: product-a")
        self.assertNotIn("ยังไม่ระบุ", result["lead_summary"])

    def test_stale_profile_starts_refresh_without_overwriting_complete_lead(self):
        old = (datetime.now(UTC) - timedelta(days=184)).isoformat()
        self.connection.execute(
            """INSERT INTO leads(owner_id,session_id,product_id,name,occupation,income_thb,
               income_period,phone_normalized,request_id,payload_hash,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.session.owner_id, self.session.session_id, "product-a", "ชื่อเดิม", "งานเดิม",
             30000, "monthly_thb", "0812345678", "old-request", "old-hash", old, old),
        )
        self.connection.commit()
        result = self.handler("สวัสดี", {"active_product_id": None})
        self.assertIn("ชื่อ", result["answer"])
        self.assertIn("อาชีพ", result["answer"])
        self.assertEqual(
            self.connection.execute("SELECT name FROM leads WHERE owner_id=?", (self.session.owner_id,)).fetchone()[0],
            "ชื่อเดิม",
        )
        self.assertIsNotNone(
            self.connection.execute("SELECT refresh_requested_at FROM leads WHERE owner_id=?", (self.session.owner_id,)).fetchone()[0]
        )

    def test_short_product_alias_is_kept_without_llm_overwriting_it(self):
        self.handler.product_aliases = {"product-a": ["Product A", "คุ้มออมสุข"]}
        self.handler.extractor = lambda *_: LeadDraft()
        result = self.handler("อยากสมัคร คุ้มออมสุข", {"active_product_id": None})
        self.assertEqual(result["lead_draft"]["product_id"], "product-a")
        self.assertNotIn("ผลิตภัณฑ์", result["lead_missing"])

    def test_partial_slots_fall_back_to_local_parser_when_llm_extractor_fails(self):
        self.handler.extractor = lambda *_: (_ for _ in ()).throw(RuntimeError("provider unavailable"))
        result = self.handler("ชื่อ เอก", self.state())
        self.assertIsNone(result.get("error_code"))
        row = self.connection.execute(
            "SELECT name FROM lead_drafts WHERE owner_id=?",
            (self.session.owner_id,),
        ).fetchone()
        self.assertEqual(row[0], "เอก")

    def test_completed_draft_is_mirrored_as_confirmed_draft(self):
        confirmed = self.handler("ข้อมูลครบแล้ว", self.state(complete_draft()))
        self.assertIsNotNone(confirmed["lead_saved_id"])
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 1)
        self.assertEqual(
            self.connection.execute("SELECT status FROM lead_drafts").fetchone()[0],
            "confirmed",
        )

    def test_correction_clears_old_income_and_confirmation(self):
        old = complete_draft()
        result = self.handler("แก้รายได้ 40000-50000 บาทต่อเดือน", self.state(old, pending=True))
        self.assertIsNone(result["lead_draft"]["income"])
        self.assertEqual(result["lead_draft"]["income_min"], "40000")
        self.assertEqual(result["lead_draft"]["income_max"], "50000")
        self.assertIn("income_confirmation", result["lead_missing"])
        self.assertFalse(result["lead_confirmation_pending"])
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 0)

    def test_cancel_then_new_interest_starts_a_clean_draft(self):
        cancelled = self.handler("ยกเลิก", self.state(complete_draft()))
        new_result = self.handler(
            "สนใจสมัคร ชื่อ บี อาชีพ ครู รายได้ 45000 บาทต่อเดือน เบอร์ 0823456789",
            cancelled,
        )
        self.assertEqual(new_result["lead_draft"]["name"], "บี")
        self.assertNotEqual(new_result["lead_draft"]["name"], "เอก")
        self.assertIsNone(new_result["lead_saved_id"])

    def test_product_change_saves_new_product_automatically(self):
        draft = complete_draft("product-a")
        changed = self.handler("ตรวจสอบอีกผลิตภัณฑ์", {**self.state(draft, pending=True), "active_product_id": "product-b"})
        self.assertEqual(changed["lead_draft"]["product_id"], "product-b")
        self.assertIsNotNone(changed["lead_saved_id"])
        self.assertEqual(self.connection.execute("SELECT product_id FROM leads").fetchone()[0], "product-b")

    def test_same_request_retry_is_idempotent(self):
        original = self.state(complete_draft())
        first = self.handler("ข้อมูลครบแล้ว", original)
        retry = self.handler("ข้อมูลครบแล้ว", original)
        self.assertEqual(first["lead_saved_id"], retry["lead_saved_id"])
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
