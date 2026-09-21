import unittest
from decimal import Decimal

from insurex.leads import (
    LeadDraft,
    confirm_lead,
    detect_product_interest,
    extract_lead_draft,
    lead_route,
    merge_lead_drafts,
    missing_lead_fields,
    normalize_phone,
    validate_lead,
)


class LeadTests(unittest.TestCase):
    def test_interest_detection_and_extraction(self):
        text = "ผมชื่อสมชาย ทำงานเป็น Software Engineer รายได้ 45,000 บาทต่อเดือน สนใจครับ เบอร์ 081-234-5678"
        self.assertTrue(detect_product_interest(text))
        draft = extract_lead_draft(text, product_id="product-a")
        draft = draft.model_copy(update={"confirmation_status": "confirmed"})
        self.assertEqual(draft.name, "สมชาย")
        self.assertEqual(draft.occupation, "Software Engineer")
        self.assertEqual(draft.income, Decimal("45000"))
        self.assertEqual(draft.income_period, "monthly_thb")
        self.assertEqual(normalize_phone(draft.phone), "0812345678")
        lead = validate_lead(draft)
        self.assertEqual(lead.product_id, "product-a")

    def test_missing_phone_is_not_invented_or_validated(self):
        draft = LeadDraft(
            name="สมชาย",
            occupation="Engineer",
            income=Decimal("45000"),
            income_period="monthly_thb",
            product_id="product-a",
            confirmation_status="confirmed",
        )
        with self.assertRaisesRegex(ValueError, "phone is required"):
            validate_lead(draft)

    def test_annual_income_requires_confirmation(self):
        draft = LeadDraft(
            name="สมชาย",
            occupation="Engineer",
            income=Decimal("540000"),
            income_period="annual_thb",
            phone="+66812345678",
            product_id="product-a",
            confirmation_status="confirmed",
        )
        with self.assertRaisesRegex(ValueError, "monthly_thb"):
            validate_lead(draft)

    def test_invalid_phone_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_phone("12345")

    def test_name_preserves_thai_letter_kor_khai_and_raw_evidence(self):
        draft = extract_lead_draft("ชื่อ เอก อาชีพ Engineer รายได้ 45,000 บาทต่อเดือน เบอร์ 0812345678", product_id="product-a")
        self.assertEqual(draft.name, "เอก")
        self.assertEqual(draft.raw_evidence["name"], "เอก")

    def test_negative_or_question_interest_is_not_confirmation(self):
        self.assertFalse(detect_product_interest("ไม่สนใจสมัครครับ"))
        self.assertFalse(detect_product_interest("ขอยกเลิกการสมัคร"))
        self.assertFalse(detect_product_interest("ขอทราบวิธีสมัคร"))
        self.assertTrue(detect_product_interest("ผมสนใจสมัครครับ"))
        self.assertEqual(lead_route("ขอทราบวิธีสมัคร"), "general")
        self.assertEqual(lead_route("ผมสนใจสมัครครับ"), "lead")

    def test_income_range_requires_confirmation_and_never_picks_endpoint(self):
        draft = extract_lead_draft("ชื่อ เอก อาชีพ Engineer รายได้ 30,000–50,000 บาทต่อเดือน เบอร์ 0812345678", product_id="product-a")
        self.assertIsNone(draft.income)
        self.assertEqual(draft.income_min, Decimal("30000"))
        self.assertEqual(draft.income_max, Decimal("50000"))
        with self.assertRaisesRegex(ValueError, "confirmation"):
            validate_lead(draft)

    def test_usd_annual_or_unknown_units_are_not_assumed_thb_monthly(self):
        usd = extract_lead_draft("name: Ek occupation: Engineer income: $30,000 per year phone: 0812345678", product_id="product-a")
        self.assertEqual(usd.income_currency, "USD")
        self.assertEqual(usd.income_period, "annual_usd")
        usd = usd.model_copy(update={"confirmation_status": "confirmed"})
        with self.assertRaisesRegex(ValueError, "THB|monthly_thb"):
            validate_lead(usd)
        unknown = extract_lead_draft("name: Ek occupation: Engineer income: 30000 phone: 0812345678", product_id="product-a")
        unknown = unknown.model_copy(update={"confirmation_status": "confirmed"})
        self.assertIsNone(unknown.income_currency)
        with self.assertRaises(ValueError):
            validate_lead(unknown)

    def test_draft_merge_and_confirmation_are_explicit(self):
        first = LeadDraft(name="เอก", product_id="product-a", raw_evidence={"name": "เอก"})
        second = LeadDraft(
            occupation="Engineer", income=Decimal("45000"), income_currency="THB",
            income_period="monthly_thb", phone="0812345678",
            raw_evidence={"income": "45,000 บาทต่อเดือน"},
        )
        merged = merge_lead_drafts(first, second)
        self.assertEqual(missing_lead_fields(merged), [])
        confirmed = confirm_lead(merged)
        self.assertEqual(confirmed.confirmation_status, "confirmed")


if __name__ == "__main__":
    unittest.main()
