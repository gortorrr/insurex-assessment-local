from decimal import Decimal
from pathlib import Path
import json
import sqlite3
import tempfile
import unittest

from insurex.db import connect_local as connect
from insurex.leads import LeadDraft, ValidatedLead, list_leads_for_session, load_account_lead_draft, save_lead, validate_lead
from insurex.session import append_message, create_session, delete_owned_session, list_owned_sessions, load_history, resume_owned_session, resume_session


class LeadSessionTests(unittest.TestCase):
    def test_legacy_session_leads_migrate_to_latest_account_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.sqlite"
            legacy = sqlite3.connect(path)
            legacy.executescript("""
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL UNIQUE, owner_token_hash TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL,
                    last_active_at TEXT NOT NULL
                );
                CREATE TABLE leads (
                    lead_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, product_id TEXT NOT NULL,
                    name TEXT NOT NULL, occupation TEXT NOT NULL,
                    income_thb NUMERIC NOT NULL, income_period TEXT NOT NULL,
                    phone_normalized TEXT NOT NULL, request_id TEXT NOT NULL UNIQUE,
                    payload_hash TEXT NOT NULL, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE lead_drafts (
                    session_id TEXT PRIMARY KEY, product_id TEXT, name TEXT,
                    occupation TEXT, income_value TEXT, income_min TEXT,
                    income_max TEXT, income_currency TEXT, income_period TEXT,
                    phone_raw TEXT, phone_normalized TEXT, status TEXT NOT NULL,
                    draft_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
            """)
            for suffix in ("old", "new"):
                legacy.execute(
                    "INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                    (suffix, "customer-a", f"thread-{suffix}", f"token-{suffix}",
                     "active", suffix, suffix),
                )
            legacy.execute(
                "INSERT INTO leads(session_id,product_id,name,occupation,income_thb,income_period,phone_normalized,request_id,payload_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("old", "p-old", "ข้อมูลเก่า", "งานเก่า", 10000, "monthly_thb",
                 "0811111111", "req-old", "hash-old", "1", "1"),
            )
            legacy.execute(
                "INSERT INTO leads(session_id,product_id,name,occupation,income_thb,income_period,phone_normalized,request_id,payload_hash,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("new", "p-new", "ข้อมูลล่าสุด", "งานใหม่", 20000, "monthly_thb",
                 "0822222222", "req-new", "hash-new", "2", "2"),
            )
            for session_id, name, updated in (("old", "ข้อมูลเก่า", "1"), ("new", "ข้อมูลล่าสุด", "2")):
                payload = json.dumps({
                    "name": name, "occupation": None, "income": None,
                    "income_min": None, "income_max": None,
                    "income_currency": None, "income_period": None,
                    "phone": None, "product_id": None, "raw_evidence": {},
                    "confirmation_status": "unconfirmed",
                }, ensure_ascii=False)
                legacy.execute(
                    "INSERT INTO lead_drafts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (session_id, None, name, None, None, None, None, None, None,
                     None, None, "unconfirmed", payload, updated),
                )
            legacy.commit()
            legacy.close()

            migrated = connect(path)
            try:
                lead = migrated.execute(
                    "SELECT owner_id, session_id, name FROM leads"
                ).fetchone()
                self.assertEqual(tuple(lead), ("customer-a", "new", "ข้อมูลล่าสุด"))
                draft = migrated.execute(
                    "SELECT owner_id, last_session_id, name FROM lead_drafts"
                ).fetchone()
                self.assertEqual(tuple(draft), ("customer-a", "new", "ข้อมูลล่าสุด"))
                self.assertEqual(
                    migrated.execute("SELECT COUNT(*) FROM lead_request_log").fetchone()[0],
                    2,
                )
            finally:
                migrated.close()

    def test_valid_lead_saves_once_and_invalid_lead_does_not_insert(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "business.sqlite")
            try:
                session = create_session(connection, "owner-a")
                valid = validate_lead(
                    LeadDraft(
                        name="สมชาย",
                        occupation="Engineer",
                        income=Decimal("45000"),
                        income_currency="THB",
                        income_period="monthly_thb",
                        phone="0812345678",
                        product_id="product-a",
                        confirmation_status="confirmed",
                    )
                )
                first_id = save_lead(connection, session_id=session.session_id, access_token=session.access_token, request_id="req-1", lead=valid)
                second_id = save_lead(connection, session_id=session.session_id, access_token=session.access_token, request_id="req-1", lead=valid)
                self.assertEqual(first_id, second_id)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 1)
                self.assertEqual(connection.execute("SELECT income_thb FROM leads").fetchone()[0], 45000)
                self.assertEqual(len(list_leads_for_session(connection, session_id=session.session_id, access_token=session.access_token)), 1)
                with self.assertRaises(PermissionError):
                    list_leads_for_session(connection, session_id=session.session_id, access_token="wrong")
                changed = valid.model_copy(update={"phone": "0812345679"})
                with self.assertRaisesRegex(ValueError, "different lead payload"):
                    save_lead(connection, session_id=session.session_id, access_token=session.access_token, request_id="req-1", lead=changed)
                with self.assertRaises(PermissionError):
                    save_lead(connection, session_id=session.session_id, access_token="wrong", request_id="req-2", lead=valid)
                invalid_constructed = ValidatedLead.model_construct(
                    name="Bad", occupation="Engineer", income=Decimal("45000"),
                    income_currency="THB", income_period="monthly_thb", phone="garbage",
                    product_id="product-a", confirmation_status="confirmed",
                )
                with self.assertRaises(ValueError):
                    save_lead(connection, session_id=session.session_id, access_token=session.access_token, request_id="req-3", lead=invalid_constructed)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 1)
                with self.assertRaises(ValueError):
                    validate_lead(
                        LeadDraft(
                            name="สมหญิง",
                            occupation="Accountant",
                            income=Decimal("0"),
                        income_period="monthly_thb",
                        product_id="product-a",
                        confirmation_status="confirmed",
                        )
                    )
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 1)
            finally:
                connection.close()

    def test_sessions_are_isolated_and_survive_connection_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "session.sqlite"
            connection = connect(path)
            session_a = create_session(connection, "owner-a")
            session_b = create_session(connection, "owner-b")
            append_message(connection, session_id=session_a.session_id, access_token=session_a.access_token, role="user", content="Product A", request_id="a-1")
            self.assertEqual(
                append_message(connection, session_id=session_a.session_id, access_token=session_a.access_token, role="user", content="Product A", request_id="a-1"),
                1,
            )
            with self.assertRaisesRegex(ValueError, "different message payload"):
                append_message(connection, session_id=session_a.session_id, access_token=session_a.access_token, role="user", content="Changed", request_id="a-1")
            append_message(connection, session_id=session_b.session_id, access_token=session_b.access_token, role="user", content="Product B", request_id="b-1")
            connection.close()

            reopened = connect(path)
            try:
                self.assertEqual(load_history(reopened, session_id=session_a.session_id, access_token=session_a.access_token)[0]["content"], "Product A")
                self.assertEqual(load_history(reopened, session_id=session_b.session_id, access_token=session_b.access_token)[0]["content"], "Product B")
                with self.assertRaises(PermissionError):
                    resume_session(reopened, session_id=session_a.session_id, access_token=session_b.access_token)
            finally:
                reopened.close()

    def test_one_account_has_one_shared_lead_profile_across_chats(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "account-lead.sqlite")
            try:
                first_chat = create_session(connection, "customer-a")
                second_chat = create_session(connection, "customer-a")
                other_customer = create_session(connection, "customer-b")

                original = validate_lead(LeadDraft(
                    name="ลูกค้าเอ", occupation="ครู", income=Decimal("30000"),
                    income_currency="THB", income_period="monthly_thb",
                    phone="0811111111", product_id="product-a",
                    confirmation_status="confirmed",
                ))
                lead_id = save_lead(
                    connection, session_id=first_chat.session_id,
                    access_token=first_chat.access_token, request_id="account-a-1",
                    lead=original,
                )
                updated = original.model_copy(update={"occupation": "อาจารย์"})
                updated_id = save_lead(
                    connection, session_id=second_chat.session_id,
                    access_token=second_chat.access_token, request_id="account-a-2",
                    lead=updated,
                )
                self.assertEqual(updated_id, lead_id)
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM leads WHERE owner_id='customer-a'"
                ).fetchone()[0], 1)
                self.assertEqual(
                    list_leads_for_session(
                        connection, session_id=first_chat.session_id,
                        access_token=first_chat.access_token,
                    )[0]["occupation"],
                    "อาจารย์",
                )

                other = original.model_copy(update={
                    "name": "ลูกค้าบี", "phone": "0822222222"
                })
                save_lead(
                    connection, session_id=other_customer.session_id,
                    access_token=other_customer.access_token,
                    request_id="account-b-1", lead=other,
                )
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM leads").fetchone()[0], 2)
                self.assertEqual(
                    list_leads_for_session(
                        connection, session_id=other_customer.session_id,
                        access_token=other_customer.access_token,
                    )[0]["name"],
                    "ลูกค้าบี",
                )
            finally:
                connection.close()

    def test_account_profile_falls_back_to_completed_lead_without_draft(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "lead-fallback.sqlite")
            try:
                first_chat = create_session(connection, "customer-a")
                later_chat = create_session(connection, "customer-a")
                lead = validate_lead(LeadDraft(
                    name="ลูกค้าเอ", occupation="ครู", income=Decimal("31000"),
                    income_currency="THB", income_period="monthly_thb",
                    phone="0811111111", product_id="product-a",
                    confirmation_status="confirmed",
                ))
                save_lead(
                    connection, session_id=first_chat.session_id,
                    access_token=first_chat.access_token, request_id="fallback-1",
                    lead=lead,
                )
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM lead_drafts").fetchone()[0], 0)
                restored = load_account_lead_draft(
                    connection, session_id=later_chat.session_id,
                    access_token=later_chat.access_token,
                )
                self.assertIsNotNone(restored)
                self.assertEqual(restored.name, "ลูกค้าเอ")
                self.assertEqual(restored.income, Decimal("31000"))
                self.assertEqual(restored.confirmation_status, "confirmed")
            finally:
                connection.close()

    def test_chat_history_lists_only_owned_sessions_without_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "history.sqlite")
            try:
                owner_session = create_session(connection, "owner-a")
                empty_session = create_session(connection, "owner-a")
                other_session = create_session(connection, "owner-b")
                append_message(
                    connection,
                    session_id=owner_session.session_id,
                    access_token=owner_session.access_token,
                    role="user",
                    content="ถามเรื่องความคุ้มครอง",
                    request_id="history-1",
                )
                append_message(
                    connection,
                    session_id=other_session.session_id,
                    access_token=other_session.access_token,
                    role="user",
                    content="ข้อมูลของอีก owner",
                    request_id="history-2",
                )
                rows = list_owned_sessions(connection, "owner-a")
                self.assertEqual([row["session_id"] for row in rows], [owner_session.session_id])
                self.assertNotIn(empty_session.session_id, [row["session_id"] for row in rows])
                self.assertEqual(rows[0]["title"], "ถามเรื่องความคุ้มครอง")
                self.assertEqual(rows[0]["message_count"], 1)
                self.assertNotIn("access_token", rows[0])
                self.assertNotIn(owner_session.access_token, str(rows))
            finally:
                connection.close()

    def test_chat_history_orders_by_latest_message_not_session_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "history-order.sqlite")
            try:
                older_conversation = create_session(connection, "owner-a")
                newer_conversation = create_session(connection, "owner-a")
                append_message(
                    connection,
                    session_id=older_conversation.session_id,
                    access_token=older_conversation.access_token,
                    role="user",
                    content="ข้อความเก่ากว่า",
                    request_id="order-old",
                )
                append_message(
                    connection,
                    session_id=newer_conversation.session_id,
                    access_token=newer_conversation.access_token,
                    role="user",
                    content="ข้อความใหม่กว่า",
                    request_id="order-new",
                )
                # Simulate reopening the older chat after the newer message.
                # Session access time must not control the history order.
                connection.execute(
                    "UPDATE messages SET created_at=? WHERE request_id=?",
                    ("2026-09-18T10:00:00+00:00", "order-old"),
                )
                connection.execute(
                    "UPDATE messages SET created_at=? WHERE request_id=?",
                    ("2026-09-19T10:00:00+00:00", "order-new"),
                )
                connection.execute(
                    "UPDATE sessions SET last_active_at=? WHERE session_id=?",
                    ("2099-01-01T00:00:00+00:00", older_conversation.session_id),
                )
                connection.execute(
                    "UPDATE sessions SET last_active_at=? WHERE session_id=?",
                    ("2000-01-01T00:00:00+00:00", newer_conversation.session_id),
                )
                connection.commit()
                rows = list_owned_sessions(connection, "owner-a")
                self.assertEqual(
                    [row["session_id"] for row in rows],
                    [newer_conversation.session_id, older_conversation.session_id],
                )
                self.assertEqual(rows[0]["conversation_last_message_at"], "2026-09-19T10:00:00+00:00")
            finally:
                connection.close()

    def test_chat_history_can_resume_by_trusted_owner_without_user_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "history-owner.sqlite")
            try:
                original = create_session(connection, "owner-a")
                append_message(
                    connection,
                    session_id=original.session_id,
                    access_token=original.access_token,
                    role="user",
                    content="ข้อความที่ต้องจำหลังเริ่ม process ใหม่",
                    request_id="owner-resume-1",
                )
                resumed = resume_owned_session(connection, original.session_id, "owner-a")
                self.assertEqual(resumed.session_id, original.session_id)
                self.assertNotEqual(resumed.access_token, original.access_token)
                self.assertEqual(
                    load_history(
                        connection,
                        session_id=resumed.session_id,
                        access_token=resumed.access_token,
                    )[0]["content"],
                    "ข้อความที่ต้องจำหลังเริ่ม process ใหม่",
                )
                with self.assertRaises(PermissionError):
                    resume_owned_session(connection, original.session_id, "owner-b")
            finally:
                connection.close()

    def test_delete_chat_history_is_owner_scoped_and_closes_only_target_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            connection = connect(Path(temporary) / "history-delete.sqlite")
            try:
                owner_session = create_session(connection, "owner-a")
                other_session = create_session(connection, "owner-b")
                append_message(
                    connection,
                    session_id=owner_session.session_id,
                    access_token=owner_session.access_token,
                    role="user",
                    content="ข้อความที่จะลบ",
                    request_id="delete-1",
                )
                append_message(
                    connection,
                    session_id=other_session.session_id,
                    access_token=other_session.access_token,
                    role="user",
                    content="ข้อความของ owner อื่น",
                    request_id="delete-2",
                )
                with self.assertRaises(PermissionError):
                    delete_owned_session(connection, owner_session.session_id, "owner-b")
                self.assertEqual(delete_owned_session(connection, owner_session.session_id, "owner-a"), 1)
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM messages WHERE session_id=?",
                        (owner_session.session_id,),
                    ).fetchone()[0],
                    0,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT status FROM sessions WHERE session_id=?",
                        (owner_session.session_id,),
                    ).fetchone()[0],
                    "closed",
                )
                self.assertEqual([row["session_id"] for row in list_owned_sessions(connection, "owner-a")], [])
                self.assertEqual([row["session_id"] for row in list_owned_sessions(connection, "owner-b")], [other_session.session_id])
                with self.assertRaises(PermissionError):
                    delete_owned_session(connection, owner_session.session_id, "owner-a")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
