import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from insurex.auth import authenticate, register
from insurex.db import connect_local
from insurex.leads import lead_profile_status
from insurex.session import (
    append_message,
    append_staff_message,
    accept_handoff,
    get_or_create_customer_session,
    list_handoff_queue,
    load_history,
    load_history_for_staff,
    request_handoff,
    resolve_handoff,
)


class StaffWorkspaceTests(unittest.TestCase):
    def test_only_one_staff_account_can_claim_and_resolve_a_case(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_local(Path(directory) / "app.sqlite")
            try:
                customer = register(connection, "customer-c", "Customer-C-2026!")
                staff_a = register(connection, "staff-one", "Staff-One-2026!", account_role="staff")
                staff_b = register(connection, "staff-two", "Staff-Two-2026!", account_role="staff")
                session = get_or_create_customer_session(connection, customer)
                case_id = request_handoff(
                    connection, session_id=session.session_id, access_token=session.access_token,
                    trigger_reason="retrieval insufficient",
                )
                accept_handoff(connection, staff_owner_id=staff_a, case_id=case_id)
                self.assertEqual(list_handoff_queue(connection, staff_b), [])
                with self.assertRaises(PermissionError):
                    accept_handoff(connection, staff_owner_id=staff_b, case_id=case_id)
                with self.assertRaises(PermissionError):
                    append_staff_message(
                        connection, staff_owner_id=staff_b,
                        session_id=session.session_id, content="ไม่ควรส่งได้",
                    )
                resolve_handoff(connection, staff_owner_id=staff_a, case_id=case_id)
            finally:
                connection.close()

    def test_customer_has_one_conversation_and_staff_reply_is_attributed(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_local(Path(directory) / "app.sqlite")
            try:
                customer = register(connection, "customer-a", "Customer-A-2026!")
                staff = register(connection, "staff-a", "Staff-A-2026!", account_role="staff")
                first = get_or_create_customer_session(connection, customer)
                second = get_or_create_customer_session(connection, customer)
                self.assertEqual(first.session_id, second.session_id)
                append_message(
                    connection, session_id=first.session_id, access_token=second.access_token,
                    role="user", content="สนใจคุ้มออมสุข", sender_type="customer",
                )
                case_id = request_handoff(
                    connection, session_id=first.session_id, access_token=second.access_token,
                    trigger_reason="no_answer",
                )
                self.assertEqual(list_handoff_queue(connection, staff)[0]["status"], "pending")
                accept_handoff(connection, staff_owner_id=staff, case_id=case_id)
                append_staff_message(
                    connection, staff_owner_id=staff, session_id=first.session_id,
                    content="ยินดีให้ข้อมูลครับ",
                )
                inbox = list_handoff_queue(connection, staff)
                self.assertEqual(inbox[0]["session_id"], first.session_id)
                staff_view = load_history_for_staff(
                    connection, staff_owner_id=staff, session_id=first.session_id
                )
                self.assertEqual([item["sender_type"] for item in staff_view], ["customer", "staff"])
                customer_view = load_history(
                    connection, session_id=first.session_id, access_token=second.access_token
                )
                self.assertEqual(customer_view[-1]["sender_label"], "staff-a")
                resolve_handoff(connection, staff_owner_id=staff, case_id=case_id)
                self.assertEqual(list_handoff_queue(connection, staff), [])
                self.assertEqual(authenticate(connection, "staff-a", "Staff-A-2026!")["account_role"], "staff")
            finally:
                connection.close()

    def test_six_month_profile_status_uses_verified_updated_then_created(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = connect_local(Path(directory) / "app.sqlite")
            try:
                owner = register(connection, "customer-b", "Customer-B-2026!")
                session = get_or_create_customer_session(connection, owner)
                old = (datetime.now(UTC) - timedelta(days=184)).isoformat()
                connection.execute(
                    """INSERT INTO leads(owner_id,session_id,product_id,name,occupation,income_thb,
                       income_period,phone_normalized,request_id,payload_hash,created_at,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (owner, session.session_id, "p", "A", "B", 10000, "monthly_thb",
                     "0812345678", "r", "h", old, old),
                )
                connection.commit()
                self.assertTrue(lead_profile_status(connection, owner)["stale"])
                connection.execute(
                    "UPDATE leads SET profile_verified_at=? WHERE owner_id=?",
                    (datetime.now(UTC).isoformat(), owner),
                )
                connection.commit()
                self.assertFalse(lead_profile_status(connection, owner)["stale"])
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
