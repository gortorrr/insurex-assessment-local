"""Create sanitized lead and session evidence using synthetic values."""

from __future__ import annotations

import json
from pathlib import Path

from insurex.db import connect_assistant_local as connect
from insurex.leads import LeadDraft, extract_lead_draft, save_lead, validate_lead
from insurex.session import append_message, create_session, load_history, resume_session


def main() -> None:
    database = Path("db/demo_bonus.sqlite")
    connection = connect(database)
    try:
        session_a = create_session(connection, "synthetic-owner-a")
        session_b = create_session(connection, "synthetic-owner-b")
        append_message(connection, session_id=session_a.session_id, access_token=session_a.access_token, role="user", content="สนใจ product-a ครับ", request_id="a-1")
        append_message(connection, session_id=session_b.session_id, access_token=session_b.access_token, role="user", content="สนใจ product-b ครับ", request_id="b-1")
        text = "ชื่อผู้ทดสอบ ทำงานเป็น Analyst รายได้ 45000 บาทต่อเดือน เบอร์ 0812345678"
        draft = extract_lead_draft(text, product_id="product-a")
        lead = validate_lead(draft.model_copy(update={"confirmation_status": "confirmed"}))
        request_id = f"lead-demo-{session_a.session_id}"
        lead_id = save_lead(connection, session_id=session_a.session_id, access_token=session_a.access_token, request_id=request_id, lead=lead)
        # The second call demonstrates request-level idempotency.
        duplicate_id = save_lead(connection, session_id=session_a.session_id, access_token=session_a.access_token, request_id=request_id, lead=lead)
        safe_validated = lead.model_dump(mode="json")
        safe_validated["phone"] = "081•••••78"
        connection.close()
        reopened = connect(database)
        try:
            history_a = load_history(reopened, session_id=session_a.session_id, access_token=session_a.access_token)
            history_b = load_history(reopened, session_id=session_b.session_id, access_token=session_b.access_token)
            resumed_a = resume_session(reopened, session_id=session_a.session_id, access_token=session_a.access_token)
        finally:
            reopened.close()
        report = {
            "synthetic": True,
            "lead": {
                "lead_id": lead_id,
                "duplicate_request_same_id": lead_id == duplicate_id,
                "validated_json": safe_validated,
                "phone_masked": "081•••••78",
            },
            "sessions": {
                "a": {"session_id": session_a.session_id, "thread_id": resumed_a.thread_id, "history": history_a},
                "b": {"session_id": session_b.session_id, "thread_id": session_b.thread_id, "history": history_b},
                "cross_session_content_check": history_a[0]["content"] != history_b[0]["content"],
                "restart_resume": True,
            },
        }
        Path("rag/reports/demo/lead_session_demo.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"database": str(database), "report": "rag/reports/demo/lead_session_demo.json", "synthetic": True}, ensure_ascii=False, indent=2))
    finally:
        # The connection is already closed above on the normal path.
        if connection:
            try:
                connection.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
