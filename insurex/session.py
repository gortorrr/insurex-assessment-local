"""Durable, owner-checked session and message storage for the local demo."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import uuid
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from insurex.db import row_value


@dataclass(frozen=True)
class SessionHandle:
    session_id: str
    owner_id: str
    thread_id: str
    access_token: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(connection: sqlite3.Connection, owner_id: str) -> SessionHandle:
    if not owner_id.strip():
        raise ValueError("owner_id is required")
    try:
        account = connection.execute(
            "SELECT account_role FROM user_accounts WHERE owner_id=?", (owner_id,)
        ).fetchone()
    except Exception:
        account = None
    if account is not None and account["account_role"] == "staff":
        raise PermissionError("staff accounts do not own customer conversations")
    if account is not None and connection.execute(
        "SELECT 1 FROM sessions WHERE owner_id=? AND status='active'", (owner_id,)
    ).fetchone():
        raise ValueError("customer account already has an active conversation")
    session_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    access_token = secrets.token_urlsafe(24)
    now = _now()
    connection.execute(
        """
        INSERT INTO sessions(session_id, owner_id, thread_id, owner_token_hash, status, created_at, last_active_at)
        VALUES (?, ?, ?, ?, 'active', ?, ?)
        """,
        (session_id, owner_id, thread_id, _token_hash(access_token), now, now),
    )
    connection.commit()
    return SessionHandle(session_id, owner_id, thread_id, access_token)


def get_or_create_customer_session(connection: sqlite3.Connection, owner_id: str) -> SessionHandle:
    """Return the account's single active conversation and rotate its browser token."""

    row = connection.execute(
        "SELECT session_id FROM sessions WHERE owner_id=? AND status='active' LIMIT 1",
        (owner_id,),
    ).fetchone()
    if row is None:
        return create_session(connection, owner_id)
    return resume_owned_session(connection, row["session_id"], owner_id)


def _authorize(connection: sqlite3.Connection, session_id: str, access_token: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT * FROM sessions WHERE session_id=? AND owner_token_hash=? AND status='active'",
        (session_id, _token_hash(access_token)),
    ).fetchone()
    if row is None:
        raise PermissionError("session is missing, closed, or not owned by this token")
    return row


def resume_session(
    connection: sqlite3.Connection,
    session_id: str,
    access_token: str,
    *,
    expected_owner_id: str | None = None,
) -> SessionHandle:
    row = _authorize(connection, session_id, access_token)
    if expected_owner_id is not None and row["owner_id"] != expected_owner_id:
        raise PermissionError("session does not belong to the trusted owner context")
    connection.execute("UPDATE sessions SET last_active_at=? WHERE session_id=?", (_now(), session_id))
    connection.commit()
    return SessionHandle(session_id, row["owner_id"], row["thread_id"], access_token)


def resume_owned_session(
    connection: sqlite3.Connection,
    session_id: str,
    owner_id: str,
) -> SessionHandle:
    """Resume a session through the trusted local owner context.

    The assessment requires session separation and history recall, but does
    not require a user-facing resume token.  The local demo therefore uses
    the trusted runtime owner identity to authorize the switch and rotates
    the internal bearer token for the active browser context.
    """

    if not session_id.strip() or not owner_id.strip():
        raise ValueError("session_id and owner_id are required")
    row = connection.execute(
        "SELECT * FROM sessions WHERE session_id=? AND owner_id=? AND status='active'",
        (session_id, owner_id),
    ).fetchone()
    if row is None:
        raise PermissionError("session is missing or not owned by this runtime context")
    access_token = secrets.token_urlsafe(24)
    connection.execute(
        "UPDATE sessions SET owner_token_hash=?, last_active_at=? WHERE session_id=?",
        (_token_hash(access_token), _now(), session_id),
    )
    connection.commit()
    return SessionHandle(session_id, row["owner_id"], row["thread_id"], access_token)


def delete_owned_session(
    connection: sqlite3.Connection,
    session_id: str,
    owner_id: str,
) -> int:
    """Delete chat messages and close one active session for its owner.

    Stored leads are retained as business records; the session is closed so
    its lead rows remain referentially valid and inaccessible through the
    active-session APIs.  The return value is the number of chat messages
    removed.
    """

    if not session_id.strip() or not owner_id.strip():
        raise ValueError("session_id and owner_id are required")
    row = connection.execute(
        "SELECT status FROM sessions WHERE session_id=? AND owner_id=?",
        (session_id, owner_id),
    ).fetchone()
    if row is None or row["status"] != "active":
        raise PermissionError("session is missing, closed, or not owned by this runtime context")
    deleted = connection.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    connection.execute(
        "UPDATE sessions SET status='closed', owner_token_hash='', last_active_at=? WHERE session_id=?",
        (_now(), session_id),
    )
    connection.commit()
    return int(deleted.rowcount)


def append_message(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    access_token: str,
    role: str,
    content: str,
    request_id: str | None = None,
    sender_type: str | None = None,
    sender_account_id: str | None = None,
    sender_label: str | None = None,
) -> int:
    session = _authorize(connection, session_id, access_token)
    if role not in {"user", "assistant", "system"}:
        raise ValueError("invalid message role")
    if not content.strip():
        raise ValueError("message content is required")
    sender_type = sender_type or {"user": "customer", "assistant": "ai", "system": "system"}[role]
    if sender_type not in {"customer", "ai", "staff", "system"}:
        raise ValueError("invalid sender_type")
    if sender_type == "customer":
        sender_account_id = sender_account_id or session["owner_id"]
        sender_label = sender_label or "ลูกค้า"
    elif sender_type == "ai":
        sender_label = sender_label or "AI ผู้ช่วยผลิตภัณฑ์"
    if request_id:
        existing = connection.execute(
            "SELECT message_id, role, content FROM messages WHERE session_id=? AND request_id=?",
            (session_id, request_id),
        ).fetchone()
        if existing:
            if existing["role"] != role or existing["content"] != content:
                raise ValueError("request_id was already used with a different message payload")
            return int(existing["message_id"])
    cursor = connection.execute(
        "INSERT INTO messages(session_id, role, content, created_at, request_id, sender_type, sender_account_id, sender_label) VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING message_id",
        (session_id, role, content, _now(), request_id, sender_type, sender_account_id, sender_label),
    )
    message_id = row_value(cursor.fetchone(), "message_id")
    connection.execute("UPDATE sessions SET last_active_at=? WHERE session_id=?", (_now(), session_id))
    connection.commit()
    return int(message_id)


def load_history(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    access_token: str,
    limit: int = 10,
    product_only: bool = False,
    include_metadata: bool = False,
) -> list[dict[str, str]]:
    _authorize(connection, session_id, access_token)
    if limit <= 0:
        return []
    columns = "role, content, created_at, sender_type, sender_account_id, sender_label" + (", metadata_json" if include_metadata else "")
    scope = " AND purpose='product'" if product_only else ""
    rows = connection.execute(
        f"SELECT {columns} FROM messages WHERE session_id=?{scope} ORDER BY message_id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [
        {"role": row["role"], "content": row["content"], "created_at": row["created_at"],
         "sender_type": row["sender_type"], "sender_account_id": row["sender_account_id"],
         "sender_label": row["sender_label"],
         **({"metadata": json.loads(row['metadata_json'])} if include_metadata else {})}
        for row in reversed(rows)
    ]


def _require_staff(connection: sqlite3.Connection, staff_owner_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT owner_id, username FROM user_accounts WHERE owner_id=? AND account_role='staff'",
        (staff_owner_id,),
    ).fetchone()
    if row is None:
        raise PermissionError("staff account is required")
    return row


def list_customer_conversations(connection: sqlite3.Connection, staff_owner_id: str) -> list[dict]:
    """List each customer's single conversation for the staff inbox."""

    _require_staff(connection, staff_owner_id)
    rows = connection.execute(
        """SELECT a.owner_id, a.username, s.session_id, s.last_active_at,
                  COUNT(m.message_id) AS message_count,
                  MAX(m.created_at) AS last_message_at
           FROM user_accounts a
           LEFT JOIN sessions s ON s.owner_id=a.owner_id AND s.status='active'
           LEFT JOIN messages m ON m.session_id=s.session_id
           WHERE a.account_role='customer'
           GROUP BY a.owner_id, a.username, s.session_id, s.last_active_at
           ORDER BY COALESCE(MAX(m.created_at), s.last_active_at) DESC, a.username"""
    ).fetchall()
    return [dict(row) for row in rows]


def get_open_handoff(connection: sqlite3.Connection, session_id: str) -> dict | None:
    row = connection.execute(
        """SELECT h.*, a.username AS assigned_staff_name
           FROM handoff_cases h
           LEFT JOIN user_accounts a ON a.owner_id=h.assigned_staff_id
           WHERE h.session_id=? AND h.status IN ('pending','active')
           ORDER BY h.case_id DESC LIMIT 1""",
        (session_id,),
    ).fetchone()
    return dict(row) if row else None


def request_handoff(
    connection: sqlite3.Connection, *, session_id: str, access_token: str,
    trigger_reason: str, trigger_request_id: str | None = None,
) -> int:
    """Queue one owner-authorized conversation for human assistance."""

    _authorize(connection, session_id, access_token)
    existing = get_open_handoff(connection, session_id)
    if existing:
        return int(existing["case_id"])
    trigger_message_id = None
    if trigger_request_id:
        row = connection.execute(
            "SELECT message_id FROM messages WHERE session_id=? AND request_id=?",
            (session_id, trigger_request_id),
        ).fetchone()
        trigger_message_id = row["message_id"] if row else None
    cursor = connection.execute(
        """INSERT INTO handoff_cases(
               session_id,status,trigger_reason,trigger_message_id,created_at
           ) VALUES (?, 'pending', ?, ?, ?) RETURNING case_id""",
        (session_id, trigger_reason, trigger_message_id, _now()),
    )
    case_id = int(row_value(cursor.fetchone(), "case_id"))
    connection.commit()
    return case_id


def list_handoff_queue(connection: sqlite3.Connection, staff_owner_id: str) -> list[dict]:
    """Return only unresolved customer conversations visible to staff."""

    _require_staff(connection, staff_owner_id)
    rows = connection.execute(
        """SELECT h.case_id,h.session_id,h.status,h.trigger_reason,h.created_at,
                  h.accepted_at,h.assigned_staff_id,
                  customer.owner_id AS customer_id,customer.username AS customer_name,
                  staff.username AS assigned_staff_name,
                  COUNT(m.message_id) AS message_count,
                  MAX(m.created_at) AS last_message_at
           FROM handoff_cases h
           JOIN sessions s ON s.session_id=h.session_id AND s.status='active'
           JOIN user_accounts customer ON customer.owner_id=s.owner_id AND customer.account_role='customer'
           LEFT JOIN user_accounts staff ON staff.owner_id=h.assigned_staff_id
           LEFT JOIN messages m ON m.session_id=h.session_id
           WHERE h.status='pending'
              OR (h.status='active' AND h.assigned_staff_id=?)
           GROUP BY h.case_id,h.session_id,h.status,h.trigger_reason,h.created_at,
                    h.accepted_at,h.assigned_staff_id,customer.owner_id,customer.username,staff.username
           ORDER BY CASE h.status WHEN 'pending' THEN 0 ELSE 1 END,
                    COALESCE(MAX(m.created_at),h.created_at) DESC""",
        (staff_owner_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def accept_handoff(connection: sqlite3.Connection, *, staff_owner_id: str, case_id: int) -> dict:
    staff = _require_staff(connection, staff_owner_id)
    cursor = connection.execute(
        """UPDATE handoff_cases SET status='active',assigned_staff_id=?,accepted_at=?
           WHERE case_id=? AND status='pending' AND assigned_staff_id IS NULL""",
        (staff_owner_id, _now(), case_id),
    )
    if cursor.rowcount != 1:
        connection.rollback()
        row = connection.execute("SELECT status,assigned_staff_id FROM handoff_cases WHERE case_id=?", (case_id,)).fetchone()
        if row and row["status"] == "active" and row["assigned_staff_id"] == staff_owner_id:
            return {"case_id": case_id, "status": "active", "assigned_staff_name": staff["username"]}
        raise PermissionError("case was already accepted or is no longer open")
    connection.commit()
    return {"case_id": case_id, "status": "active", "assigned_staff_name": staff["username"]}


def resolve_handoff(
    connection: sqlite3.Connection, *, staff_owner_id: str, case_id: int,
    resolution_note: str = "resolved by sales staff",
) -> None:
    _require_staff(connection, staff_owner_id)
    cursor = connection.execute(
        """UPDATE handoff_cases SET status='resolved',resolved_at=?,resolution_note=?
           WHERE case_id=? AND status='active' AND assigned_staff_id=?""",
        (_now(), resolution_note.strip() or "resolved by sales staff", case_id, staff_owner_id),
    )
    if cursor.rowcount != 1:
        connection.rollback()
        raise PermissionError("only the assigned staff member can resolve this active case")
    connection.commit()


def load_history_for_staff(connection: sqlite3.Connection, *, staff_owner_id: str, session_id: str, limit: int = 100) -> list[dict]:
    _require_staff(connection, staff_owner_id)
    row = connection.execute(
        "SELECT 1 FROM sessions s JOIN user_accounts a ON a.owner_id=s.owner_id WHERE s.session_id=? AND a.account_role='customer'",
        (session_id,),
    ).fetchone()
    if row is None:
        raise PermissionError("customer conversation not found")
    rows = connection.execute(
        "SELECT role,content,created_at,sender_type,sender_account_id,sender_label,metadata_json FROM messages WHERE session_id=? ORDER BY message_id DESC LIMIT ?",
        (session_id, limit),
    ).fetchall()
    return [{**dict(row), "metadata": json.loads(row["metadata_json"] or "{}")} for row in reversed(rows)]


def append_staff_message(connection: sqlite3.Connection, *, staff_owner_id: str, session_id: str, content: str) -> int:
    staff = _require_staff(connection, staff_owner_id)
    if not content.strip():
        raise ValueError("message content is required")
    customer = connection.execute(
        "SELECT 1 FROM sessions s JOIN user_accounts a ON a.owner_id=s.owner_id WHERE s.session_id=? AND s.status='active' AND a.account_role='customer'",
        (session_id,),
    ).fetchone()
    if customer is None:
        raise PermissionError("customer conversation not found")
    handoff = get_open_handoff(connection, session_id)
    if not handoff or handoff["status"] != "active" or handoff["assigned_staff_id"] != staff_owner_id:
        raise PermissionError("staff must accept this handoff before replying")
    cursor = connection.execute(
        "INSERT INTO messages(session_id,role,content,created_at,request_id,sender_type,sender_account_id,sender_label) VALUES (?, 'assistant', ?, ?, ?, 'staff', ?, ?) RETURNING message_id",
        (session_id, content.strip(), _now(), f"staff-{uuid.uuid4().hex}", staff_owner_id, staff["username"]),
    )
    message_id = int(row_value(cursor.fetchone(), "message_id"))
    connection.execute("UPDATE sessions SET last_active_at=? WHERE session_id=?", (_now(), session_id))
    connection.commit()
    return message_id


def annotate_turn(connection, session_id, access_token, request_id, *, product_turn, citations):
    _authorize(connection, session_id, access_token)
    connection.execute("UPDATE messages SET purpose=? WHERE session_id=? AND request_id IN (?,?)",
                       ('product' if product_turn else 'private', session_id, request_id, request_id+':assistant'))
    connection.execute("UPDATE messages SET metadata_json=? WHERE session_id=? AND request_id=?",
                       (json.dumps({'citations': citations}), session_id, request_id+':assistant'))
    connection.commit()


def list_owned_sessions(
    connection: sqlite3.Connection,
    owner_id: str,
    *,
    limit: int = 30,
) -> list[dict[str, str | int]]:
    """List chat summaries for one trusted owner without exposing tokens."""

    if not owner_id.strip():
        raise ValueError("owner_id is required")
    if limit <= 0:
        return []
    rows = connection.execute(
        """
        SELECT
            s.session_id,
            s.created_at,
            s.last_active_at,
            s.status,
            COALESCE(
                (
                    SELECT SUBSTR(m.content, 1, 72)
                    FROM messages AS m
                    WHERE m.session_id = s.session_id AND m.role = 'user'
                    ORDER BY m.message_id ASC
                    LIMIT 1
                ),
                'การสนทนาใหม่'
            ) AS title,
            (
                SELECT COUNT(*)
                FROM messages AS mc
                WHERE mc.session_id = s.session_id
            ) AS message_count
            ,COALESCE(
                (
                    SELECT MAX(ml.created_at)
                    FROM messages AS ml
                    WHERE ml.session_id = s.session_id
                ),
                s.created_at
            ) AS conversation_last_message_at
        FROM sessions AS s
        WHERE s.owner_id=?
          AND s.status='active'
          AND EXISTS (
              SELECT 1
              FROM messages AS me
              WHERE me.session_id = s.session_id
          )
        ORDER BY conversation_last_message_at DESC, s.created_at DESC
        LIMIT ?
        """,
        (owner_id, limit),
    ).fetchall()
    return [
        {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "last_active_at": row["last_active_at"],
            "conversation_last_message_at": row["conversation_last_message_at"],
            "status": row["status"],
            "title": row["title"],
            "message_count": int(row["message_count"]),
        }
        for row in rows
    ]
