"""Structured lead extraction, validation, ownership checks, and persistence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from insurex.db import row_value


class LeadDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    occupation: str | None = None
    income: Decimal | None = None
    income_min: Decimal | None = None
    income_max: Decimal | None = None
    income_currency: str | None = None
    income_period: str | None = None
    phone: str | None = None
    product_id: str | None = None
    raw_evidence: dict[str, str] = Field(default_factory=dict)
    confirmation_status: Literal["unconfirmed", "confirmed", "cancelled"] = "unconfirmed"


class ValidatedLead(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    occupation: str = Field(min_length=1)
    income: Decimal = Field(ge=Decimal("0"))
    income_currency: Literal["THB"] = "THB"
    income_period: Literal["monthly_thb"] = "monthly_thb"
    phone: str
    product_id: str = Field(min_length=1)
    confirmation_status: Literal["confirmed"]

    @field_validator("name", "occupation", "product_id", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.strip()

    @field_validator("income")
    @classmethod
    def validate_money_precision(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or value < 0:
            raise ValueError("income must be finite and non-negative")
        if value * 100 != (value * 100).to_integral_value():
            raise ValueError("income must have at most 2 decimal places")
        return value

    @field_validator("phone")
    @classmethod
    def validate_phone_value(cls, value: str) -> str:
        return normalize_phone(value)


INTEREST_TERMS = (
    "สนใจ", "อยากสมัคร", "ต้องการสมัคร", "ขอสมัคร", "สมัครเลย",
    "interested", "want to buy", "want to apply",
)
NEGATIVE_INTEREST_TERMS = (
    "ไม่สนใจ", "ไม่ต้องการ", "ไม่อยาก", "ไม่สมัคร", "ไม่เอา", "ยกเลิก", "cancel",
)
INTEREST_QUESTION_TERMS = (
    "วิธีสมัคร", "วิธีการสมัคร", "สมัครอย่างไร", "สมัครยังไง", "ขั้นตอนสมัคร",
    "how to apply", "how can i apply", "สนใจไหม", "สนใจหรือไม่", "are you interested",
)
LEAD_REQUIRED_FIELDS = ("name", "occupation", "income", "income_currency", "income_period", "phone", "product_id")


def detect_product_interest(text: str) -> bool:
    """Return true only for explicit positive product interest."""

    lowered = text.lower()
    if any(term in lowered for term in NEGATIVE_INTEREST_TERMS):
        return False
    if any(term in lowered for term in INTEREST_QUESTION_TERMS):
        return False
    return any(term in lowered for term in INTEREST_TERMS)


def lead_route(text: str) -> Literal["lead", "general"]:
    """Route explicit product interest into lead collection."""

    return "lead" if detect_product_interest(text) else "general"


def normalize_phone(value: str) -> str:
    compact = re.sub(r"[\s().-]", "", value)
    if compact.startswith("+66"):
        compact = "0" + compact[3:]
    elif compact.startswith("0066"):
        compact = "0" + compact[4:]
    if not re.fullmatch(r"0\d{8,9}", compact):
        raise ValueError("phone must be a Thai 9-10 digit national number after normalization")
    return compact


def _first_match(patterns: list[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            # Do not strip Thai letters: for example, "เอก" must remain "เอก".
            return match.group(1).strip(" \t,;:|\n")
    return None


NUMBER = r"[0-9][0-9,]*(?:\.[0-9]+)?"
CURRENCY = r"(?:บาท|thb|฿|usd|ดอลลาร์|\$)"
PERIOD = r"(?:ต่อปี|ต่อเดือน|รายปี|รายเดือน|per\s*year|per\s*month|/year|/month)"


def _currency(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"บาท", "thb", "฿"}:
        return "THB"
    if lowered in {"usd", "ดอลลาร์", "$"}:
        return "USD"
    return None


def _period(value: str | None, currency: str | None) -> str | None:
    if not value or not currency:
        return None
    lowered = value.lower()
    if "ปี" in lowered or "year" in lowered:
        return f"annual_{currency.lower()}"
    if "เดือน" in lowered or "month" in lowered:
        return f"monthly_{currency.lower()}"
    return None


def _decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def _extract_income(text: str) -> dict[str, Any]:
    label = r"(?:รายได้|เงินเดือน|income)\s*[:=]?\s*"
    range_match = re.search(
        rf"{label}(?P<before>{CURRENCY})?\s*(?P<low>{NUMBER})\s*(?:-|–|—|ถึง|to)\s*(?P<high>{NUMBER})\s*(?P<after>{CURRENCY})?\s*(?P<period>{PERIOD})?",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        low = _decimal(range_match.group("low"))
        high = _decimal(range_match.group("high"))
        currency = _currency(range_match.group("before") or range_match.group("after"))
        return {
            "income": None, "income_min": low, "income_max": high,
            "income_currency": currency,
            "income_period": _period(range_match.group("period"), currency),
            "raw": range_match.group(0),
        }

    single_match = re.search(
        rf"{label}(?P<before>{CURRENCY})?\s*(?P<value>{NUMBER})\s*(?P<after>{CURRENCY})?\s*(?P<period>{PERIOD})?",
        text,
        flags=re.IGNORECASE,
    )
    if not single_match:
        # A unit-only correction is still income evidence. Clear the previous
        # amount and ask for a complete value instead of silently retaining it.
        context = re.search(
            rf"(?P<currency>{CURRENCY})|(?P<period>{PERIOD})",
            text,
            flags=re.IGNORECASE,
        )
        if context:
            currency = _currency(context.group("currency"))
            raw = context.group(0)
            period = _period(raw, currency)
            if period is None:
                lowered = raw.lower()
                if "ปี" in lowered or "year" in lowered:
                    period = "annual_pending"
                elif "เดือน" in lowered or "month" in lowered:
                    period = "monthly_pending"
            return {
                "income": None, "income_min": None, "income_max": None,
                "income_currency": currency, "income_period": period, "raw": raw,
            }
        return {"income": None, "income_min": None, "income_max": None, "income_currency": None, "income_period": None, "raw": None}
    currency = _currency(single_match.group("before") or single_match.group("after"))
    return {
        "income": _decimal(single_match.group("value")), "income_min": None, "income_max": None,
        "income_currency": currency, "income_period": _period(single_match.group("period"), currency),
        "raw": single_match.group(0),
    }


def extract_lead_draft(text: str, *, product_id: str | None = None) -> LeadDraft:
    """Extract obvious user-provided fields without inventing missing values."""

    name = _first_match(
        [
            r"ชื่อ\s*[:=]?\s*(.+?)(?=\s+(?:ทำงาน|อาชีพ|รายได้|เงินเดือน|สนใจ|เบอร์)|[,;\n]|$)",
            r"name\s*[:=]?\s*(.+?)(?=\s+(?:occupation|income|phone)|[,;\n]|$)",
        ], text,
    )
    occupation = _first_match(
        [
            r"อาชีพ\s*[:=]?\s*(.+?)(?=\s+(?:รายได้|เงินเดือน|สนใจ|เบอร์)|[,;\n]|$)",
            r"ทำงาน(?:เป็น|ด้าน)?\s*(.+?)(?=\s+(?:รายได้|เงินเดือน|สนใจ|เบอร์)|[,;\n]|$)",
            r"occupation\s*[:=]?\s*(.+?)(?=\s+(?:รายได้|income|สนใจ|เบอร์|phone)|[,;\n]|$)",
        ], text,
    )
    income_data = _extract_income(text)
    phone_match = re.search(r"(?:\+66|0066|0)[0-9\s().-]{8,14}", text)
    phone = phone_match.group(0).strip() if phone_match else None
    evidence: dict[str, str] = {}
    for key, value in {"name": name, "occupation": occupation, "income": income_data["raw"], "phone": phone}.items():
        if value:
            evidence[key] = value
    return LeadDraft(
        name=name, occupation=occupation, income=income_data["income"],
        income_min=income_data["income_min"], income_max=income_data["income_max"],
        income_currency=income_data["income_currency"], income_period=income_data["income_period"],
        phone=phone, product_id=product_id, raw_evidence=evidence,
    )


def merge_lead_drafts(existing: LeadDraft, incoming: LeadDraft) -> LeadDraft:
    """Merge newly supplied fields without overwriting them with missing values."""

    values = existing.model_dump()
    incoming_values = incoming.model_dump()
    income_updated = bool(incoming.raw_evidence.get("income")) or any(
        incoming_values[field] is not None
        for field in ("income", "income_min", "income_max", "income_currency", "income_period")
    )
    fields = ("name", "occupation", "phone", "product_id")
    for field in fields:
        if incoming_values[field] is not None:
            values[field] = incoming_values[field]
    if income_updated:
        for field in ("income", "income_min", "income_max", "income_currency", "income_period"):
            values[field] = incoming_values[field]
    values["raw_evidence"] = {**existing.raw_evidence, **incoming.raw_evidence}
    if incoming.confirmation_status in {"confirmed", "cancelled"}:
        values["confirmation_status"] = incoming.confirmation_status
    return LeadDraft.model_validate(values)


def missing_lead_fields(draft: LeadDraft) -> list[str]:
    """List fields still requiring user follow-up before confirmation."""

    missing: list[str] = []
    if not draft.name:
        missing.append("name")
    if not draft.occupation:
        missing.append("occupation")
    if draft.income is None:
        missing.append("income_confirmation" if draft.income_min is not None else "income")
    if draft.income_currency != "THB":
        missing.append("income_currency")
    if draft.income_period != "monthly_thb":
        missing.append("income_period")
    if not draft.phone:
        missing.append("phone")
    if not draft.product_id:
        missing.append("product_id")
    return missing


def confirm_lead(draft: LeadDraft) -> LeadDraft:
    """Mark a draft confirmed only after the caller has shown its fields to the user."""

    if missing_lead_fields(draft):
        raise ValueError("cannot confirm lead with missing or unverified fields")
    return draft.model_copy(update={"confirmation_status": "confirmed"})


def draft_fingerprint(draft: LeadDraft) -> str:
    """Create a stable confirmation binding for the fields shown to the user."""

    payload = draft.model_dump(mode="json", exclude={"confirmation_status"})
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def is_explicit_confirmation(text: str) -> bool:
    """Accept only a complete confirmation phrase in a confirmation turn."""

    normalized = re.sub(r"[\s,.;:!?؟]+", "", text.strip().lower())
    if not normalized or normalized.startswith(("ไม่", "ยังไม่", "ขอ", "เดี๋ยว", "ยัง")):
        return False
    return normalized in {
        "ยืนยัน", "ยืนยันข้อมูล", "ยืนยันครับ", "ยืนยันค่ะ",
        "ตกลง", "ตกลงครับ", "ตกลงค่ะ", "ใช่", "ใช่ครับ", "ใช่ค่ะ",
        "confirm", "yes", "yesconfirm", "ok", "okay",
    }


def is_explicit_cancellation(text: str) -> bool:
    """Recognize cancellation without treating an ordinary correction as one."""

    normalized = re.sub(r"[\s,.;:!?؟]+", "", text.strip().lower())
    return normalized in {
        "ยกเลิก", "ยกเลิกครับ", "ยกเลิกค่ะ", "ไม่สนใจ", "ไม่สนใจครับ", "ไม่สนใจค่ะ",
        "ไม่ต้องการ", "ไม่ต้องการครับ", "ไม่ต้องการค่ะ", "ไม่สมัคร", "cancel", "cancelled",
    }


def lead_summary(draft: LeadDraft) -> str:
    """Render only fields already supplied; never expose empty-slot noise."""

    parts: list[str] = []
    if draft.name:
        parts.append(f"ชื่อ: {draft.name}")
    if draft.occupation:
        parts.append(f"อาชีพ: {draft.occupation}")
    if draft.income_min is not None or draft.income_max is not None:
        income = f"{draft.income_min or '?'}–{draft.income_max or '?'}"
    elif draft.income is not None:
        income = str(draft.income)
    else:
        income = ""
    if income:
        if draft.income_currency and draft.income_period:
            unit = f"{draft.income_currency} / {draft.income_period}"
        else:
            unit = draft.income_currency or draft.income_period or ""
        parts.append(f"รายได้: {income}{(' ' + unit) if unit else ''}")
    if draft.phone:
        parts.append(f"เบอร์: {draft.phone}")
    if draft.product_id:
        parts.append(f"ผลิตภัณฑ์: {draft.product_id}")
    return " | ".join(parts)


def lead_field_prompt(fields: list[str]) -> str:
    labels = {
        "name": "ชื่อ", "occupation": "อาชีพ", "income": "รายได้พร้อมสกุลเงินและช่วงเวลา",
        "income_confirmation": "ยืนยันรายได้แบบตัวเลขเดียว", "income_currency": "สกุลเงิน เช่น บาท/THB",
        "income_period": "ช่วงเวลา เช่น ต่อเดือน", "phone": "เบอร์โทรศัพท์", "product_id": "ผลิตภัณฑ์",
    }
    if 'income' in fields:
        fields = [field for field in fields if field not in {'income_currency','income_period'}]
    return ", ".join(labels.get(field, field) for field in fields)


def validate_lead(draft: LeadDraft) -> ValidatedLead:
    errors: list[str] = []
    if draft.confirmation_status != "confirmed":
        errors.append("lead confirmation is required before save")
    if not draft.name or not draft.name.strip():
        errors.append("name is required")
    if not draft.occupation or not draft.occupation.strip():
        errors.append("occupation is required")
    if draft.income is None:
        errors.append("income range requires confirmation" if draft.income_min is not None else "income is required")
    elif not draft.income.is_finite() or draft.income < 0:
        errors.append("income must be finite and non-negative")
    elif draft.income * 100 != (draft.income * 100).to_integral_value():
        errors.append("income must have at most 2 decimal places")
    if draft.income_currency != "THB":
        errors.append("income currency must be confirmed as THB")
    if draft.income_period != "monthly_thb":
        errors.append("income must be confirmed as monthly_thb")
    if not draft.phone:
        errors.append("phone is required")
    if not draft.product_id or not draft.product_id.strip():
        errors.append("product_id is required")
    normalized_phone = None
    if draft.phone:
        try:
            normalized_phone = normalize_phone(draft.phone)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError("; ".join(errors))
    assert draft.income is not None and normalized_phone is not None and draft.name and draft.occupation and draft.product_id
    return ValidatedLead(
        name=draft.name.strip(), occupation=draft.occupation.strip(), income=draft.income,
        income_currency="THB", income_period="monthly_thb", phone=normalized_phone,
        product_id=draft.product_id.strip(), confirmation_status="confirmed",
    )


def income_to_thb_storage(value: Decimal) -> str:
    """Return an exact decimal THB value suitable for SQLite/PostgreSQL."""

    if value * 100 != (value * 100).to_integral_value():
        raise ValueError("income has more than 2 decimal places; refusing silent truncation")
    return format(value, "f")


def save_lead_draft(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    access_token: str,
    draft: LeadDraft,
) -> None:
    """Persist every extracted lead slot in the account's shared profile."""

    if not access_token:
        raise PermissionError("access_token is required to save a lead draft")
    from insurex.session import _authorize

    session = _authorize(connection, session_id, access_token)
    owner_id = session["owner_id"]
    normalized_phone = None
    if draft.phone:
        try:
            normalized_phone = normalize_phone(draft.phone)
        except ValueError:
            # Keep the raw value so the user can correct it in a later turn.
            normalized_phone = None
    now = datetime.now(UTC).isoformat()
    payload = json.dumps(draft.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    connection.execute(
        """
        INSERT INTO lead_drafts(
            owner_id, last_session_id, product_id, name, occupation, income_value, income_min,
            income_max, income_currency, income_period, phone_raw,
            phone_normalized, status, draft_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(owner_id) DO UPDATE SET
            last_session_id=excluded.last_session_id,
            product_id=excluded.product_id,
            name=excluded.name,
            occupation=excluded.occupation,
            income_value=excluded.income_value,
            income_min=excluded.income_min,
            income_max=excluded.income_max,
            income_currency=excluded.income_currency,
            income_period=excluded.income_period,
            phone_raw=excluded.phone_raw,
            phone_normalized=excluded.phone_normalized,
            status=excluded.status,
            draft_json=excluded.draft_json,
            updated_at=excluded.updated_at
        """,
        (
            owner_id,
            session_id,
            draft.product_id,
            draft.name,
            draft.occupation,
            str(draft.income) if draft.income is not None else None,
            str(draft.income_min) if draft.income_min is not None else None,
            str(draft.income_max) if draft.income_max is not None else None,
            draft.income_currency,
            draft.income_period,
            draft.phone,
            normalized_phone,
            draft.confirmation_status,
            payload,
            now,
        ),
    )
    connection.commit()


def load_account_lead_draft(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    access_token: str,
) -> LeadDraft | None:
    """Load the one customer profile draft shared by all of an account's chats."""

    if not access_token:
        raise PermissionError("access_token is required to load a lead draft")
    from insurex.session import _authorize

    session = _authorize(connection, session_id, access_token)
    row = connection.execute(
        "SELECT draft_json FROM lead_drafts WHERE owner_id=?",
        (session["owner_id"],),
    ).fetchone()
    if row is not None:
        return LeadDraft.model_validate_json(row["draft_json"])
    # Older completed leads may predate immediate draft persistence. Rebuild
    # the shared profile from the canonical account lead so a new chat does
    # not ask the customer for known fields again.
    lead = connection.execute(
        """
        SELECT name, occupation, income_thb, income_period,
               phone_normalized, product_id
        FROM leads WHERE owner_id=?
        """,
        (session["owner_id"],),
    ).fetchone()
    if lead is None:
        return None
    return LeadDraft(
        name=lead["name"],
        occupation=lead["occupation"],
        income=Decimal(str(lead["income_thb"])),
        income_currency="THB",
        income_period=lead["income_period"],
        phone=lead["phone_normalized"],
        product_id=lead["product_id"],
        confirmation_status="confirmed",
    )


def lead_profile_status(connection: sqlite3.Connection, owner_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Return freshness for the account profile using a six-month (183 day) window."""

    row = connection.execute(
        "SELECT lead_id, created_at, updated_at, profile_verified_at, refresh_requested_at, product_id FROM leads WHERE owner_id=?",
        (owner_id,),
    ).fetchone()
    if row is None:
        return {"exists": False, "stale": False, "reference_at": None, "refresh_requested_at": None, "product_id": None}
    reference = row["profile_verified_at"] or row["updated_at"] or row["created_at"]
    try:
        reference_dt = datetime.fromisoformat(reference)
        if reference_dt.tzinfo is None:
            reference_dt = reference_dt.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        reference_dt = datetime.min.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    return {
        "exists": True,
        "stale": current - reference_dt >= timedelta(days=183),
        "reference_at": reference,
        "refresh_requested_at": row["refresh_requested_at"],
        "product_id": row["product_id"],
    }


def mark_lead_refresh_requested(connection: sqlite3.Connection, owner_id: str) -> None:
    connection.execute(
        "UPDATE leads SET refresh_requested_at=COALESCE(refresh_requested_at, ?) WHERE owner_id=?",
        (datetime.now(UTC).isoformat(), owner_id),
    )
    connection.commit()


def _lead_payload_hash(lead: ValidatedLead) -> str:
    payload = {
        "name": lead.name, "occupation": lead.occupation, "income": str(lead.income),
        "income_currency": lead.income_currency, "income_period": lead.income_period,
        "phone": lead.phone, "product_id": lead.product_id, "confirmation_status": lead.confirmation_status,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def save_lead(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    access_token: str,
    request_id: str,
    lead: ValidatedLead,
    product_catalog: set[str] | None = None,
) -> int:
    """Validate ownership and payload again, then insert idempotently."""

    if not request_id.strip():
        raise ValueError("request_id is required")
    validated = ValidatedLead.model_validate(lead.model_dump())
    if product_catalog is not None and validated.product_id not in product_catalog:
        raise ValueError("product_id is not present in the active product catalog")
    if not access_token:
        raise PermissionError("access_token is required to save a lead")
    from insurex.session import _authorize

    session = _authorize(connection, session_id, access_token)
    owner_id = session["owner_id"]
    digest = _lead_payload_hash(validated)
    existing = connection.execute(
        "SELECT lead_id, owner_id, payload_hash FROM lead_request_log WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if existing:
        if existing["owner_id"] != owner_id or existing["payload_hash"] != digest:
            raise ValueError("request_id was already used with a different lead payload")
        return int(existing["lead_id"])
    now = datetime.now(UTC).isoformat()
    cursor = connection.execute(
        """
        INSERT INTO leads(
            owner_id, session_id, product_id, name, occupation, income_thb,
            income_period, phone_normalized, request_id, payload_hash, created_at, updated_at,
            profile_verified_at, refresh_requested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(owner_id) DO UPDATE SET
            session_id=excluded.session_id,
            product_id=excluded.product_id,
            name=excluded.name,
            occupation=excluded.occupation,
            income_thb=excluded.income_thb,
            income_period=excluded.income_period,
            phone_normalized=excluded.phone_normalized,
            request_id=excluded.request_id,
            payload_hash=excluded.payload_hash,
            updated_at=excluded.updated_at,
            profile_verified_at=excluded.profile_verified_at,
            refresh_requested_at=NULL
        RETURNING lead_id
        """,
        (
            owner_id, session_id, validated.product_id, validated.name, validated.occupation,
            income_to_thb_storage(validated.income), validated.income_period, validated.phone,
            request_id, digest, now, now, now,
        ),
    )
    lead_id = row_value(cursor.fetchone(), "lead_id")
    connection.execute(
        """
        INSERT INTO lead_request_log(request_id, owner_id, lead_id, payload_hash, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (request_id, owner_id, lead_id, digest, now),
    )
    connection.commit()
    return int(lead_id)


def list_leads_for_session(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    access_token: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Read the one lead profile owned by the authorized account."""

    if limit <= 0:
        return []
    from insurex.session import _authorize

    session = _authorize(connection, session_id, access_token)
    rows = connection.execute(
        """
        SELECT lead_id, product_id, name, occupation, income_thb,
               phone_normalized, created_at
        FROM leads WHERE owner_id=? ORDER BY lead_id DESC LIMIT ?
        """,
        (session["owner_id"], limit),
    ).fetchall()
    return [dict(row) for row in rows]
