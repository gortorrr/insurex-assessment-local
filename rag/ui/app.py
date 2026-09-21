"""Streamlit local RAG assistant for the InsureX product corpus.

Run after installing the approved UI dependencies with:
    python -m streamlit run rag/ui/app.py
"""

from __future__ import annotations

import json
import os
import uuid
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from threading import Event, Lock
from typing import Any

import streamlit as st
import extra_streamlit_components as stx

from insurex.config import Settings
from insurex.auth import (
    LOGIN_TTL_SECONDS,
    authenticate,
    create_login_session,
    register,
    restore_login_session,
    revoke_login_session,
)
from insurex.telemetry import TurnTrace
from insurex.leads import extract_lead_draft, lead_route, lead_summary, load_account_lead_draft, lead_profile_status
from insurex.rag.lead_extractor import LlmLeadExtractor
from insurex.suggestions import DEFAULT_QUESTIONS, context_key, generate_questions, product_history, select_grounded_questions
from insurex.db import connect_assistant_runtime
from insurex.session import (
    SessionHandle,
    append_message,
    append_staff_message,
    accept_handoff,
    create_session,
    get_or_create_customer_session,
    delete_owned_session,
    list_owned_sessions,
    load_history,
    load_history_for_staff,
    list_handoff_queue,
    get_open_handoff,
    request_handoff,
    resolve_handoff,
    resume_owned_session,
    resume_session,
    annotate_turn,
)
from insurex.rag.checkpoint import persistent_checkpointer
from insurex.rag.graph import build_rag_graph
from insurex.rag.runtime import (
    LeadTurnHandler,
    RequestBudget,
    build_gemini_services,
    build_offline_services,
)


ROOT = Path(__file__).resolve().parents[1]
AUTH_COOKIE_NAME = "insurex_login"


def connect_from_settings(settings: Settings):
    """Open the isolated Test Case #2 application database."""

    return connect_assistant_runtime(settings)


def should_handoff(state: dict[str, Any], citations: list[dict[str, Any]]) -> bool:
    """Hand off explicit requests or grounded product questions that failed safely."""

    return bool(
        state.get("handoff_requested")
        or (
            state.get("intent") == "question"
            and not citations
            and state.get("response_type") != "clarification"
            and state.get("evidence_status") in {"insufficient", "error"}
        )
    )


def delete_cookie_safely(
    cookie_manager: stx.CookieManager,
    cookie_name: str,
    *,
    key: str,
) -> bool:
    """Delete a browser cookie even when the component cache already removed it.

    Streamlit request cookies can remain visible for one rerun after logout,
    while extra-streamlit-components has already removed the same cookie from
    its in-memory dictionary.  CookieManager.delete sends the browser command
    before deleting that dictionary entry, so the resulting KeyError is safe
    to suppress.
    """

    try:
        cookie_manager.delete(cookie_name, key=key)
    except KeyError:
        return False
    return True


@st.cache_resource(show_spinner=False)
def _ui_job_runtime() -> tuple[ThreadPoolExecutor, Lock, dict[str, dict[str, Any]]]:
    """Keep background turn jobs alive across Streamlit script reruns."""

    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="insurex-ui"), Lock(), {}


_UI_JOB_EXECUTOR, _UI_JOB_LOCK, _UI_JOBS = _ui_job_runtime()


@st.cache_resource(show_spinner=False)
def _suggestion_job_runtime() -> tuple[Lock, dict[str, Any]]:
    return Lock(), {}


_SUGGESTION_JOB_LOCK, _SUGGESTION_JOBS = _suggestion_job_runtime()


@st.cache_data(show_spinner=False)
def local_font_css() -> str:
    """Embed the official InsureX SukhumvitSet assets for an offline-stable UI."""

    font_dir = Path(__file__).resolve().parent / "assets" / "fonts"
    faces = (
        ("SukhumvitSet", 300, "SukhumvitSet-Light.0kj_6km3byoum.ttf"),
        ("SukhumvitSet", 500, "SukhumvitSet-Medium.1-75sq-32636u.ttf"),
        ("SukhumvitSet", 700, "SukhumvitSet-Bold.3mo7ul_8tw0ke.ttf"),
    )
    css = []
    for family, weight, filename in faces:
        encoded = b64encode((font_dir / filename).read_bytes()).decode("ascii")
        css.append(
            "@font-face {"
            f"font-family: '{family}'; font-style: normal; font-weight: {weight}; "
            f"font-display: swap; src: url(data:font/ttf;base64,{encoded}) format('truetype');"
            "}"
        )
    return "\n".join(css)


def ensure_ui_session(settings: Settings) -> tuple[str, str]:
    """Create one owner-scoped session per Streamlit browser session."""

    owner_id = st.session_state["principal"]["owner_id"]
    st.session_state["owner_id"] = owner_id
    if "lead_session_id" not in st.session_state:
        connection = connect_from_settings(settings)
        try:
            handle = get_or_create_customer_session(connection, owner_id)
            st.session_state["lead_session_id"] = handle.session_id
            st.session_state["lead_access_token"] = handle.access_token
            st.session_state["assistant_thread_id"] = handle.thread_id
            st.session_state["assistant_messages"] = []
            st.session_state["history_loaded_for"] = None
            st.session_state.setdefault("session_tokens", {})[handle.session_id] = handle.access_token
        finally:
            connection.close()
    return st.session_state["lead_session_id"], st.session_state["lead_access_token"]


def activate_session(handle: SessionHandle) -> None:
    """Switch the UI to a session already authorized by this browser context."""

    st.session_state["lead_session_id"] = handle.session_id
    st.session_state["lead_access_token"] = handle.access_token
    st.session_state["assistant_thread_id"] = handle.thread_id
    st.session_state["assistant_messages"] = []
    st.session_state["lead_panel"] = ""
    st.session_state["citation_by_content"] = {}
    st.session_state["history_loaded_for"] = None
    st.session_state.setdefault("session_tokens", {})[handle.session_id] = handle.access_token


def load_ui_history(settings: Settings) -> None:
    """Hydrate the visible transcript from the owner-checked persistent store."""

    session_id = st.session_state.get("lead_session_id")
    access_token = st.session_state.get("lead_access_token")
    if not session_id or not access_token:
        return
    connection = connect_from_settings(settings)
    try:
        history = load_history(
            connection,
            session_id=session_id,
            access_token=access_token,
            limit=40,
            include_metadata=True,
        )
        account_draft = load_account_lead_draft(
            connection,
            session_id=session_id,
            access_token=access_token,
        )
        handoff = get_open_handoff(connection, session_id)
    finally:
        connection.close()
    citation_map = st.session_state.get("citation_by_content", {})
    st.session_state["assistant_messages"] = [
        {
            "role": item["role"],
            "content": item["content"],
            "sender_type": item.get("sender_type"),
            "sender_label": item.get("sender_label"),
            "citations": item.get("metadata", {}).get("citations", citation_map.get(item["content"], [])),
        }
        for item in history
        if item["role"] in {"user", "assistant"}
    ]
    st.session_state["history_loaded_for"] = session_id
    st.session_state["handoff_status"] = handoff["status"] if handoff else None
    st.session_state["handoff_staff_name"] = handoff.get("assigned_staff_name") if handoff else None
    if account_draft is not None:
        st.session_state["lead_panel"] = lead_summary(account_draft)
    else:
        with persistent_checkpointer(settings) as saver:
            state = saver.get({"configurable": {"thread_id": st.session_state["assistant_thread_id"]}})
            st.session_state["lead_panel"] = (state or {}).get("channel_values", {}).get("lead_summary", "")


def compact_datetime(value: str) -> str:
    """Return a short local timestamp for the chat history list."""

    try:
        return datetime.fromisoformat(value).astimezone().strftime("%d %b %H:%M")
    except (TypeError, ValueError):
        return ""


@st.dialog("ลบประวัติแชต")
def render_delete_dialog(
    settings: Settings,
    session_id: str,
    owner_id: str,
    current_session: str | None,
) -> None:
    """Confirm deletion without placing destructive controls in the sidebar."""

    st.warning("ต้องการลบประวัติการสนทนานี้หรือไม่?")
    confirm_col, cancel_col = st.columns(2, gap="small")
    with confirm_col:
        if st.button("ยืนยันลบประวัติ", key="confirm_delete_dialog", type="primary", use_container_width=True):
            connection = connect_from_settings(settings)
            try:
                delete_owned_session(connection, session_id, owner_id)
                handle = create_session(connection, owner_id) if session_id == current_session else None
            finally:
                connection.close()
            st.session_state["session_tokens"] = {
                key: value
                for key, value in st.session_state.get("session_tokens", {}).items()
                if key != session_id
            }
            if handle is not None:
                activate_session(handle)
            st.rerun()
    with cancel_col:
        if st.button("ยกเลิก", key="cancel_delete_dialog", use_container_width=True):
            st.rerun()


def create_new_conversation(settings: Settings) -> None:
    connection = connect_from_settings(settings)
    try:
        handle = create_session(connection, st.session_state["owner_id"])
    finally:
        connection.close()
    activate_session(handle)


def render_chat_history(settings: Settings) -> None:
    """Render a ChatGPT-like owner-scoped conversation rail."""

    owner_id = st.session_state["owner_id"]
    connection = connect_from_settings(settings)
    try:
        sessions = list_owned_sessions(connection, owner_id, limit=30)
    finally:
        connection.close()
    current_session = st.session_state.get("lead_session_id")
    st.markdown("<div class='history-heading'>ประวัติการสนทนา</div>", unsafe_allow_html=True)
    if not sessions:
        st.caption("ยังไม่มีบทสนทนาอื่น")
        return
    for item in sessions:
        session_id = str(item["session_id"])
        title = str(item["title"]).replace("\n", " ").strip() or "การสนทนาใหม่"
        title = title[:42] + ("…" if len(title) > 42 else "")
        is_current = session_id == current_session
        label = ("● กำลังดูอยู่ · " if is_current else "○ ") + title
        history_col, delete_col = st.columns([0.78, 0.22], gap="small")
        with history_col:
            if is_current:
                st.markdown(
                    "<div class='history-current-row'>"
                    f"<span>{escape(label)}</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            elif st.button(
                label,
                key=f"history_session_{session_id}",
                use_container_width=True,
                help=f"{compact_datetime(str(item['conversation_last_message_at']))} · {item['message_count']} ข้อความ",
            ):
                connection = connect_from_settings(settings)
                try:
                    handle = resume_owned_session(connection, session_id, owner_id)
                finally:
                    connection.close()
                activate_session(handle)
                st.rerun()
        with delete_col:
            if st.button("🗑️", key=f"delete_session_{session_id}", help="ลบประวัติแชตนี้"):
                render_delete_dialog(settings, session_id, owner_id, current_session)


def stable_turn_request_id(question: str) -> str:
    """Keep one logical request ID across a UI retry of the same turn."""

    pending = st.session_state.get("pending_turn")
    if pending and pending.get("question") == question:
        return str(pending["request_id"])
    request_id = f"ui-turn-{uuid.uuid4().hex}"
    st.session_state["pending_turn"] = {"question": question, "request_id": request_id}
    return request_id


def _run_turn_job(
    settings: Settings,
    *,
    session_id: str,
    access_token: str,
    owner_id: str,
    question: str,
    request_id: str,
    history: list[dict[str, str]],
    product_ids: set[str],
    mode: str,
    thread_id: str,
    cancel_event: Event,
    runtime_events: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run one graph turn outside the Streamlit script thread."""

    if runtime_events is None:
        runtime_events = []

    def emit(stage: str, title: str, detail: str, status: str = "running") -> None:
        # Never include the query, slot values, model output, or exception text:
        # those values can contain personal information.
        with _UI_JOB_LOCK:
            runtime_events.append({
                "time": datetime.now().astimezone().strftime("%H:%M:%S"),
                "stage": stage,
                "title": title,
                "detail": detail,
                "status": status,
            })

    if cancel_event.is_set():
        return {"cancelled": True}
    emit("request", "รับคำขอแล้ว", "เริ่มงานใน session ของบัญชีที่ยืนยันตัวตนแล้ว")
    connection = connect_from_settings(settings)
    trace = TurnTrace(settings.trace_path, owner_id, session_id)
    budget = RequestBudget(max_requests=10)
    trace_status = "error"
    try:
        handle = resume_session(connection, session_id, access_token, expected_owner_id=owner_id)
        if handle.thread_id != thread_id:
            raise PermissionError("thread does not belong to the authorized session")
        emit("session", "ตรวจสอบ Session", "owner, session และ LangGraph thread ตรงกัน", "complete")
        lead_handler = LeadTurnHandler(
            connection,
            session_id,
            access_token,
            product_ids,
            request_id=request_id,
            product_aliases={
                item["product_id"]: [item["product_name"], *item.get("product_aliases", [])]
                for item in json.loads(settings.kb_manifest_path.read_text(encoding="utf-8"))["documents"]
            },
        )
        profile_status = lead_profile_status(connection, owner_id)
        parsed_lead_fields = extract_lead_draft(question)
        lead_fast_path = lead_route(question) == "lead" or bool(parsed_lead_fields.raw_evidence)
        if mode == "Gemini live" and not lead_fast_path:
            gemini = build_gemini_services(settings, budget=budget)
            lead_handler.extractor = LlmLeadExtractor(gemini.model, budget)
            services = gemini.services(lead_turn=lead_handler)
        else:
            services = build_offline_services(settings, lead_turn=lead_handler)
        with persistent_checkpointer(settings) as saver:
            graph = build_rag_graph(services, checkpointer=saver)
            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"session_id": session_id, "query": question, "active_product_id": None,
                      "conversation_history": history,
                      "lead_refresh_required": bool(profile_status["stale"])}
            final_update: dict[str, Any] = {}
            for update in graph.stream(inputs, config, stream_mode="updates"):
                for node, values in update.items():
                    trace.node(node, values)
                    if isinstance(values, dict):
                        final_update.update(values)
                    if node == "retrieve":
                        chunks = values.get("retrieved_chunks", []) if isinstance(values, dict) else []
                        sources = sorted({f"{item.doc_id} p.{item.page_number}" for item in chunks})
                        detail = f"พบ {len(chunks)} chunks"
                        if sources:
                            detail += " · " + ", ".join(sources[:5])
                        emit(node, "ค้นจาก ChromaDB", detail, "complete")
                    elif node == "assess_evidence":
                        assessment = values.get("evidence_status", "unknown") if isinstance(values, dict) else "unknown"
                        emit(node, "ตรวจความเพียงพอของหลักฐาน", str(assessment), "complete")
                    elif node == "collect_lead":
                        missing = values.get("lead_missing", []) if isinstance(values, dict) else []
                        saved_id = values.get("lead_saved_id") if isinstance(values, dict) else None
                        if saved_id is not None:
                            emit(node, "Structured Lead", f"ตรวจ schema ครบและบันทึก SQLite แล้ว · lead_id={saved_id}", "complete")
                        else:
                            fields = ", ".join(missing) if missing else "บันทึก draft แล้ว"
                            emit(node, "Structured Lead", f"บันทึก draft ลง SQLite · ยังขาด {fields}", "complete")
                    else:
                        labels = {
                            "prepare_turn": ("เตรียม State", "โหลดบริบทที่จำเป็นจากบทสนทนานี้"),
                            "route_intent": ("จำแนก Intent", "เลือก RAG, Lead, บทสนทนาทั่วไป หรือขอพนักงาน"),
                            "direct_response": ("ตอบบทสนทนาทั่วไป", "ตอบโดยไม่เรียก ChromaDB หรือ Gemini"),
                            "resolve_query": ("ปรับคำถามด้วยบริบท", "สร้าง standalone query สำหรับ retrieval"),
                            "rewrite_query_once": ("ปรับ Retrieval Query", "ลองค้นใหม่แบบจำกัดหนึ่งรอบ"),
                            "generate_grounded_answer": ("สร้างคำตอบ", "ใช้เฉพาะหลักฐานที่ผ่านการตรวจ"),
                            "validate_answer_citations": ("ตรวจคำตอบและ Citation", "ตรวจแหล่งเอกสารและเลขหน้าก่อนส่ง"),
                            "fallback": ("Controlled fallback", "ไม่พบหลักฐานเพียงพอ จึงไม่สร้างข้อเท็จจริง"),
                        }
                        title, detail = labels.get(node, (node, "LangGraph node ทำงานเสร็จ"))
                        emit(node, title, detail, "complete")
            # Some SQLite checkpointer versions expose the preceding snapshot
            # until the stream context is fully closed. Node updates are the
            # authoritative result of this run, so overlay them on the durable
            # snapshot before returning the UI response.
            state = {**graph.get_state(config).values, **final_update}
        if cancel_event.is_set():
            trace_status = "cancelled"
            return {"cancelled": True}
        answer = state.get("answer") or "No answer was found in the product documents."
        citations = [
            {
                "doc_id": item.doc_id,
                "page": item.page_number,
                "source_url": item.source_url,
            }
            for item in state.get("citations", [])
        ]
        handoff_required = should_handoff(state, citations)
        if handoff_required and not state.get("handoff_requested"):
            answer = answer.rstrip() + "\n\nส่งคำถามนี้ให้พนักงานขายช่วยตรวจสอบต่อแล้วครับ"
        append_message(
            connection,
            session_id=session_id,
            access_token=access_token,
            role="assistant",
            content=answer,
            request_id=f"{request_id}:assistant",
            sender_type="ai",
            sender_label="AI ผู้ช่วยผลิตภัณฑ์",
        )
        handoff_case_id = None
        if handoff_required:
            reason = (
                "customer_requested_human"
                if state.get("handoff_requested")
                else state.get("error_code") or state.get("evidence_reason") or "no_answer"
            )
            handoff_case_id = request_handoff(
                connection,
                session_id=session_id,
                access_token=access_token,
                trigger_reason=str(reason)[:240],
                trigger_request_id=request_id,
            )
            emit("handoff", "ส่งต่อพนักงานขาย", "พัก AI/RAG สำหรับ session นี้และเพิ่มเข้าคิวพนักงาน", "complete")
        trace_status = "error" if state.get("error_code") else "complete"
        emit("checkpoint", "บันทึก Conversation State", "อัปเดต LangGraph checkpoint แยกตาม thread แล้ว", "complete")
        annotate_turn(connection, session_id, access_token, request_id,
                      # Keep a product question in the user's suggestion history
                      # even when retrieval returns a controlled no-answer.  A
                      # missing citation is a retrieval result, not a reason to
                      # classify the user's turn as private chat.
                      product_turn=state.get("intent") == "question", citations=citations)
        return {
            "lead_summary": state.get("lead_summary", ""),
            "lead_missing": state.get("lead_missing", []),
            "lead_saved_id": state.get("lead_saved_id"),
            "answer": answer,
            "citations": citations,
            "error_code": state.get("error_code"),
            "handoff_case_id": handoff_case_id,
        }
    except Exception as exc:
        emit("error", "เกิดข้อผิดพลาด", f"หยุดอย่างปลอดภัยด้วย {type(exc).__name__}", "error")
        return {"error_code": type(exc).__name__}
    finally:
        try:
            trace.finish(trace_status, budget)
        finally:
            connection.close()


def _start_turn_job(
    settings: Settings,
    *,
    question: str,
    request_id: str,
    product_ids: set[str],
    mode: str,
) -> bool:
    session_id, access_token = ensure_ui_session(settings)
    connection = connect_from_settings(settings)
    try:
        history = load_history(
            connection,
            session_id=session_id,
            access_token=access_token,
            limit=12,
            product_only=True,
        )
        append_message(
            connection,
            session_id=session_id,
            access_token=access_token,
            role="user",
            content=question,
            request_id=request_id,
            sender_type="customer",
            sender_account_id=st.session_state["principal"]["owner_id"],
            sender_label=st.session_state["principal"]["username"],
        )
        handoff = get_open_handoff(connection, session_id)
        if handoff:
            parsed = extract_lead_draft(question)
            lead_result: dict[str, Any] = {}
            if lead_route(question) == "lead" or bool(parsed.raw_evidence):
                manifest = json.loads(settings.kb_manifest_path.read_text(encoding="utf-8"))
                handler = LeadTurnHandler(
                    connection,
                    session_id,
                    access_token,
                    product_ids,
                    request_id=request_id,
                    product_aliases={
                        item["product_id"]: [item["product_name"], *item.get("product_aliases", [])]
                        for item in manifest["documents"]
                    },
                )
                # Structured lead tooling remains active during handoff, but
                # its answer is deliberately discarded: only staff may reply.
                lead_result = handler(question, {"active_product_id": None})
            st.session_state["lead_panel"] = lead_result.get("lead_summary", st.session_state.get("lead_panel", ""))
            st.session_state["handoff_status"] = handoff["status"]
            st.session_state["assistant_messages"].append({
                "role": "user", "content": question, "sender_type": "customer",
                "sender_label": st.session_state["principal"]["username"],
            })
            st.session_state.pop("pending_turn", None)
            return False
    finally:
        connection.close()
    thread_id = st.session_state.get("assistant_thread_id", session_id)
    cancel_event = Event()
    job_id = uuid.uuid4().hex
    runtime_events: list[dict[str, str]] = []
    future = _UI_JOB_EXECUTOR.submit(
        _run_turn_job,
        settings,
        session_id=session_id,
        access_token=access_token,
        owner_id=st.session_state["principal"]["owner_id"],
        question=question,
        request_id=request_id,
        history=history,
        product_ids=product_ids,
        mode=mode,
        thread_id=thread_id,
        cancel_event=cancel_event,
        runtime_events=runtime_events,
    )
    with _UI_JOB_LOCK:
        _UI_JOBS[job_id] = {"future": future, "cancel_event": cancel_event,
                            "owner_id": st.session_state["principal"]["owner_id"], "session_id": session_id,
                            "events": runtime_events}
    st.session_state["active_job_id"] = job_id
    st.session_state["assistant_messages"].append({
        "role": "user", "content": question, "sender_type": "customer",
        "sender_label": st.session_state["principal"]["username"],
    })
    return True


def render_boot_loading() -> None:
    """Show the first-paint loading dialog without scheduling reruns.

    The CSS animation dismisses this short-lived overlay in the browser.  A
    scheduled fragment would compete with button events (notably logout) on
    some Streamlit builds, so the initial indicator is deliberately static.
    """

    st.markdown(
        """
        <div class="loading-dialog loading-dialog-boot" role="status" aria-live="polite">
            <span class="loading-spinner" aria-hidden="true"></span>
            <div><strong>Loading...</strong><small>รอสักครู่</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_action_loading(message: str) -> None:
    """Render a centered, short-lived modal for auth actions."""

    st.markdown(
        f"""
        <div class="loading-dialog loading-dialog-action" role="status" aria-live="polite">
            <span class="loading-spinner" aria-hidden="true"></span>
            <div><strong>Loading...</strong><small>{escape(message)}</small></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="500ms")
def render_active_turn() -> bool:
    """Show a live status and a cooperative stop control for the active turn."""

    job_id = st.session_state.get("active_job_id")
    if not job_id:
        return False
    with _UI_JOB_LOCK:
        job = _UI_JOBS.get(job_id)
    if job is None:
        st.session_state.pop("active_job_id", None)
        return False
    if (job["owner_id"] != st.session_state["principal"]["owner_id"]
            or job["session_id"] != st.session_state.get("lead_session_id")):
        st.session_state.pop("active_job_id", None)
        return False
    future = job["future"]
    cancel_event = job["cancel_event"]
    if future.done():
        try:
            result = future.result()
        except Exception as exc:
            result = {"error_code": type(exc).__name__}
        with _UI_JOB_LOCK:
            st.session_state["last_runtime_events"] = list(job.get("events", []))
            _UI_JOBS.pop(job_id, None)
        st.session_state.pop("active_job_id", None)
        if result.get("cancelled"):
            st.session_state["assistant_messages"].append(
                {"role": "assistant", "content": "หยุดการประมวลผลแล้ว", "citations": []}
            )
        elif result.get("answer"):
            st.session_state["lead_panel"] = result.get("lead_summary", "")
            if result.get("handoff_case_id"):
                st.session_state["handoff_status"] = "pending"
            answer = result["answer"]
            citations = result.get("citations", [])
            st.session_state["assistant_messages"].append(
                {
                    "role": "assistant",
                    "sender_type": "ai",
                    "sender_label": "AI ผู้ช่วยผลิตภัณฑ์",
                    "content": answer,
                    "citations": citations,
                    "error_code": result.get("error_code"),
                }
            )
            st.session_state["citation_by_content"][answer] = citations
        else:
            st.session_state["assistant_messages"].append(
                {
                    "role": "assistant",
                    "content": "The assistant service is temporarily unavailable.",
                    "citations": [],
                    "error_code": result.get("error_code", "UNKNOWN_ERROR"),
                }
            )
        st.session_state.pop("pending_turn", None)
        st.rerun()
    stopping = cancel_event.is_set()
    with st.status("กำลังหยุด..." if stopping else "กำลังประมวลผล...", expanded=True):
        st.write("กำลังค้นเอกสาร ตรวจหลักฐาน และเตรียมคำตอบ")
        if stopping:
            st.caption("กำลังรอให้คำขอที่กำลังทำงานยุติตาม timeout")
        elif st.button("หยุด", key=f"stop_turn_{job_id}", type="secondary"):
            cancel_event.set()
            st.rerun()
    return True


@st.fragment(run_every="500ms")
def render_runtime_log() -> None:
    """Show the actual LangGraph node stream without exposing user content."""

    events = list(st.session_state.get("last_runtime_events", []))
    job_id = st.session_state.get("active_job_id")
    if job_id:
        with _UI_JOB_LOCK:
            job = _UI_JOBS.get(job_id)
            if job and job.get("owner_id") == st.session_state["principal"]["owner_id"]:
                events = list(job.get("events", []))
    with st.container(key="runtime_log_control"):
        with st.popover("ตัวอย่าง Log", icon=":material/terminal:"):
            st.markdown("#### Agent runtime")
            st.caption("node ที่รันจริงแบบ real time โดยไม่แสดงข้อความหรือข้อมูลส่วนบุคคล")
            if not events:
                st.info("ส่งคำถามหรือเริ่มเก็บ Lead เพื่อดู workflow")
            for event in reversed(events[-16:]):
                symbol = "✓" if event["status"] == "complete" else ("!" if event["status"] == "error" else "•")
                st.markdown(f"**{symbol} {event['title']}** · `{event['time']}`")
                st.caption(event["detail"])


def render_session_controls(settings: Settings) -> None:
    """Expose the account's one durable conversation."""

    st.session_state["owner_id"] = st.session_state["principal"]["owner_id"]
    ensure_ui_session(settings)
    with st.sidebar:
        st.caption("บทสนทนาหลักของบัญชีนี้")


def _generate_suggestions_job(settings, products, history, owner, session_id, key):
    budget = RequestBudget(max_requests=1)
    trace = TurnTrace(settings.trace_path, owner, session_id)
    status = 'error'
    candidates = []
    try:
        if settings.llm_provider == 'google_ai_studio':
            gemini = build_gemini_services(settings, budget=budget)
            names = [
                alias
                for item in products.values()
                for alias in item.get('product_aliases', [item['product_name']])
            ]
            candidates = generate_questions(
                gemini.model, budget, history, names, candidate_count=8
            )
            questions, _ = select_grounded_questions(
                candidates, gemini.services(), products
            )
            source = 'AI แนะนำจากประวัติของคุณ · ตรวจหลักฐานแล้ว' if history else 'AI แนะนำให้เริ่มต้น · ตรวจหลักฐานแล้ว'
        else:
            offline = build_offline_services(settings)
            questions, _ = select_grounded_questions([], offline, products)
            source = 'คำถามเริ่มต้น · ตรวจหลักฐานแล้ว'
        if len(questions) != 4:
            raise ValueError('fewer than four evidence-backed suggestions')
        status = 'complete'
        return {'key': key, 'questions': questions, 'source': source, 'pending': False}
    except Exception:
        return {
            'key': key,
            'questions': list(DEFAULT_QUESTIONS),
            'source': 'คำถามเริ่มต้นจากฐานความรู้',
            'pending': False,
        }
    finally:
        trace.node('suggest_questions', {
            'candidate_count': len(candidates),
            'grounded_count': len(locals().get('questions', [])),
        })
        trace.finish(status, budget)


def render_suggestions(settings, products):
    owner = st.session_state['principal']['owner_id']
    product_terms = [alias for item in products.values() for alias in [item['product_name'], *item.get('product_aliases', [])]]
    connection = connect_from_settings(settings)
    try:
        try:
            history = product_history(connection, owner, product_terms=product_terms)
        except TypeError as exc:
            # A long-lived Streamlit worker can briefly retain an older
            # imported suggestions module during hot reload.  Keep the UI
            # usable while that worker is replaced instead of surfacing a
            # signature error to the user.
            if "product_terms" not in str(exc):
                raise
            history = product_history(connection, owner)
    finally:
        connection.close()
    corpus_key = '|'.join(
        f"{product_id}:{item.get('sha256', '')}"
        for product_id, item in sorted(products.items())
    ) + ':' + settings.retrieval_mode
    # Generate from this user's current history once per corpus/runtime cache.
    # Rebuilding synchronously after every chat turn disabled the composer and
    # made users click Send twice. A fresh login/worker still personalizes from
    # the latest persisted account history.
    key = context_key(owner, [], corpus_key)
    cached = st.session_state.get('suggestions', {})
    skip_boot = bool(st.session_state.pop('skip_suggestion_boot', False))
    with _SUGGESTION_JOB_LOCK:
        future = _SUGGESTION_JOBS.get(key)
    if future and future.done():
        cached = future.result()
        st.session_state['suggestions'] = cached
        with _SUGGESTION_JOB_LOCK:
            _SUGGESTION_JOBS.pop(key, None)
    elif cached.get('key') != key:
        cached = {
            'key': key,
            'questions': list(DEFAULT_QUESTIONS),
            'source': 'คำถามเริ่มต้นจากฐานความรู้' if skip_boot else 'คำถามเริ่มต้นจากฐานความรู้ · AI กำลังปรับจากประวัติ',
            'pending': not skip_boot,
        }
        st.session_state['suggestions'] = cached
        with _SUGGESTION_JOB_LOCK:
            if not skip_boot and key not in _SUGGESTION_JOBS:
                _SUGGESTION_JOBS[key] = _UI_JOB_EXECUTOR.submit(
                    _generate_suggestions_job,
                    settings,
                    products,
                    history,
                    owner,
                    st.session_state['lead_session_id'],
                    key,
                )
    st.markdown('#### ถามต่อเรื่องไหนดี')
    st.caption(cached['source'])
    selected = None
    for offset in range(0, len(cached['questions']), 2):
        columns = st.columns(2)
        for column, index in zip(columns, range(offset, min(offset+2, len(cached['questions'])))):
            with column:
                if st.button(cached['questions'][index], key=f'suggest_{index}', use_container_width=True):
                    selected = cached['questions'][index]
    return selected


def display_text(text, products):
    text = text.replace('monthly_thb', 'เดือน').replace('annual_thb', 'ปี').replace('THB', 'บาท')
    for product_id, item in products.items():
        text = text.replace(product_id, next(iter(item.get('product_aliases', [])), item['product_name']))
    return text


def render_message(message: dict[str, Any], products: dict[str, Any]) -> None:
    """Render a transcript row with an explicit human/AI identity."""

    sender_type = message.get("sender_type") or ("customer" if message.get("role") == "user" else "ai")
    label = message.get("sender_label") or {
        "customer": "ลูกค้า", "staff": "พนักงานขาย", "ai": "AI ผู้ช่วยผลิตภัณฑ์"
    }.get(sender_type, "ระบบ")
    avatar = "👤" if sender_type == "customer" else ("🧑‍💼" if sender_type == "staff" else "🤖")
    with st.chat_message("user" if sender_type == "customer" else "assistant", avatar=avatar):
        st.caption(label)
        st.markdown(display_text(message["content"], products))
        if message.get("citations"):
            st.caption("แหล่งอ้างอิง")
            for citation in message["citations"]:
                st.markdown(
                    f"- `{citation['doc_id']}` p.{citation['page']} "
                    f"[เอกสารต้นทาง]({citation['source_url']})"
                )
        if message.get("error_code"):
            st.error(f"Service state: {message['error_code']}")


@st.fragment(run_every="2s")
def sync_customer_handoff_status(settings: Settings) -> None:
    """Refresh handoff ownership without making the customer reload the page."""

    session_id = st.session_state.get("lead_session_id")
    if not session_id:
        return
    connection = connect_from_settings(settings)
    try:
        handoff = get_open_handoff(connection, session_id)
    finally:
        connection.close()
    status = handoff["status"] if handoff else None
    staff_name = handoff.get("assigned_staff_name") if handoff else None
    if (
        status != st.session_state.get("handoff_status")
        or staff_name != st.session_state.get("handoff_staff_name")
    ):
        st.session_state["handoff_status"] = status
        st.session_state["handoff_staff_name"] = staff_name
        load_ui_history(settings)
        st.rerun(scope="app")


def render_assistant_live(settings: Settings) -> None:
    render_session_controls(settings)
    load_ui_history(settings)
    if settings.llm_provider == "google_ai_studio":
        if not settings.google_api_key or not settings.llm_model:
            st.error("Google AI Studio is selected but GOOGLE_API_KEY or LLM_MODEL is not configured.")
            return
        mode = "Gemini live"
    elif settings.llm_provider == "none":
        mode = "Offline evidence extract"
    elif settings.llm_provider == "azure_openai":
        st.error("Azure OpenAI is selected, but this local UI build does not support the Azure LLM path yet. No offline fallback was selected automatically.")
        return
    else:
        st.error("Unsupported LLM provider configuration.")
        return
    manifest = json.loads((ROOT / "knowledge_base" / "manifest.json").read_text(encoding="utf-8"))
    products = {item["product_id"]: item for item in manifest["documents"]}
    if not products:
        st.error("No InsureX product documents are configured.")
        return
    boot_loading = bool(st.session_state.get("boot_first_paint"))
    is_loading = bool(st.session_state.get("active_job_id") or boot_loading)
    account_icon = (
        '<span class="loading-spinner boot-spinner"></span><span class="boot-person">♙</span>'
        if boot_loading
        else ('<span class="loading-spinner"></span>' if st.session_state.get("active_job_id") else '♙')
    )
    account_class = "is-loading boot-loading" if boot_loading else ("is-loading" if st.session_state.get("active_job_id") else "")
    st.markdown(
        f"""
        <section class="insurex-hero">
            <div class="hero-topline">
                <div class="brand-lockup">
                    <span class="brand-mark">iX</span>
                    <div>
                        <div class="brand-name">INSUREX</div>
                        <div class="brand-subtitle">PRODUCT INTELLIGENCE</div>
                    </div>
                </div>
                <div class="hero-actions">
                    <span class="account-chip {account_class}">
                        <span class="account-icon" aria-hidden="true">{account_icon}</span>
                        <span>{st.session_state['principal']['username']}</span>
                    </span>
                </div>
            </div>
            <h1>ข้อมูลชัดเจน เพื่อทุกบทสนทนา</h1>
            <p>ระบบค้นคืนจากเอกสาร InsureX ทั้ง corpus ตามคำถาม แล้วแสดงเอกสารและเลขหน้าเป็นหลักฐาน</p>
            <div class="hero-note">ค้นเอกสาร · ตรวจแหล่งอ้างอิง · เก็บข้อมูลอย่างเป็นขั้นตอน</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("citation_by_content", {})
    st.session_state.setdefault("assistant_messages", [])
    render_runtime_log()
    for message in st.session_state["assistant_messages"]:
        render_message(message, products)

    with st.expander("ข้อมูลผู้สนใจของบัญชีนี้", expanded=bool(st.session_state.get("lead_panel"))):
        st.write(display_text(st.session_state.get("lead_panel") or "เริ่มด้วยคำว่า สนใจสมัคร แล้วระบุชื่อผลิตภัณฑ์", products))
        st.caption("ข้อมูลที่ให้ไว้สำหรับการติดต่อกลับ")
    sync_customer_handoff_status(settings)
    handoff_status = st.session_state.get("handoff_status")
    if handoff_status == "pending":
        st.info("ส่งคำถามให้พนักงานขายแล้ว กำลังรอพนักงานรับเคส — AI/RAG จะพักการตอบในบทสนทนานี้ชั่วคราว")
    elif handoff_status == "active":
        staff_name = st.session_state.get("handoff_staff_name") or "พนักงานขาย"
        st.info(f"{staff_name} กำลังดูแลบทสนทนานี้ — AI/RAG จะกลับมาหลังพนักงานปิดเคส")
    if render_active_turn():
        return
    placeholder = "ส่งข้อความถึงพนักงานขาย" if handoff_status else "ถามเรื่องประกัน หรือพิมพ์ สนใจสมัคร เพื่อเริ่มเก็บข้อมูล"
    question = st.chat_input(placeholder)
    if not question and not handoff_status:
        question = render_suggestions(settings, products)
    if not question:
        return
    request_id = stable_turn_request_id(question)
    try:
        _start_turn_job(
            settings,
            question=question,
            request_id=request_id,
            product_ids=set(products),
            mode=mode,
        )
        st.rerun()
    except Exception as exc:
        st.session_state["assistant_messages"].append({"role": "user", "content": question, "sender_type": "customer"})
        st.session_state["assistant_messages"].append(
            {
                "role": "assistant",
                "sender_type": "ai",
                "sender_label": "AI ผู้ช่วยผลิตภัณฑ์",
                "content": "The assistant service is temporarily unavailable.",
                "citations": [],
                "error_code": type(exc).__name__,
            }
        )
        st.session_state.pop("pending_turn", None)
        st.rerun()


def render_staff_workspace(settings: Settings) -> None:
    """Human handoff queue. Staff messages never invoke Gemini or LangGraph."""

    manifest = json.loads((ROOT / "knowledge_base" / "manifest.json").read_text(encoding="utf-8"))
    products = {item["product_id"]: item for item in manifest["documents"]}
    staff_id = st.session_state["principal"]["owner_id"]
    connection = connect_from_settings(settings)
    try:
        cases = list_handoff_queue(connection, staff_id)
    finally:
        connection.close()
    with st.sidebar:
        st.markdown("#### คิวส่งต่อจาก AI")
        if not cases:
            st.caption("ไม่มีเคสที่รอพนักงาน")
        for item in cases:
            badge = "รอรับเคส" if item["status"] == "pending" else f"กำลังดูแล · {item.get('assigned_staff_name') or 'พนักงาน'}"
            label = f"{item['customer_name']} · {badge}"
            if st.button(label, key=f"handoff_case_{item['case_id']}", use_container_width=True):
                st.session_state["staff_case_id"] = item["case_id"]
                st.session_state["staff_customer_session"] = item["session_id"]
                st.session_state["staff_customer_name"] = item["customer_name"]
                st.rerun()
    case_id = st.session_state.get("staff_case_id")
    selected = st.session_state.get("staff_customer_session")
    selected_case = next((item for item in cases if item["case_id"] == case_id), None)
    if not selected or not selected_case:
        st.markdown("<section class='insurex-hero'><h1>คิวช่วยเหลือลูกค้า</h1><p>แสดงบทสนทนาที่ AI/RAG ตอบไม่ได้ หรือลูกค้าขอคุยกับพนักงานโดยตรง</p></section>", unsafe_allow_html=True)
        return
    connection = connect_from_settings(settings)
    try:
        history = load_history_for_staff(
            connection, staff_owner_id=staff_id, session_id=selected, limit=100
        )
        lead = connection.execute(
            "SELECT l.* FROM leads l JOIN sessions s ON s.owner_id=l.owner_id WHERE s.session_id=?",
            (selected,),
        ).fetchone()
        lead_status = lead_profile_status(connection, lead["owner_id"]) if lead else None
    finally:
        connection.close()
    customer_name = st.session_state.get("staff_customer_name", "ลูกค้า")
    st.markdown(
        f"<section class='insurex-hero'><div class='hero-note'>HUMAN HANDOFF</div><h1>{escape(customer_name)}</h1><p>AI/RAG หยุดตอบชั่วคราวจนกว่าพนักงานจะปิดเคส</p></section>",
        unsafe_allow_html=True,
    )
    st.caption("สาเหตุที่ส่งต่อ: " + str(selected_case["trigger_reason"]))
    if lead:
        stale_text = "ควรยืนยันข้อมูลใหม่" if lead_status and lead_status["stale"] else "ข้อมูลยังอยู่ในรอบ 6 เดือน"
        with st.expander("ข้อมูล Lead", expanded=False):
            st.write(f"ชื่อ: {lead['name']} | อาชีพ: {lead['occupation']} | รายได้: {lead['income_thb']} บาท/เดือน | เบอร์: {lead['phone_normalized']} | ผลิตภัณฑ์: {lead['product_id']}")
            st.caption(stale_text)
    for message in history:
        render_message(message, products)
    if selected_case["status"] == "pending":
        if st.button("รับเคส", key=f"accept_case_{case_id}", type="primary", use_container_width=True):
            connection = connect_from_settings(settings)
            try:
                accept_handoff(connection, staff_owner_id=staff_id, case_id=int(case_id))
            finally:
                connection.close()
            st.rerun()
        st.info("กดรับเคสก่อนตอบลูกค้า เพื่อป้องกันพนักงานหลายคนตอบพร้อมกัน")
        return
    assigned_to_me = selected_case.get("assigned_staff_id") == staff_id
    if not assigned_to_me:
        st.info(f"เคสนี้กำลังดูแลโดย {selected_case.get('assigned_staff_name') or 'พนักงานคนอื่น'}")
        return
    action_col, resolve_col = st.columns([0.72, 0.28], gap="small")
    with resolve_col:
        if st.button("แก้ไขปัญหาแล้ว", key=f"resolve_case_{case_id}", type="primary", use_container_width=True):
            connection = connect_from_settings(settings)
            try:
                resolve_handoff(connection, staff_owner_id=staff_id, case_id=int(case_id))
            finally:
                connection.close()
            for key in ("staff_case_id", "staff_customer_session", "staff_customer_name"):
                st.session_state.pop(key, None)
            st.rerun()
    reply = st.chat_input("ตอบลูกค้าในนามพนักงานขาย")
    if reply:
        connection = connect_from_settings(settings)
        try:
            append_staff_message(connection, staff_owner_id=staff_id, session_id=selected, content=reply)
        finally:
            connection.close()
        st.rerun()

def render_authentication(settings: Settings, cookie_manager: stx.CookieManager | None = None) -> bool:
    # Streamlit exposes request cookies synchronously on a fresh browser
    # connection. The component is used only for browser-side set/delete.
    cookies = dict(st.context.cookies)
    cookie_token = cookies.get(AUTH_COOKIE_NAME)

    if not st.session_state.get("principal") and cookie_token:
        connection = connect_from_settings(settings)
        try:
            principal = restore_login_session(connection, cookie_token)
        finally:
            connection.close()
        if principal:
            st.session_state["principal"] = principal
            st.session_state["auth_session_token"] = cookie_token
        else:
            cookie_manager = cookie_manager or stx.CookieManager(key="insurex_cookie_manager_expired")
            delete_cookie_safely(
                cookie_manager,
                AUTH_COOKIE_NAME,
                key="insurex_auth_cookie_expired",
            )

    if st.session_state.get("principal"):
        with st.sidebar:
            st.markdown("### " + st.session_state["principal"]["username"])
            if st.button("ออกจากระบบ", key="logout", disabled=bool(st.session_state.get("active_job_id"))):
                cookie_manager = cookie_manager or stx.CookieManager(key="insurex_cookie_manager_logout")
                connection = connect_from_settings(settings)
                try:
                    revoke_login_session(
                        connection,
                        st.session_state.get("auth_session_token") or cookie_token,
                    )
                finally:
                    connection.close()
                delete_cookie_safely(
                    cookie_manager,
                    AUTH_COOKIE_NAME,
                    key="insurex_auth_cookie_logout",
                )
                st.session_state.clear()
                st.rerun()
        for key in ('login_password', 'signup_password', 'signup_confirm'):
            st.session_state.pop(key, None)
        return True
    st.markdown('<div class="auth-banner"><span>PRODUCT ASSISTANT</span><h1>ผู้ช่วยข้อมูลผลิตภัณฑ์ประกันสำหรับลูกค้า</h1></div>', unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["เข้าสู่ระบบ", "สร้างบัญชี"])
    with login_tab:
        with st.form("login_form"):
            username = st.text_input("ชื่อบัญชี", key="login_username")
            password = st.text_input("รหัสผ่าน", type="password", key="login_password")
            submitted = st.form_submit_button("เข้าสู่พื้นที่ทำงาน", type="primary")
        if submitted:
            loading_slot = st.empty()
            with loading_slot:
                render_action_loading("กำลังตรวจสอบบัญชี")
            connection = connect_from_settings(settings)
            try:
                principal = authenticate(connection, username, password)
                token = create_login_session(connection, principal["owner_id"])
                st.session_state["principal"] = principal
                st.session_state["auth_session_token"] = token
                st.session_state["skip_suggestion_boot"] = True
                # Persist on the authenticated rerun. This lets the workspace
                # paint immediately instead of waiting for the cookie iframe
                # before the login rerun can start.
                st.session_state["pending_auth_cookie"] = token
            except PermissionError as exc:
                st.error(str(exc))
            else:
                st.rerun()
            finally:
                connection.close()
                loading_slot.empty()
    with signup_tab:
        with st.form("signup_form"):
            username = st.text_input("ชื่อบัญชี", key="signup_username")
            password = st.text_input("รหัสผ่านอย่างน้อย 10 ตัว", type="password", key="signup_password")
            confirm = st.text_input("ยืนยันรหัสผ่าน", type="password", key="signup_confirm")
            submitted = st.form_submit_button("สร้างบัญชี")
        if submitted:
            connection = connect_from_settings(settings)
            try:
                if password != confirm:
                    raise ValueError("รหัสผ่านไม่ตรงกัน")
                register(connection, username, password)
                st.success("สร้างบัญชีแล้ว กรุณาเข้าสู่ระบบ")
            except ValueError as exc:
                st.error(str(exc))
            finally:
                connection.close()
    return False


def main() -> None:
    st.set_page_config(page_title="InsureX · Service workspace", page_icon="◈", layout="wide")
    st.markdown("<style>" + local_font_css() + (ROOT / "ui/assets/workspace.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)
    first_paint = "boot_first_paint" not in st.session_state
    if first_paint:
        st.session_state["boot_first_paint"] = True
    settings = Settings.from_env()
    role = (st.session_state.get("principal") or {}).get("account_role", "customer")
    workspace_label = "SALES WORKSPACE" if role == "staff" else "CUSTOMER WORKSPACE"
    with st.sidebar:
        st.markdown(f'<div class="workspace-brand">iX <span>INSUREX<br><small>{workspace_label}</small></span></div>', unsafe_allow_html=True)
    if first_paint:
        render_boot_loading()
    if render_authentication(settings):
        if st.session_state["principal"].get("account_role") == "staff":
            render_staff_workspace(settings)
        else:
            render_assistant_live(settings)
        pending_cookie = st.session_state.pop("pending_auth_cookie", None)
        if pending_cookie:
            # Run the cookie component after the workspace has already emitted
            # its visible UI, so persistence no longer delays first paint.
            cookie_manager = stx.CookieManager(key="insurex_cookie_manager")
            cookie_manager.set(
                AUTH_COOKIE_NAME,
                pending_cookie,
                key="insurex_auth_cookie_set",
                expires_at=datetime.now() + timedelta(seconds=LOGIN_TTL_SECONDS),
                max_age=LOGIN_TTL_SECONDS,
                same_site="strict",
            )
    if first_paint:
        st.session_state["boot_first_paint"] = False


if __name__ == "__main__":
    main()
