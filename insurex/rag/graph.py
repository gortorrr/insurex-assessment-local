"""LangGraph RAG workflow adapter.

The imports are lazy so offline analytics and lead tests remain runnable when
LangChain/LangGraph are not installed. Live graph creation must use approved,
tested dependency versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal, TypedDict

from insurex.rag.contracts import (
    AMBIGUOUS_PRODUCT_MESSAGE,
    NO_ANSWER_MESSAGE,
    RETRIEVAL_ERROR_MESSAGE,
    Citation,
    RetrievedChunk,
    assess_candidates,
    validate_answer_support,
)
from insurex.leads import lead_route


class RagDependencyError(RuntimeError):
    """Raised when the optional LangGraph runtime is unavailable."""


class RagState(TypedDict, total=False):
    messages: list[Any]
    session_id: str
    query: str
    standalone_query: str
    active_product_id: str | None
    intent: str
    retrieved_chunks: list[RetrievedChunk]
    evidence_status: str
    evidence_reason: str
    answer: str
    response_type: str
    citations: list[Citation]
    retrieval_attempts: int
    error_code: str | None
    lead_collection_active: bool
    lead_draft: dict[str, Any]
    lead_missing: list[str]
    lead_saved_id: int | None
    lead_confirmation_pending: bool
    lead_confirmation_fingerprint: str | None
    lead_summary: str
    lead_product_candidates: list[str]
    conversation_history: list[dict[str, str]]
    lead_refresh_required: bool
    handoff_requested: bool


@dataclass(frozen=True)
class RagServices:
    retrieve: Callable[[str, str | None], list[RetrievedChunk]]
    assess: Callable[[str, list[RetrievedChunk]], str]
    generate: Callable[[str, list[RetrievedChunk]], tuple[str, list[Citation]] | tuple[str, list[Citation], str]]
    rewrite: Callable[[str], str]
    lead_turn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None


def build_standalone_query(query: str, history: list[dict[str, str]] | None = None, *, max_messages: int = 6, max_characters: int = 2400) -> str:
    """Resolve only a likely follow-up; unrelated questions stay history-free."""

    clean_query = query.strip()
    if not history or not _needs_follow_up_context(clean_query):
        return clean_query
    pieces: list[str] = []
    remaining = max_characters
    for item in history[-max_messages:]:
        role = item.get("role", "")
        content = str(item.get("content", "")).strip()
        # Product answers are useful context. Lead summaries and user field
        # collection turns can contain PII and are never copied into a product
        # retrieval/generation prompt.
        if role not in {"assistant", "user"} or _looks_like_lead_content(content) or not content or remaining <= 0:
            continue
        content = content[: min(400, remaining)]
        content = _redact_sensitive_context(content)
        pieces.append(f"{role}: {content}")
        remaining -= len(content) + len(role) + 2
    if not pieces:
        return clean_query
    return "บริบทคำตอบผลิตภัณฑ์ก่อนหน้า: " + " | ".join(pieces) + " | คำถามใหม่: " + clean_query


def _needs_follow_up_context(query: str) -> bool:
    lowered = query.lower().strip()
    if not lowered:
        return False
    follow_up_terms = (
        "แล้ว", "แบบนั้น", "ดังกล่าว", "อันนี้", "กี่ปี", "กี่เดือน", "เท่าไร",
        "เท่าไหร่", "ต่ออายุ", "จ่ายยังไง", "ชำระยังไง", "follow up", "how many", "what about",
    )
    return any(term in lowered for term in follow_up_terms)


def _looks_like_lead_content(content: str) -> bool:
    lowered = content.lower()
    return any(term in lowered for term in ("ชื่อ", "อาชีพ", "รายได้", "เบอร์", "โทรศัพท์", "phone", "income"))


def _redact_sensitive_context(content: str) -> str:
    content = __import__("re").sub(r"(?:\+66|0066|0)\d[\d\s().-]{7,13}", "[redacted-phone]", content)
    content = __import__("re").sub(
        r"(?i)(รายได้|income|เงินเดือน)\s*[:=]?\s*\d[\d,]*(?:\.\d+)?\s*(?:บาท|THB|USD|ดอลลาร์)?",
        r"\1 [redacted-income]",
        content,
    )
    return content


def prepare_turn(state: RagState) -> dict[str, Any]:
    """Reset per-turn retrieval state while preserving conversation context."""

    return {
        "standalone_query": "",
        "intent": "",
        "retrieved_chunks": [],
        "evidence_status": "",
        "evidence_reason": "",
        "answer": "",
        "response_type": "answer",
        "citations": [],
        "retrieval_attempts": 0,
        "error_code": None,
        "handoff_requested": False,
    }


def _route_intent(state: RagState) -> dict[str, Any]:
    query = state.get("query", "").strip()
    direct_intent = _direct_conversation_intent(query)
    if direct_intent:
        return {"intent": direct_intent}
    if lead_route(query) == "lead" and not _is_product_question(query):
        return {"intent": "lead"}
    if state.get("lead_collection_active") and not _is_product_question(query):
        return {"intent": "lead"}
    if not _is_product_related(query, state.get("conversation_history", [])):
        return {"intent": "out_of_scope" if query else "unsupported"}
    if state.get("lead_refresh_required"):
        return {"intent": "lead"}
    return {"intent": "question"}


def _normalized_chat_text(query: str) -> str:
    import re

    return re.sub(r"[\s!?.,;:]+", "", query.casefold())


def _direct_conversation_intent(query: str) -> str | None:
    """Route common conversation turns without retrieval or an LLM call."""

    compact = _normalized_chat_text(query)
    if not compact:
        return None
    if any(term in compact for term in (
        "ขอคุยกับพนักงาน", "ขอคุยกับเจ้าหน้าที่", "ติดต่อพนักงาน", "ติดต่อเจ้าหน้าที่",
        "เรียกพนักงาน", "เรียกเจ้าหน้าที่", "humanagent", "talktoagent",
    )):
        return "human_handoff"
    if compact in {"สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "หวัดดี", "ดีครับ", "ดีค่ะ", "hello", "hi", "hey"}:
        return "greeting"
    if compact.startswith("ขอบคุณ") or compact in {"ขอบใจ", "thanks", "thankyou", "โอเคขอบคุณ", "โอเคครับขอบคุณ"}:
        return "thanks"
    if compact in {"ลาก่อน", "ไว้คุยใหม่", "จบบทสนทนา", "บาย", "bye", "goodbye"}:
        return "goodbye"
    if compact in {
        "ช่วยอะไรได้บ้าง", "ทำอะไรได้บ้าง", "ถามอะไรได้บ้าง", "มีประกันอะไรบ้าง",
        "มีผลิตภัณฑ์อะไรบ้าง", "แนะนำการใช้งาน", "help", "whatcanyoudo",
    }:
        return "help"
    return None


def _is_product_related(query: str, history: list[dict[str, str]] | None = None) -> bool:
    lowered = query.casefold()
    product_terms = (
        "ประกัน", "กรมธรรม์", "ความคุ้มครอง", "คุ้มครอง", "ผลประโยชน์", "เบี้ย",
        "ชำระ", "เคลม", "ผู้เอาประกัน", "ผู้รับผลประโยชน์", "สัญญาเพิ่มเติม",
        "คุ้มตลอดชีพ", "คุ้มออมสุข", "คุ้มมั่นใจ", "ตลอดชีพ 90/20",
        "insurance", "policy", "coverage", "premium", "claim", "benefit",
    )
    if any(term in lowered for term in product_terms):
        return True
    return bool(history and _needs_follow_up_context(query))


def _is_product_question(query: str) -> bool:
    lowered = query.lower()
    terms = (
        "คุ้มครอง", "ผลประโยชน์", "ชำระ", "จ่าย", "เบี้ย", "อายุ", "ระยะเวลา",
        "เงื่อนไข", "กี่", "เท่าไร", "เท่าไหร่", "มีไหม", "ได้ไหม", "coverage",
        "benefit", "premium", "how many", "how long",
    )
    return "?" in query or any(term in lowered for term in terms)


def _route_after_intent(state: RagState) -> Literal["resolve_query", "collect_lead", "direct_response"]:
    if state.get("intent") == "lead":
        return "collect_lead"
    return "resolve_query" if state.get("intent") == "question" else "direct_response"


def build_rag_graph(services: RagServices, *, checkpointer: Any = None) -> Any:
    """Build and compile the explicit retrieve/grade/rewrite/generate graph."""

    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import add_messages
        # LangGraph resolves the locally declared schema through module globals
        # on Python 3.12; expose the optional reducer only after the optional
        # dependency has been imported.
        globals()["add_messages"] = add_messages
    except ImportError as exc:
        raise RagDependencyError(
            "LangGraph is not installed; install approved dependencies before building the live graph"
        ) from exc

    class GraphState(TypedDict, total=False):
        messages: Annotated[list[Any], add_messages]
        session_id: str
        query: str
        standalone_query: str
        active_product_id: str | None
        intent: str
        retrieved_chunks: list[RetrievedChunk]
        evidence_status: str
        evidence_reason: str
        answer: str
        response_type: str
        citations: list[Citation]
        retrieval_attempts: int
        error_code: str | None
        lead_collection_active: bool
        lead_draft: dict[str, Any]
        lead_missing: list[str]
        lead_saved_id: int | None
        lead_confirmation_pending: bool
        lead_confirmation_fingerprint: str | None
        lead_summary: str
        lead_product_candidates: list[str]
        conversation_history: list[dict[str, str]]
        lead_refresh_required: bool
        handoff_requested: bool

    def resolve_query(state: dict[str, Any]) -> dict[str, Any]:
        history = state.get("conversation_history", [])
        product = (state.get("lead_draft") or {}).get("product_id")
        if not history and product and state.get("lead_collection_active"):
            history = [{"role": "user", "content": "ผลิตภัณฑ์ " + product.replace("-", " ")}]
        return {
            "standalone_query": build_standalone_query(
                state.get("query", ""), history
            )
        }

    def collect_lead(state: dict[str, Any]) -> dict[str, Any]:
        if services.lead_turn is None:
            return {"answer": RETRIEVAL_ERROR_MESSAGE, "error_code": "LEAD_SERVICE_UNAVAILABLE"}
        try:
            return services.lead_turn(state.get("query", ""), dict(state))
        except Exception:
            return {
                "answer": RETRIEVAL_ERROR_MESSAGE,
                "error_code": "LEAD_SERVICE_ERROR",
                "lead_collection_active": True,
            }

    def direct_response(state: dict[str, Any]) -> dict[str, Any]:
        intent = state.get("intent")
        responses = {
            "greeting": (
                "สวัสดีครับ 👋 ผมช่วยค้นข้อมูลผลิตภัณฑ์ประกัน InsureX "
                "เปรียบเทียบความคุ้มครอง และรับข้อมูลสำหรับให้พนักงานติดต่อกลับได้ครับ"
            ),
            "thanks": "ยินดีครับ หากต้องการสอบถามเรื่องผลิตภัณฑ์ประกันเพิ่มเติม พิมพ์ถามได้เลยครับ",
            "goodbye": "ขอบคุณที่ใช้บริการครับ แล้วพบกันใหม่ครับ",
            "help": (
                "### สิ่งที่ผมช่วยได้\n\n"
                "- ตอบคำถามจากเอกสารผลิตภัณฑ์ **คุ้มตลอดชีพ พลัส**, **คุ้มตลอดชีพ ซีไอ พลัส**, "
                "**คุ้มออมสุข 25/15**, **คุ้มมั่นใจชัวร์** และ **ตลอดชีพ 90/20**\n"
                "- เปรียบเทียบความคุ้มครอง ระยะเวลา และเงื่อนไขจากเอกสาร\n"
                "- รับข้อมูลผู้สนใจเมื่อพิมพ์ว่า **สนใจสมัคร**\n"
                "- ส่งต่อพนักงานเมื่อพิมพ์ว่า **ขอคุยกับพนักงาน**"
            ),
            "human_handoff": "รับทราบครับ กำลังส่งบทสนทนานี้ให้พนักงานขายช่วยดูแลต่อ",
            "out_of_scope": (
                "ผมช่วยตอบเฉพาะข้อมูลผลิตภัณฑ์ประกัน InsureX จากเอกสารในระบบครับ "
                "ลองถามเรื่องความคุ้มครอง เบี้ย ระยะเวลา หรือเงื่อนไขของผลิตภัณฑ์ได้เลย"
            ),
            "unsupported": "กรุณาพิมพ์คำถามเกี่ยวกับผลิตภัณฑ์ประกันที่ต้องการทราบครับ",
        }
        return {
            "answer": responses.get(intent, responses["unsupported"]),
            "citations": [],
            "response_type": "direct",
            "handoff_requested": intent == "human_handoff",
        }

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        try:
            chunks = services.retrieve(
                state.get("standalone_query", ""), state.get("active_product_id")
            )
            return {"retrieved_chunks": chunks, "error_code": None}
        except Exception:
            return {"retrieved_chunks": [], "error_code": "RETRIEVAL_ERROR"}

    def assess(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error_code"):
            return {"evidence_status": "error", "evidence_reason": "retrieval service raised an exception"}
        chunks = state.get("retrieved_chunks", [])
        try:
            # Relevance is evaluated against the current user intent. The
            # resolved query is retrieval context, never proof that an old
            # conversation topic supports the new question.
            status = services.assess(state.get("query", ""), chunks)
            structural = assess_candidates(
                chunks,
                evidence_status=status,
                query=state.get("query", ""),
                active_product_id=state.get("active_product_id"),
            )
            return {"evidence_status": structural.status, "evidence_reason": structural.reason}
        except Exception:
            return {"evidence_status": "error", "evidence_reason": "evidence assessment raised an exception", "error_code": "ASSESS_ERROR"}

    def rewrite(state: dict[str, Any]) -> dict[str, Any]:
        attempts = state.get("retrieval_attempts", 0) + 1
        try:
            return {
                "standalone_query": services.rewrite(state.get("standalone_query", "")),
                "retrieval_attempts": attempts,
            }
        except Exception:
            return {"retrieval_attempts": attempts, "error_code": "REWRITE_ERROR", "evidence_status": "error"}

    def generate(state: dict[str, Any]) -> dict[str, Any]:
        try:
            generated = services.generate(
                state.get("standalone_query", ""), state.get("retrieved_chunks", [])
            )
            if len(generated) == 3:
                answer, citations, response_type = generated
            else:
                answer, citations = generated
                response_type = "answer"
            return {"answer": answer, "citations": citations, "response_type": response_type}
        except Exception:
            return {"answer": RETRIEVAL_ERROR_MESSAGE, "citations": [], "error_code": "GENERATION_ERROR", "evidence_status": "error"}

    def validate_answer(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("response_type") == "no_answer":
            return {"answer": NO_ANSWER_MESSAGE, "citations": [], "evidence_status": "insufficient"}
        if state.get("response_type") == "clarification" and state.get("answer") and not state.get("citations"):
            product_ids = {
                chunk.product_id
                for chunk in state.get("retrieved_chunks", [])
                if chunk.product_id
            }
            # A clarification from a multi-product turn must be produced by
            # the routing fallback. Gemini can otherwise label a factual
            # "no data, because ..." explanation as clarification and leak it
            # past citation validation.
            if state.get("evidence_status") == "sufficient" and len(product_ids) > 1:
                return {
                    "answer": "",
                    "citations": [],
                    "response_type": "answer",
                    "evidence_status": "ambiguous",
                    "evidence_reason": "multi-product clarification must be routed through the product disambiguation fallback",
                }
            return {
                "answer": "กรุณาระบุรายละเอียดที่ต้องการทราบเกี่ยวกับผลิตภัณฑ์นี้เพิ่มเติม",
                "evidence_status": "sufficient",
                "evidence_reason": "use a fixed clarification so unvalidated provider prose cannot introduce product claims",
            }
        try:
            valid, reason = validate_answer_support(
                state.get("answer", ""),
                state.get("citations", []), state.get("retrieved_chunks", [])
            )
        except Exception:
            return {"answer": RETRIEVAL_ERROR_MESSAGE, "citations": [], "evidence_status": "error", "evidence_reason": "citation validation raised an exception", "error_code": "CITATION_VALIDATION_ERROR"}
        if valid:
            return {"evidence_reason": reason}
        return {"answer": NO_ANSWER_MESSAGE, "citations": [], "evidence_status": "insufficient", "evidence_reason": reason}

    def fallback(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("error_code") or state.get("evidence_status") == "error":
            message = RETRIEVAL_ERROR_MESSAGE
        elif state.get("evidence_status") == "ambiguous":
            product_names = sorted(
                {
                    chunk.product_name.strip()
                    for chunk in state.get("retrieved_chunks", [])
                    if chunk.product_name and chunk.product_name.strip()
                }
            )
            if product_names:
                message = (
                    "คำถามนี้ตรงกับหลายผลิตภัณฑ์ที่เป็นไปได้:\n\n"
                    + "\n".join(f"- {name}" for name in product_names)
                    + "\n\nกรุณาระบุชื่อผลิตภัณฑ์ที่ต้องการ เช่น คุ้มตลอดชีพ พลัส หรือ คุ้มตลอดชีพ ซีไอ พลัส"
                )
            else:
                message = AMBIGUOUS_PRODUCT_MESSAGE
        else:
            message = NO_ANSWER_MESSAGE
        return {"answer": message, "citations": []}

    def route_after_assess(state: dict[str, Any]) -> str:
        status = state.get("evidence_status")
        if status == "sufficient":
            return "generate"
        if status == "weak" and state.get("retrieval_attempts", 0) < 1:
            return "rewrite"
        return "fallback"

    def route_after_validate(state: dict[str, Any]) -> str:
        if state.get("evidence_status") == "ambiguous" and not state.get("citations"):
            return "fallback"
        if state.get("response_type") == "clarification":
            return END
        return "fallback" if not state.get("citations") else END

    builder = StateGraph(GraphState)
    builder.add_node("prepare_turn", prepare_turn)
    builder.add_node("route_intent", _route_intent)
    builder.add_node("direct_response", direct_response)
    builder.add_node("resolve_query", resolve_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("assess_evidence", assess)
    builder.add_node("rewrite_query_once", rewrite)
    builder.add_node("generate_grounded_answer", generate)
    builder.add_node("validate_answer_citations", validate_answer)
    builder.add_node("fallback", fallback)
    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "route_intent")
    builder.add_conditional_edges("route_intent", _route_after_intent)
    builder.add_node("collect_lead", collect_lead)
    builder.add_edge("collect_lead", END)
    builder.add_edge("direct_response", END)
    builder.add_edge("resolve_query", "retrieve")
    builder.add_edge("retrieve", "assess_evidence")
    builder.add_conditional_edges(
        "assess_evidence",
        route_after_assess,
        {"generate": "generate_grounded_answer", "rewrite": "rewrite_query_once", "fallback": "fallback"},
    )
    builder.add_edge("rewrite_query_once", "retrieve")
    builder.add_edge("generate_grounded_answer", "validate_answer_citations")
    builder.add_conditional_edges(
        "validate_answer_citations",
        route_after_validate,
        {"fallback": "fallback", END: END},
    )
    builder.add_edge("fallback", END)
    return builder.compile(checkpointer=checkpointer)
