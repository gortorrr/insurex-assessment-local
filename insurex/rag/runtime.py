"""Local RAG runtime wiring for Chroma, LangGraph, Gemini, and leads.

The retrieval and lead paths are usable offline. Gemini is optional at runtime;
when it is absent, the graph returns a controlled provider error rather than
turning a deterministic test result into a live-model claim.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError

from insurex.config import Settings
from insurex.leads import (
    LeadDraft,
    confirm_lead,
    extract_lead_draft,
    is_explicit_cancellation,
    lead_route,
    lead_field_prompt,
    lead_summary,
    lead_profile_status,
    mark_lead_refresh_requested,
    load_account_lead_draft,
    merge_lead_drafts,
    missing_lead_fields,
    save_lead,
    save_lead_draft,
    validate_lead,
)
from insurex.rag.contracts import Citation, RetrievedChunk, validate_answer_support
from insurex.rag.graph import RagServices
from insurex.rag.contracts import evidence_supports_query
from insurex.rag.index import find_product_matches, retrieve_from_chroma, verify_corpus_current


class GeneratedCitation(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    chunk_id: str
    source_url: str


def format_offline_evidence_extract(text: str, *, doc_id: str, page_number: int) -> str:
    """Render verified text as readable evidence without pretending it is LLM output."""

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    quoted = "\n".join(f"> {line}" for line in lines)
    return (
        "### ข้อความหลักฐานจากเอกสาร\n\n"
        "> โหมดออฟไลน์: ข้อความนี้คัดจาก PDF ที่ตรวจแล้ว ไม่ใช่คำตอบที่สร้างโดย LLM\n\n"
        f"**เอกสาร:** `{doc_id}`  \n**หน้า:** {page_number}\n\n"
        f"{quoted}\n\n"
        "> ข้อความอาจเริ่มหรือต่อเนื่องจากช่วงอื่นของหน้า ให้ใช้ citation เปิดเอกสารต้นฉบับประกอบ"
    )


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[GeneratedCitation] = Field(default_factory=list)
    response_type: Literal["answer", "clarification", "no_answer"] = "answer"


def _flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Prepare a Pydantic schema for Gemini's native JSON-schema mode.

    This follows the provider-adapter pattern used by SillyTavern: resolve
    local definitions first, then remove keywords that the Gemini schema
    surface does not accept. It keeps the application schema as the source of
    truth and does not weaken post-response Pydantic validation.
    """

    definitions = schema.get("$defs", {}) if isinstance(schema, dict) else {}

    def resolve(value: Any, parents: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [resolve(item, parents) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            if name in parents or name not in definitions:
                return {}
            return resolve(definitions[name], parents + (name,))
        unsupported = {"$defs", "$schema", "default", "additionalProperties", "exclusiveMinimum", "propertyNames"}
        return {
            key: resolve(item, parents)
            for key, item in value.items()
            if key not in unsupported
        }

    return resolve(schema)


def _contains_no_answer_framing(answer: str) -> bool:
    """Detect a no-answer explanation that must not be accepted as grounded prose."""

    compact = re.sub(r"\s+", "", answer.casefold())
    markers = (
        "\u0e44\u0e21\u0e48\u0e1e\u0e1a\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25",
        "\u0e44\u0e21\u0e48\u0e1e\u0e1a\u0e23\u0e32\u0e22\u0e25\u0e30\u0e40\u0e2d\u0e35\u0e22\u0e14",
        "\u0e44\u0e21\u0e48\u0e21\u0e35\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25",
        "\u0e02\u0e49\u0e2d\u0e21\u0e39\u0e25\u0e44\u0e21\u0e48\u0e40\u0e1e\u0e35\u0e22\u0e07\u0e1e\u0e2d",
    )
    return any(marker in compact for marker in markers)


def format_grounded_evidence_fallback(text: str) -> str:
    """Return a citation-safe extract when model drafts fail validation.

    Retrieval has already passed the graph's evidence gate before generation.
    Copying the verified source text is safer than converting a useful result
    into a false no-answer after Gemini exhausts its repair attempts.
    """

    compact = re.sub(r"\s+", " ", text).strip()
    # Preserve visible list boundaries from the PDF instead of returning one
    # dense paragraph. Numbered disease lists are common in the brochures.
    compact = re.sub(r"\s*•\s*", "\n", compact)
    compact = re.sub(r"\s+(?=\([1-9]\)\s)", "\n", compact)
    compact = re.sub(r"\s+(?=[1-9]\.\s+[ก-๙A-Za-z])", "\n", compact)
    raw_items = [item.strip(" -") for item in compact.splitlines() if item.strip(" -")]

    items: list[str] = []
    used = 0
    for item in raw_items:
        remaining = 1400 - used
        if remaining <= 0:
            break
        if len(item) > remaining:
            item = item[:remaining].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"
        items.append(item)
        used += len(item)

    bullets = "\n\n".join(f"- {item}" for item in items)
    return (
        "### ข้อมูลที่พบในเอกสารผลิตภัณฑ์\n\n"
        "รายละเอียดต่อไปนี้คัดจากส่วนที่เกี่ยวข้องมากที่สุดในเอกสาร:\n\n"
        f"{bullets}"
    )


class RequestBudgetExceeded(RuntimeError):
    pass


@dataclass
class RequestBudget:
    """Finite external-request budget used by the live smoke runner."""

    max_requests: int = 10
    requests: int = 0
    retry_requests: int = 0
    usage: list[dict[str, Any]] = field(default_factory=list)

    def call(self, fn: Callable[[], Any]) -> Any:
        if self.requests >= self.max_requests:
            raise RequestBudgetExceeded(f"request budget exhausted at {self.max_requests}")
        self.requests += 1
        try:
            result = fn()
        except Exception:
            self.retry_requests += 1
            raise
        self.record_usage(result)
        return result

    def record_usage(self, result: Any) -> None:
        raw = result.get("raw") if isinstance(result, dict) else result
        metadata = getattr(raw, "usage_metadata", None) or getattr(raw, "response_metadata", {}).get("usage_metadata")
        if metadata:
            self.usage.append({str(k): v for k, v in dict(metadata).items()})


def _local_embedding_path(settings: Settings) -> str:
    candidate = Path(settings.embedding_model)
    if candidate.exists():
        return str(candidate)
    local = Path("rag/models") / "multilingual-e5-small"
    return str(local) if local.exists() else settings.embedding_model


def _llm_evidence_text(text: str) -> str:
    """Keep raw extraction unchanged; unverified chunks are quarantined upstream."""

    return text


def _generation_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    usable = [chunk for chunk in chunks if chunk.text_quality == "verified"]
    if not usable:
        raise ValueError("all retrieved PDF evidence is unverified and has been quarantined")
    return usable


def _assess_local(query: str, chunks: list[RetrievedChunk]) -> str:
    """Conservative lexical evidence gate, not a calibrated probability."""

    if not chunks:
        return "insufficient"
    if evidence_supports_query(query, chunks):
        return "sufficient"
    follow_up = any(term in query.lower() for term in ("แล้ว", "แบบนั้น", "ดังกล่าว", "follow up", "what about"))
    if follow_up and len({chunk.product_id for chunk in chunks}) == 1:
        return "sufficient"
    if not chunks or not evidence_supports_query(query, chunks):
        return "insufficient"
    return "sufficient"


def _rewrite_query(query: str) -> str:
    """Make one bounded retrieval rewrite useful without inventing facts."""

    suffix = " ความคุ้มครอง ระยะเวลา เงื่อนไข"
    return query if suffix.strip() in query else query.strip() + suffix


def _retrieve_factory(settings: Settings) -> Callable[[str, str | None], list[RetrievedChunk]]:
    model = _local_embedding_path(settings)
    try:
        manifest = json.loads(Path(settings.kb_manifest_path).read_text(encoding="utf-8"))
        product_aliases = {
            str(item["product_id"]): [str(alias) for alias in item.get("product_aliases", [])]
            for item in manifest.get("documents", [])
        }
    except (OSError, KeyError, TypeError, ValueError):
        product_aliases = {}

    def retrieve(query: str, product_id: str | None) -> list[RetrievedChunk]:
        verify_corpus_current(settings.chroma_path, settings.kb_manifest_path)
        return retrieve_from_chroma(
            query,
            chroma_path=settings.chroma_path,
            embedding_model=model,
            embedding_device=settings.embedding_device,
            product_id=product_id,
            product_aliases=product_aliases,
            top_k=5,
            retrieval_mode=settings.retrieval_mode,
        )

    return retrieve


def _citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            doc_id=chunk.doc_id,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            source_url=chunk.source_url,
        )
        for chunk in chunks
    ]


def build_offline_services(settings: Settings, *, lead_turn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> RagServices:
    retrieve = _retrieve_factory(settings)

    def generate(query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
        chunks = _generation_chunks(chunks)
        if not chunks:
            raise ValueError("cannot generate without retrieved evidence")
        # This is an evidence extract for offline contract tests, not an LLM result.
        chunk = chunks[0]
        return format_offline_evidence_extract(chunk.text, doc_id=chunk.doc_id, page_number=chunk.page_number), _citations([chunk])

    return RagServices(
        retrieve=retrieve,
        assess=_assess_local,
        generate=generate,
        rewrite=_rewrite_query,
        lead_turn=lead_turn,
    )


@dataclass
class GeminiServices:
    settings: Settings
    model: Any
    budget: RequestBudget

    def services(self, *, lead_turn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> RagServices:
        retrieve = _retrieve_factory(self.settings)

        def generate(query: str, chunks: list[RetrievedChunk]) -> tuple[str, list[Citation]]:
            chunks = _generation_chunks(chunks)
            if not chunks:
                raise ValueError("cannot generate without retrieved evidence")
            evidence = "\n\n".join(
                f"[{c.doc_id} p.{c.page_number} {c.chunk_id} source_url={c.source_url}] "
                f"{_llm_evidence_text(c.text[:2200])}"
                for c in chunks
            )
            prompt = (
                "ตอบคำถามภาษาไทยโดยใช้เฉพาะหลักฐานด้านล่าง ห้ามเดาและห้ามอ้างข้อมูลนอกหลักฐาน "
                "คืน JSON ตาม schema ที่กำหนด โดย citations ต้องคัดลอก doc_id, page_number, "
                "chunk_id และ source_url จากหลักฐานให้ตรงทุกตัวอักษร หากหลักฐานไม่พอให้ตอบว่าไม่พบข้อมูล\n\n"
                f"คำถาม: {query}\n\nหลักฐาน:\n{evidence}"
            )
            current_time = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(timespec="seconds")
            prompt += (
                "\n\nRUNTIME CONTEXT:\n"
                f"Current date and time in Asia/Bangkok: {current_time}\n"
                "Use this only to interpret relative time words such as today or currently; "
                "it is not product evidence.\n"
                "Reason over all retrieved products before answering. If multiple products are supported, "
                "compare or summarize them when the question allows it; ask for a product only when the evidence "
                "cannot support a safe answer. Do not emit a generic no-answer merely because multiple products "
                "are present. Answer in Thai Markdown and cite every factual claim. Set response_type to answer for a grounded answer, clarification for a useful question that makes no product claim, or no_answer only when the evidence cannot answer. Return only the structured schema."
            )
            prompt += (
                "\n\nMARKDOWN PRESENTATION RULES:\n"
                "Make the answer easy to scan in a sales workspace. Start with a level-3 heading (###), "
                "then one short summary paragraph. Put each distinct benefit, condition, or plan on its own "
                "Markdown bullet using '- **Label:** detail'. Insert a blank line before and after every list. "
                "Use level-4 headings (####) only when there are multiple sections. Keep paragraphs to at most "
                "three sentences. Do not use an HTML table and do not put citations inside the answer text; "
                "the application renders sources separately."
            )
            prompt += (
                "\n\nSTRICT MULTI-PRODUCT RULE:\n"
                "If a shared generic product name matches multiple retrieved products and the question asks "
                "for conditions or benefits, compare every supported product in the evidence and label each "
                "product clearly. Never say information is missing for one product merely because another "
                "product has the matching passage. A no_answer response must be short, contain no product facts, "
                "and have no citations."
            )
            structured_schema = _flatten_json_schema(GeneratedAnswer.model_json_schema())
            structured_schema["required"] = ["answer", "citations", "response_type"]
            structured = self.model.with_structured_output(
                structured_schema,
                method="json_schema",
                include_raw=True,
            )
            repair_reason = ""
            last_response_type: Literal["answer", "clarification", "no_answer"] = "answer"

            def grounded_fallback() -> tuple[str, list[Citation], Literal["answer"]]:
                chunk = chunks[0]
                return format_grounded_evidence_fallback(chunk.text), _citations([chunk]), "answer"

            for attempt in range(3):
                request_prompt = prompt
                if repair_reason:
                    request_prompt += (
                        "\n\nVALIDATION REPAIR:\n"
                        f"The previous draft failed evidence validation: {repair_reason}\n"
                        "Rewrite the answer using only exact claims supported by the supplied chunks. "
                        "Remove unsupported numbers or conditions, and cite the chunk that supports each claim."
                    )
                # LangChain retries transport failures internally, but a live
                # Streamlit worker can still receive a transient provider
                # failure after model warm-up. Retry the whole structured
                # request within the configured and traced request budget.
                for provider_attempt in range(self.settings.max_retrieval_retries + 1):
                    try:
                        result = self.budget.call(lambda: structured.invoke(request_prompt))
                        break
                    except Exception:
                        if provider_attempt >= self.settings.max_retrieval_retries:
                            # Retrieval and the graph evidence gate already
                            # succeeded. Keep the user on the grounded RAG path
                            # during a transient Gemini failure by returning a
                            # cited extract from the best verified chunk.
                            return grounded_fallback()
                        time.sleep(0.5 * (provider_attempt + 1))
                parsed = result.get("parsed") if isinstance(result, dict) else result
                if parsed is None:
                    raise ValueError("Gemini returned no structured answer")
                citation_parse_failed = False
                try:
                    payload = parsed if isinstance(parsed, GeneratedAnswer) else GeneratedAnswer.model_validate(parsed)
                except ValidationError as exc:
                    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
                        raise
                    response_type_value = parsed.get("response_type", "answer")
                    if response_type_value not in {"answer", "clarification", "no_answer"}:
                        response_type_value = "answer"
                    payload = GeneratedAnswer(
                        answer=parsed["answer"],
                        citations=[],
                        response_type=response_type_value,
                    )
                    citation_parse_failed = True
                    repair_reason = (
                        "The provider returned an incomplete citation object. Return every citation field "
                        "exactly: doc_id, page_number, chunk_id, and source_url."
                    )
                citations = [
                    Citation(
                        doc_id=item.doc_id,
                        page_number=item.page_number,
                        chunk_id=item.chunk_id,
                        source_url=item.source_url,
                    )
                    for item in payload.citations
                ]
                # A no_answer payload is only trustworthy when it is concise and
                # contains no product claim.  Gemini sometimes emits a factual
                # explanation with response_type=no_answer; returning that value
                # would contradict the evidence and hide a grounded answer.
                response_type = payload.response_type
                last_response_type = response_type
                if response_type == "clarification" and not citations:
                    return payload.answer, citations, response_type
                if citation_parse_failed and attempt == 2:
                    if response_type == "no_answer" or _contains_no_answer_framing(payload.answer):
                        return payload.answer, [], "no_answer"
                    fallback_citations = _citations(chunks)
                    valid, fallback_reason = validate_answer_support(
                        payload.answer, fallback_citations, chunks
                    )
                    if valid:
                        return payload.answer, fallback_citations, "answer"
                    repair_reason = fallback_reason
                    return grounded_fallback()
                if response_type == "no_answer":
                    repair_reason = (
                        "response_type=no_answer is invalid when the draft contains product facts or citations. "
                        "Use the supplied evidence to answer the question, including all supported products, "
                        "or return only a short no-answer with no factual explanation and no citations."
                    )
                    if attempt == 2:
                        return grounded_fallback()
                    continue
                if _contains_no_answer_framing(payload.answer):
                    repair_reason = (
                        "The draft uses no-answer framing such as '???????????' while also discussing product "
                        "evidence. Rewrite it as a direct comparison of the retrieved products with citations; "
                        "do not claim that a product lacks information when its evidence is present."
                    )
                    if attempt == 2:
                        return grounded_fallback()
                    continue
                valid, reason = validate_answer_support(payload.answer, citations, chunks)
                if valid:
                    return payload.answer, citations, response_type
                repair_reason = reason
                if attempt == 2:
                    # A model can select an incomplete citation subset even
                    # when its answer is supported by another retrieved chunk.
                    # After bounded repair attempts, validate once against the
                    # full evidence set and attach that set if it supports every
                    # claim. This preserves a concise grounded answer instead
                    # of dumping an unrelated top-ranked PDF chunk.
                    full_citations = _citations(chunks)
                    full_valid, full_reason = validate_answer_support(
                        payload.answer, full_citations, chunks
                    )
                    if full_valid:
                        return payload.answer, full_citations, response_type
                    repair_reason = full_reason
            return grounded_fallback()

        return RagServices(
            retrieve=retrieve,
            assess=_assess_local,
            generate=generate,
            rewrite=_rewrite_query,
            lead_turn=lead_turn,
        )


def build_gemini_services(settings: Settings, *, budget: RequestBudget | None = None, lead_turn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> GeminiServices:
    from insurex.llm import build_chat_model

    request_budget = budget or RequestBudget(max_requests=10)
    model = build_chat_model(settings)
    return GeminiServices(settings=settings, model=model, budget=request_budget)


@dataclass
class LeadTurnHandler:
    """LangGraph lead node implementation using a trusted local session context."""

    connection: Any
    session_id: str
    access_token: str
    product_catalog: set[str]
    request_id: str
    product_aliases: dict[str, list[str]] = field(default_factory=dict)
    extractor: Callable | None = None

    def __call__(self, text: str, state: dict[str, Any]) -> dict[str, Any]:
        from insurex.session import _authorize

        owner_id = _authorize(self.connection, self.session_id, self.access_token)["owner_id"]
        freshness = lead_profile_status(self.connection, owner_id)
        stored = load_account_lead_draft(
            self.connection,
            session_id=self.session_id,
            access_token=self.access_token,
        )
        previous_data = state.get("lead_draft") or {}
        previous = stored or (LeadDraft.model_validate(previous_data) if previous_data else LeadDraft())
        if freshness["stale"] and not freshness["refresh_requested_at"]:
            # Keep the old complete profile in `leads` until replacement data
            # is complete; collect the refresh in the account draft.
            previous = LeadDraft(product_id=freshness["product_id"])
            mark_lead_refresh_requested(self.connection, owner_id)
        # A cancelled collection may start fresh. A confirmed customer profile
        # remains shared across chats and is updated rather than duplicated.
        if previous.confirmation_status == "cancelled" and lead_route(text) == "lead":
            previous = LeadDraft()
        products = [
            (product_id, product_id.replace("-", " "))
            for product_id in sorted(self.product_catalog)
        ]
        previous_candidates = [
            product_id
            for product_id in state.get("lead_product_candidates", [])
            if product_id in self.product_catalog
        ]
        selected_product = self._selected_candidate(text, previous_candidates)
        product_candidates = find_product_matches(
            text,
            products,
            self.product_aliases,
        )
        if selected_product:
            product_id = selected_product
            product_candidates = []
        elif len(product_candidates) == 1:
            product_id = product_candidates[0]
            product_candidates = []
        elif len(product_candidates) > 1:
            product_id = None
        else:
            product_candidates = previous_candidates
            product_id = None if product_candidates else state.get("active_product_id")
        incoming = extract_lead_draft(text, product_id=product_id)
        deterministic_has_slots = bool(
            incoming.product_id
            or incoming.name
            or incoming.occupation
            or incoming.income is not None
            or incoming.income_min is not None
            or incoming.phone
        )
        if self.extractor and not deterministic_has_slots and not is_explicit_cancellation(text):
            try:
                llm_incoming = self.extractor(text, product_id, missing_lead_fields(previous))
                # Keep deterministic product routing and merge only additional
                # slots returned by the model.
                incoming = merge_lead_drafts(incoming, llm_incoming)
            except Exception:
                # The deterministic parser above remains available when the
                # provider is unavailable or rejects a partial-slot payload.
                # Never discard user-provided lead fields because an optional
                # LLM extraction pass failed.
                pass
        draft = merge_lead_drafts(previous, incoming)
        if product_candidates:
            # Do not silently retain a previous product when the customer has
            # just mentioned a different but ambiguous product family.
            draft = draft.model_copy(
                update={"product_id": None, "confirmation_status": "unconfirmed"}
            )
        if is_explicit_cancellation(text):
            draft = draft.model_copy(update={"confirmation_status": "cancelled"})
            save_lead_draft(
                self.connection, session_id=self.session_id,
                access_token=self.access_token, draft=draft,
            )
            return {
                "answer": "ยกเลิกการเก็บข้อมูลผู้สนใจแล้ว",
                "lead_collection_active": False,
                "lead_draft": draft.model_dump(mode="json"),
                "lead_missing": [],
                "lead_saved_id": None,
                "lead_confirmation_pending": False,
                "lead_confirmation_fingerprint": None,
                "lead_summary": lead_summary(draft),
                "lead_product_candidates": [],
            }
        try:
            save_lead_draft(
                self.connection, session_id=self.session_id,
                access_token=self.access_token, draft=draft,
            )
        except (ValueError, PermissionError) as exc:
            return {
                "answer": f"ยังบันทึกข้อมูลที่ได้รับไม่ได้: {exc}",
                "lead_collection_active": True,
                "lead_draft": draft.model_dump(mode="json"),
                "lead_missing": missing_lead_fields(draft),
                "lead_saved_id": None,
                "lead_confirmation_pending": False,
                "lead_confirmation_fingerprint": None,
                "lead_summary": lead_summary(draft),
                "error_code": "LEAD_DRAFT_SAVE_ERROR",
                "lead_product_candidates": product_candidates,
            }
        missing = missing_lead_fields(draft)
        if not missing:
            confirmed = confirm_lead(draft)
            try:
                validated = validate_lead(confirmed)
                lead_id = save_lead(
                    self.connection,
                    session_id=self.session_id,
                    access_token=self.access_token,
                    request_id=self.request_id,
                    lead=validated,
                    product_catalog=self.product_catalog,
                )
            except ValueError as exc:
                return {
                    "answer": f"ยังบันทึกไม่ได้ กรุณาแก้ข้อมูล: {exc}",
                    "lead_collection_active": True,
                    "lead_draft": draft.model_dump(mode="json"),
                    "lead_missing": missing,
                    "lead_saved_id": None,
                    "lead_confirmation_pending": False,
                    "lead_confirmation_fingerprint": None,
                    "lead_summary": lead_summary(draft),
                    "error_code": "LEAD_VALIDATION_ERROR",
                    "lead_product_candidates": product_candidates,
                }
            save_lead_draft(
                self.connection, session_id=self.session_id,
                access_token=self.access_token, draft=confirmed,
            )
            return {
                "answer": self._product_follow_up(confirmed.product_id),
                "lead_collection_active": False,
                "lead_draft": confirmed.model_dump(mode="json"),
                "lead_missing": [],
                "lead_saved_id": lead_id,
                "lead_confirmation_pending": False,
                "lead_confirmation_fingerprint": None,
                "lead_summary": lead_summary(confirmed),
                "lead_product_candidates": [],
            }
        if "product_id" in missing and product_candidates:
            question = self._product_choice_question(product_candidates)
        else:
            question = "กรุณาแจ้ง " + lead_field_prompt(missing) + " เพิ่มเติม"
        return {
            "answer": question,
            "lead_collection_active": True,
            "lead_draft": draft.model_dump(mode="json"),
            "lead_missing": missing,
            "lead_saved_id": None,
            "lead_confirmation_pending": False,
            "lead_confirmation_fingerprint": None,
            "lead_summary": lead_summary(draft),
            "lead_product_candidates": product_candidates,
        }

    def _selected_candidate(self, text: str, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        clean = re.sub(r"\s+", " ", text.strip().casefold())
        numbered = re.fullmatch(r"(?:ข้อ|ตัวเลือก|รายการ|อัน)?\s*(\d+)", clean)
        if numbered:
            index = int(numbered.group(1)) - 1
            if 0 <= index < len(candidates):
                return candidates[index]
        ordinals = {
            "อันแรก": 0, "ข้อแรก": 0, "รายการแรก": 0,
            "อันที่สอง": 1, "ข้อที่สอง": 1, "รายการที่สอง": 1,
            "อันสอง": 1, "ข้อสอง": 1,
        }
        index = ordinals.get(clean)
        return candidates[index] if index is not None and index < len(candidates) else None

    def _product_choice_question(self, candidates: list[str]) -> str:
        choices = "\n".join(
            f"{index}. **{self._product_label(product_id)}**"
            for index, product_id in enumerate(candidates, start=1)
        )
        return (
            "พบผลิตภัณฑ์ชื่อใกล้กัน กรุณาเลือกว่าหมายถึงรายการใด\n\n"
            f"{choices}\n\n"
            "พิมพ์ชื่อผลิตภัณฑ์หรือหมายเลขรายการ"
        )

    def _product_label(self, product_id: str) -> str:
        aliases = self.product_aliases.get(product_id, [])
        return next((alias for alias in aliases if re.search(r"[ก-๙]", alias)), product_id)

    def _product_follow_up(self, product_id: str) -> str:
        label = self._product_label(product_id)
        return (
            f"สอบถามเกี่ยวกับ {label} ต่อได้เลย เช่น ความคุ้มครอง "
            "ระยะเวลาชำระเบี้ย หรือผลประโยชน์ตามกรมธรรม์"
        )


def write_request_evidence(path: Path, budget: RequestBudget, *, model: str, status: str) -> None:
    payload = {
        "status": status,
        "model": model,
        "requests": budget.requests,
        "retry_requests": budget.retry_requests,
        "usage": budget.usage,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
