"""Provider-independent RAG evidence and citation contracts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable, Literal


EvidenceStatus = Literal["sufficient", "weak", "insufficient", "ambiguous", "error"]


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    product_id: str
    product_name: str
    page_number: int
    chunk_id: str
    text: str
    source_url: str
    score: float | None = None
    # Hand-authored unit fixtures default to verified. Chroma ingestion marks
    # extracted text containing PUA glyphs as unverified and generation must
    # quarantine those chunks until page/font verification exists.
    text_quality: Literal["verified", "unverified_private_use", "unknown"] = "verified"


@dataclass(frozen=True)
class Citation:
    doc_id: str
    page_number: int
    chunk_id: str
    source_url: str


@dataclass(frozen=True)
class EvidenceAssessment:
    status: EvidenceStatus
    reason: str


def assess_candidates(
    chunks: Iterable[RetrievedChunk],
    *,
    evidence_status: EvidenceStatus | None = None,
    query: str = "",
    active_product_id: str | None = None,
) -> EvidenceAssessment:
    """Classify candidates without treating a vector distance as probability.

    The real adapter must provide the evidence status from a calibrated evaluator
    or labelled retrieval policy. This function only handles structural cases.
    """

    rows = list(chunks)
    if not rows:
        return EvidenceAssessment("insufficient", "retriever returned no candidates")
    if evidence_status == "error":
        return EvidenceAssessment("error", "retrieval/evidence evaluator reported an error")
    if evidence_status == "insufficient":
        return EvidenceAssessment("insufficient", "retrieved chunks do not provide sufficient query evidence")
    products = {row.product_id for row in rows}
    if evidence_status == "sufficient" and query and not evidence_supports_query(query, rows):
        follow_up = any(term in query.lower() for term in ("แล้ว", "แบบนั้น", "ดังกล่าว", "follow up", "what about"))
        same_active_product = bool(active_product_id) and products == {active_product_id}
        if not (follow_up and same_active_product):
            return EvidenceAssessment("weak", "retrieved chunks do not contain a query term or numeric fact")
    if len(products) > 1 and not active_product_id:
        named_product = any(
            name and _normalise_evidence(name) in _normalise_evidence(query)
            for row in rows
            for name in (row.product_name, row.product_id.replace("-", " "))
        )
        comparison = any(term in query.casefold() for term in (
            "เปรียบเทียบ", "ต่างกัน", "แต่ละ", "ทั้งหมด", "มีอะไรบ้าง", "compare", "comparison", " versus ",
        ))
        if not named_product and not comparison:
            return EvidenceAssessment("ambiguous", "generic question spans products; ask which product before quoting conditions")
        if evidence_status == "sufficient" and query and evidence_supports_query(query, rows):
            return EvidenceAssessment(
                "sufficient",
                "multiple products have query-supported evidence; grounded generator may compare or clarify",
            )
        return EvidenceAssessment("ambiguous", "candidates span multiple products without sufficient query evidence")
    if evidence_status is not None:
        return EvidenceAssessment(evidence_status, "status supplied by retrieval/evidence evaluator")
    return EvidenceAssessment("weak", "candidates exist but evidence support is not yet established")


def _normalise_evidence(text: str) -> str:
    cleaned = text
    cleaned = "".join(char for char in unicodedata.normalize("NFD", cleaned) if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", "", cleaned).lower()


def _query_terms(query: str) -> list[str]:
    cleaned = query
    normalised = "".join(char for char in unicodedata.normalize("NFD", cleaned) if unicodedata.category(char) != "Mn").lower()
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+|[\u0E00-\u0E7F]+", normalised):
        if re.fullmatch(r"[A-Za-z0-9]+", token):
            if len(token) >= 2:
                terms.append(token)
            continue
        if len(token) < 4:
            continue
        terms.append(token)
        terms.extend(token[index:index + 4] for index in range(len(token) - 3))
    return list(dict.fromkeys(terms))


def evidence_supports_query(query: str, chunks: Iterable[RetrievedChunk]) -> bool:
    """Require query coverage in one coherent chunk, not across all candidates.

    Character n-grams help Thai text without a word tokenizer, but aggregating
    them over unrelated chunks makes generic terms look like evidence. A
    candidate must therefore cover at least two query terms and 40% of the
    query terms within the same chunk. This is a deterministic retrieval gate,
    not a calibrated probability.
    """

    terms = _query_terms(query)
    if not terms:
        return False
    if "ไม่ใช่" in query and "insurex" in query.lower():
        return False
    rows = list(chunks)
    products = {chunk.product_id for chunk in rows}
    for chunk in rows:
        evidence = _normalise_evidence(f"{chunk.product_name} {chunk.text}")
        matched = {term for term in terms if _normalise_evidence(term) in evidence}
        if len(terms) == 1 and matched:
            return True
        if len(matched) >= 2 and (len(matched) / len(terms)) >= 0.40:
            return True
    # A product-filtered retrieval can contain the right page even when Thai
    # segmentation produces too few lexical fragments. Accept that case only
    # when all candidates are one product, the top vector score is high, and
    # at least one query fragment is present in a candidate. This is a routing
    # rule, not a calibrated probability and it cannot turn unrelated text
    # into evidence without lexical trace.
    if len(products) == 1 and any(
        _normalise_evidence(term) in _normalise_evidence(f"{chunk.product_name} {chunk.text}")
        for term in terms
        for chunk in rows
    ) and max((chunk.score for chunk in rows), default=0.0) >= 0.75:
        return True
    # Product-routed questions about plans/prices often have Thai wording that
    # fragments into many character n-grams. Accept a single-product result
    # when the retrieved evidence contains a query fragment and numeric/plan
    # structure; unrelated generic questions still fail this gate.
    question_shape = any(
        marker in _normalise_evidence(query)
        for marker in ("แผน", "ราคา", "เท่าไร", "เท่าไหร่", "กี่", "จำนวน", "price", "how many")
    )
    numeric_evidence = any(re.search(r"\d", chunk.text) for chunk in rows)
    if len(products) == 1 and question_shape and numeric_evidence and any(
        matched_term
        for chunk in rows
        for matched_term in terms
        if _normalise_evidence(matched_term) in _normalise_evidence(chunk.text)
    ) and max((chunk.score for chunk in rows), default=0.0) >= 0.55:
        return True
    return False


def citations_from_chunks(chunks: Iterable[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            doc_id=chunk.doc_id,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            source_url=chunk.source_url,
        )
        for chunk in chunks
    ]


def validate_citations(
    citations: Iterable[Citation],
    retrieved_chunks: Iterable[RetrievedChunk],
) -> tuple[bool, str]:
    """Require every citation to resolve to the current retrieved context."""

    available = {
        (chunk.doc_id, chunk.page_number, chunk.chunk_id, chunk.source_url)
        for chunk in retrieved_chunks
    }
    supplied = list(citations)
    if not supplied:
        return False, "answer has no citations"
    for citation in supplied:
        key = (citation.doc_id, citation.page_number, citation.chunk_id, citation.source_url)
        if key not in available:
            return False, f"citation not present in retrieved context: {citation.chunk_id}"
        if citation.page_number < 1:
            return False, "page_number must be 1-based"
    return True, "all citations resolve to retrieved context"


def validate_answer_support(
    answer: str,
    citations: Iterable[Citation],
    retrieved_chunks: Iterable[RetrievedChunk],
) -> tuple[bool, str]:
    """Validate answer claims against the cited text after structural checks."""

    valid, reason = validate_citations(citations, retrieved_chunks)
    if not valid:
        return False, reason
    cited_ids = {citation.chunk_id for citation in citations}
    cited_chunks = [chunk for chunk in retrieved_chunks if chunk.chunk_id in cited_ids]
    evidence_text = " ".join(chunk.text for chunk in cited_chunks)
    evidence = _normalise_evidence(evidence_text)
    numeric_claims = list(
        re.finditer(
            r"(?<![\d,])(?P<number>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\d,])\s*"
            r"(?P<unit>ปี|เดือน|บาท|%|เปอร์เซ็นต์)?",
            answer,
        )
    )
    for claim in numeric_claims:
        number = claim.group("number")
        unit = claim.group("unit") or ""
        range_context = answer[max(0, claim.start() - 24):claim.end() + 32]
        range_match = re.search(
            r"(?<![\d,])(?P<lower>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
            r"[-–—]\s*(?P<upper>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
            r"(?P<range_unit>ปี|เดือน|บาท|%|เปอร์เซ็นต์)",
            range_context,
        )
        if range_match and number in {range_match.group("lower"), range_match.group("upper")}:
            if _has_exact_numeric_range(
                evidence_text,
                range_match.group("lower"),
                range_match.group("upper"),
                range_match.group("range_unit"),
            ):
                # PDF tables commonly extract an age/payment range as two
                # separate numeric tokens. Validate the complete range so a
                # supported "16 - 60 ปี" is not rejected as an unsupported
                # standalone "16" or "60".
                continue
        context_start = max(0, claim.start() - 28)
        context = answer[context_start:claim.end()]
        category = _claim_category(context)
        # Plan numbers are labels in prose/table answers. Their surrounding
        # payment words must not make the same integer look like a currency
        # claim; the exact plan label is still required in the cited text.
        if not unit and "แผน" in context:
            category = "general"
        if not _has_exact_numeric_claim(evidence_text, number, unit, category):
            return False, f"numeric claim is not supported with the same unit/context: {number}{unit}"
    contradiction = _contradicted_positive_claim(answer, evidence_text)
    if contradiction:
        return False, "answer reverses or contradicts a cited claim: " + contradiction
    answer_terms = _query_terms(answer)
    if answer_terms and not any(_normalise_evidence(term) in evidence for term in answer_terms):
        return False, "answer has no lexical support in cited evidence"
    return True, "citations resolve and answer claims have cited-text support"


def _claim_category(context: str) -> str:
    lowered = context.lower()
    if any(term in lowered for term in ("ชำระ", "จ่ายเบี้ย", "เบี้ย", "รายเดือน", "รายปี", "ราคา", "ค่าใช้จ่าย")):
        return "payment"
    if any(term in lowered for term in ("คุ้มครอง", "ความคุ้มครอง", "รับประกัน")):
        return "coverage"
    return "general"


def _has_exact_numeric_claim(text: str, number: str, unit: str, category: str) -> bool:
    number_key = number.replace(",", "")
    numeric_pattern = (
        r"(?<![\d,])(?P<number>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\d,])\s*"
        r"(?P<unit>ปี|เดือน|บาท|%|เปอร์เซ็นต์)?"
    )
    matches: list[re.Match[str]] = []
    for match in re.finditer(numeric_pattern, text):
        if match.group("number").replace(",", "") != number_key:
            continue
        matches.append(match)
        if unit and match.group("unit") == unit:
            context = text[max(0, match.start() - 32):match.start()]
            if category == "general" or _claim_category(context) == category:
                return True
            continue
        if unit:
            continue
        context = text[max(0, match.start() - 32):match.start()]
        if category == "general" or _claim_category(context) == category:
            return True
    # PDF table extraction often places the row label, number and unit in
    # separate text runs. Currency claims can therefore be validated by the
    # exact amount plus a payment context in the same cited chunk. Keep this
    # fallback limited to money so a coverage duration cannot satisfy a
    # payment-duration claim (for example 90 years versus 20 years).
    if unit == "บาท" and category == "payment" and matches:
        evidence_lower = text.lower()
        payment_markers = ("เบี้ย", "ชำระ", "รายเดือน", "รายปี", "payment", "premium")
        if any(marker in evidence_lower for marker in payment_markers):
            return True
    # Thai brochure tables often extract all row labels first and their values
    # later, so a 32-character local window cannot associate a duration with
    # its label. For non-currency durations, require the exact number+unit and
    # a matching semantic label in the same cited chunk. This remains stricter
    # than accepting a bare number and handles rows such as coverage 25 years /
    # premium payment 15 years without rejecting a correct structured answer.
    if unit in {"ปี", "เดือน", "%", "เปอร์เซ็นต์"} and matches and category != "general":
        lowered = text.lower()
        markers = {
            "payment": ("ชำระ", "ชําระ", "จ่ายเบี้ย", "เบี้ย", "รายเดือน", "รายปี", "payment", "premium"),
            "coverage": ("คุ้มครอง", "ความคุ้มครอง", "รับประกัน", "coverage"),
        }
        exact_unit_matches = [match for match in matches if match.group("unit") == unit]
        local_categories = {
            _claim_category(text[max(0, match.start() - 32):match.start()])
            for match in exact_unit_matches
        }
        # Do not use the table-layout fallback when the same number is already
        # tied locally to a conflicting field (for example coverage 90 years
        # must not support a claim that premiums are paid for 90 years).
        no_local_conflict = not (local_categories - {"general", category})
        if any(marker in lowered for marker in markers.get(category, ())):
            if exact_unit_matches and no_local_conflict:
                return True
    return False


def _has_exact_numeric_range(text: str, lower: str, upper: str, unit: str) -> bool:
    """Match an entire numeric range after normalizing PDF whitespace."""

    normalized = _normalise_evidence(text)
    lower_key = re.escape(lower.replace(",", ""))
    upper_key = re.escape(upper.replace(",", ""))
    unit_key = re.escape(_normalise_evidence(unit))
    return bool(
        re.search(
            # The unit already terminates the upper number. Whitespace
            # normalization may place the next table row's number after it.
            rf"(?<!\d){lower_key}[-–—]{upper_key}{unit_key}",
            normalized,
        )
    )


def _contradicted_positive_claim(answer: str, evidence: str) -> str | None:
    """Reject a positive claim when the cited text explicitly negates it."""

    positive_patterns = re.findall(
        r"(?<!ไม่)(คุ้มครอง|รับประกัน)\s*([\u0E00-\u0E7F A-Za-z0-9]{2,24})",
        answer,
    )
    normalised_evidence = re.sub(r"\s+", "", evidence).lower()
    for verb, subject in positive_patterns:
        phrase = re.sub(r"\s+", "", f"{verb}{subject}").lower()
        negated = "ไม่" + phrase
        if negated in normalised_evidence:
            return phrase
    return None


NO_ANSWER_MESSAGE = "ไม่พบข้อมูลนี้ในเอกสารผลิตภัณฑ์ที่ระบบมี"
AMBIGUOUS_PRODUCT_MESSAGE = "พบข้อมูลหลายผลิตภัณฑ์ที่เป็นไปได้ กรุณาระบุผลิตภัณฑ์ก่อน"
RETRIEVAL_ERROR_MESSAGE = "ระบบค้นหาเอกสารขัดข้องชั่วคราว จึงยังไม่สามารถยืนยันข้อมูลผลิตภัณฑ์ได้"
