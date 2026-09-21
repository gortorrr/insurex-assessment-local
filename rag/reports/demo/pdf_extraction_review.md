# InsureX PDF extraction review

Review date: 2026-09-18. The active corpus contains exactly five InsureX/SCB Protect product PDFs from the official InsureX product sources. The PDFs may contain FWD branding and underwriting references inside the official documents; these references are retained as document content. No unrelated FWD PDF was added.

## Verification method

- Raw extraction is preserved per page with `pypdf` in `knowledge_base/extracted/raw/`.
- The embedded `SukhumvitSet` Type0 fonts expose `ToUnicode` CMaps whose private-use targets are present in the raw text. The deterministic mapping is recorded as revision `sukhumvitset-pua-thai-v1` in `scripts/repair_pdf_extraction.py`.
- All 14 active-corpus pages were rendered with Poppler `pdftoppm` and reviewed against the visible Thai headings, labels, tables, repeated words, and numeric terms. Render paths and SHA-256 values are recorded in `knowledge_base/extraction_verification.json`.
- Corrected text is stored separately in `knowledge_base/extracted/corrected/`. No glyph was discarded, no OCR was used, and no LLM was used to fill text. Only corrected pages with matching PDF provenance, page hashes, mapping revision, and no remaining private-use characters are accepted by ingestion.

## Corpus decision

| Document | Pages | Decision |
|---|---:|---|
| Khum Talodcheep CI Plus | 3 | Corrected text accepted after CMap and rendered-page verification |
| Khum Talodcheep Plus | 4 | Corrected text accepted after CMap and rendered-page verification |
| Khum Aomsook 25/15 | 2 | Corrected text accepted after CMap and rendered-page verification |
| Khum Manjai Sure Term | 3 | Corrected text accepted after CMap and rendered-page verification |
| Whole Life 90/20 | 2 | Corrected text accepted after CMap and rendered-page verification |

The raw artifacts remain an audit record. Pages with unknown mappings, failed provenance, failed hashes, remaining private-use characters, or failed visual review would remain quarantined; none of the five current documents has such a page after this verification run.

## Rebuilt index evidence

The corrected corpus was ingested into persistent Chroma without reset. The first corrected rebuild removed 13 stale chunks from the previous 44-chunk index. A repeat ingest then reported 31 chunks before and after ingest, with `stale_chunks_removed=0` and `idempotent_count=true`.

- Documents/pages: 5 / 14
- Current chunks: 31
- Index fingerprint: `6670aef22ef3eff2ab9898f3f68789ebc01e1ed0cf9b14c60fe7766c5e6c8c9f`
- Manifest SHA-256: `b8661f89e50add2a241f47480d6cfe85ea3b2953e4637f743a07edf9535b28d7`
- Extraction verification SHA-256: `6a06aac2bddf17845456f750f1506a298ab83f5aef4be654173bcaf75929fe4c`
- Embedding revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Model files fingerprint: `520a82130d7a3f3b0a0c73746e95964224076263fefe6abdbe1a44c0a4bf2369`
- Chunk configuration: 1,800 characters, 240 overlap, 400-token ceiling

The lower chunk count is a consequence of corrected Thai combining marks producing a different token/character boundary from the old PUA text. It is not evidence that pages were dropped; the index metadata and manifest both report 5 documents and 14 pages.

## Evaluation status

Offline retrieval/grounding evaluation was rerun against this fingerprint with zero live LLM calls. The held-out-v2 set passed 20/20 and the legacy 30-case set passed 30/30. The evidence gate now checks lexical coverage within a coherent chunk and only permits a high-similarity single-product fallback when there is also a lexical trace; generic cross-domain terms cannot turn an unrelated vehicle or banking question into sufficient evidence. Ambiguity remains a separate status and is never counted as no-answer.

The local end-to-end evidence run now produces an `offline_evidence_extract` answer with a PDF/page citation and an `offline_no_answer_gate` result with no citation. These are extract/gating evidence only, not generated answers. Gemini and Azure LLM were not called in this review.
