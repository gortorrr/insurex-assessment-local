"""Optional persistent Chroma index with deterministic upsert/retrieval."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import math
from functools import wraps
from threading import Lock
from typing import Any

from insurex.rag.contracts import RetrievedChunk
from insurex.rag.ingestion import CHUNKING_REVISION, chunk_page_text, load_verified_pages
from insurex.rag.manifest import file_sha256, load_manifest, validate_manifest


class RagIndexDependencyError(RuntimeError):
    pass


def stale_chunk_ids(existing_ids: list[str], current_ids: set[str]) -> list[str]:
    """Return only IDs that no longer belong to the current corpus/config."""

    return [item_id for item_id in existing_ids if item_id not in current_ids]


def index_fingerprint(
    manifest_path: Path,
    *,
    root: Path,
    embedding_model: str,
    max_characters: int = 1800,
    overlap: int = 240,
    max_tokens: int | None = 400,
) -> tuple[str, str | None, dict[str, Any]]:
    """Return a reproducibility fingerprint for an ingestion configuration."""

    embedding_revision = None
    embedding_manifest_sha256 = None
    model_files_fingerprint = None
    extraction_verification_sha256 = None
    extraction_revision = None
    model_manifest = root / "knowledge_base" / "embedding_model_manifest.json"
    if model_manifest.exists():
        model_data = json.loads(model_manifest.read_text(encoding="utf-8"))
        embedding_revision = model_data.get("revision")
        embedding_manifest_sha256 = file_sha256(model_manifest)
        model_path = Path(embedding_model)
        if not model_path.is_absolute():
            # Runtime paths are relative to the repository working directory,
            # while manifest document paths are relative to the RAG root.
            model_path = model_path.resolve() if model_path.exists() else root / model_path
        expected_files = model_data.get("essential_files", [])
        actual_files: dict[str, str] = {}
        for item in expected_files:
            file_path = model_path / item["path"]
            if not file_path.exists() or file_sha256(file_path) != item["sha256"]:
                raise RuntimeError(f"embedding model file does not match manifest: {item['path']}")
            actual_files[item["path"]] = item["sha256"]
        model_files_fingerprint = hashlib.sha256(
            json.dumps(actual_files, sort_keys=True).encode("utf-8")
        ).hexdigest()
    extraction_verification = root / "knowledge_base" / "extraction_verification.json"
    if extraction_verification.exists():
        extraction_verification_sha256 = file_sha256(extraction_verification)
        extraction_revision = json.loads(extraction_verification.read_text(encoding="utf-8")).get("mapping_revision")
    config = {
        "manifest_sha256": file_sha256(manifest_path),
        "embedding_model": embedding_model,
        "embedding_revision": embedding_revision,
        "embedding_manifest_sha256": embedding_manifest_sha256,
        "model_files_fingerprint": model_files_fingerprint,
        "extraction_verification_sha256": extraction_verification_sha256,
        "extraction_revision": extraction_revision,
        "max_characters": max_characters,
        "overlap": overlap,
        "max_tokens": max_tokens,
        "chunking_revision": CHUNKING_REVISION,
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fingerprint, embedding_revision, config


def has_private_use_glyphs(text: str) -> bool:
    return any(0xE000 <= ord(char) <= 0xF8FF for char in text)


def find_product_matches(
    query: str,
    products: list[tuple[str, str]],
    product_aliases: dict[str, list[str]] | None = None,
) -> list[str]:
    """Return deterministic product candidates mentioned by the user.

    A full product name wins over a shorter shared phrase.  When the query
    contains only a meaningful part shared by multiple aliases, every tied
    product is returned so the conversation can ask the user to choose.
    """

    def normalise(value: str) -> str:
        return re.sub(r"[^a-z0-9\u0E00-\u0E7F]+", " ", value.casefold()).strip()

    clean_query = normalise(query)
    if not clean_query:
        return []
    product_aliases = product_aliases or {}
    aliases_by_product: dict[str, set[str]] = {}
    for product_id, product_name in products:
        aliases_by_product[product_id] = {
            alias
            for alias in {
                normalise(product_id.replace("-", " ")),
                normalise(product_name),
                *(normalise(alias) for alias in product_aliases.get(product_id, [])),
            }
            if alias
        }

    # Prefer the longest complete alias found in the query.  This makes
    # "คุ้มตลอดชีพ ซีไอ พลัส" resolve to one product even though the shorter
    # phrase "คุ้มตลอดชีพ" also belongs to another product.
    full_scores = {
        product_id: max((len(alias) for alias in aliases if alias in clean_query), default=0)
        for product_id, aliases in aliases_by_product.items()
    }
    best_full = max(full_scores.values(), default=0)
    if best_full:
        return sorted(product_id for product_id, score in full_scores.items() if score == best_full)

    # Thai particles are commonly attached without spaces (for example
    # "คุ้มตลอดชีพอะ"). Match substantial alias tokens as substrings, but
    # return all products tied on the most specific token instead of guessing.
    token_scores: dict[str, int] = {}
    for product_id, aliases in aliases_by_product.items():
        tokens = {
            token
            for alias in aliases
            for token in alias.split()
            if len(token) >= 4 and not token.isdigit()
        }
        token_scores[product_id] = max(
            (len(token) for token in tokens if token in clean_query),
            default=0,
        )
    best_token = max(token_scores.values(), default=0)
    if not best_token:
        return []
    return sorted(product_id for product_id, score in token_scores.items() if score == best_token)


def infer_product_id_from_query(
    query: str,
    products: list[tuple[str, str]],
    product_aliases: dict[str, list[str]] | None = None,
) -> str | None:
    """Infer a single product scope from an explicit product name in a query.

    This is query routing, not an answer or a default product choice. If no
    unique product alias is present, retrieval remains corpus-wide and the
    evidence gate can return an explicit ambiguity response.
    """

    matches = find_product_matches(query, products, product_aliases)
    return matches[0] if len(matches) == 1 else None


def _chunk_ids_hash(ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def index_metadata_path(chroma_path: Path) -> Path:
    return chroma_path / "index_metadata.json"


def load_index_metadata(chroma_path: Path) -> dict[str, Any]:
    path = index_metadata_path(chroma_path)
    if not path.exists():
        raise RuntimeError("index metadata is missing; rebuild the index before evaluation")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("collection_name") != "insurex_kb":
        raise RuntimeError("index metadata collection is incompatible")
    return data


def verify_index_metadata(
    chroma_path: Path,
    *,
    expected_fingerprint: str,
    expected_count: int | None = None,
) -> dict[str, Any]:
    metadata = load_index_metadata(chroma_path)
    collection = _client(chroma_path).get_collection("insurex_kb")
    actual_ids = collection.get().get("ids", [])
    actual_count = collection.count()
    actual_ids_hash = _chunk_ids_hash(actual_ids)
    if metadata.get("index_fingerprint") != expected_fingerprint:
        raise RuntimeError("index fingerprint does not match current manifest/model/chunk configuration")
    if metadata.get("chunk_ids_sha256") != actual_ids_hash or metadata.get("chunk_count") != actual_count:
        raise RuntimeError("index metadata does not match the actual Chroma collection")
    if expected_count is not None and actual_count != expected_count:
        raise RuntimeError("actual Chroma count does not match evaluation expectation")
    return {"metadata": metadata, "actual_count": actual_count, "actual_ids_sha256": actual_ids_hash}


class SentenceTransformerEmbeddings:
    """Small adapter for multilingual-e5-small without importing it offline."""

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RagIndexDependencyError("sentence-transformers is not installed") from exc
        self.model = SentenceTransformer(model_name, device=device)
        # Streamlit may run suggestion retrieval and a chat worker together.
        # Protect the shared model because encode is not guaranteed re-entrant.
        self._encode_lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = [f"passage: {text}" for text in texts]
        with self._encode_lock:
            return self.model.encode(inputs, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        with self._encode_lock:
            return self.model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()

    def passage_tokenizer(self) -> Any:
        """Return a tokenizer that counts the E5 passage prefix as input."""

        backend = getattr(self.model.tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return _PrefixedBackendTokenizer(backend, "passage: ")
        return _PrefixedCallableTokenizer(self.model.tokenizer, "passage: ")


class _PrefixedBackendTokenizer:
    def __init__(self, backend: Any, prefix: str):
        self.backend = backend
        self.prefix = prefix

    def __call__(self, text: str) -> list[int]:
        return self.backend.encode(self.prefix + text).ids


class _PrefixedCallableTokenizer:
    def __init__(self, tokenizer: Any, prefix: str):
        self.tokenizer = tokenizer
        self.prefix = prefix

    def __call__(self, text: str) -> Any:
        return self.tokenizer(self.prefix + text)


_EMBEDDING_CACHE_LOCK = Lock()
_EMBEDDING_CACHE: dict[tuple[str, str], SentenceTransformerEmbeddings] = {}
_CHROMA_CLIENT_LOCK = Lock()
_CHROMA_CLIENTS: dict[str, Any] = {}


def _cached_embeddings(model_name: str, device: str) -> SentenceTransformerEmbeddings:
    # functools.lru_cache can execute duplicate cache misses concurrently.
    # Loading several E5 instances at first paint wastes memory and leaves the
    # caller exposed before the per-model encode lock exists.
    key = (model_name, device)
    with _EMBEDDING_CACHE_LOCK:
        if key not in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE[key] = SentenceTransformerEmbeddings(model_name, device)
        return _EMBEDDING_CACHE[key]


def _client(path: Path) -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RagIndexDependencyError("chromadb is not installed") from exc
    resolved = str(path.resolve())
    path.mkdir(parents=True, exist_ok=True)
    # Chroma's embedded Rust client uses a process-global system registry. Two
    # simultaneous PersistentClient constructors for one path can race during
    # tenant initialization, so construct once and share the thread-safe client.
    with _CHROMA_CLIENT_LOCK:
        if resolved not in _CHROMA_CLIENTS:
            _CHROMA_CLIENTS[resolved] = chromadb.PersistentClient(path=resolved)
        return _CHROMA_CLIENTS[resolved]


def _index_write_guard(function):
    @wraps(function)
    def run(*args, **kwargs):
        path = Path(kwargs['chroma_path'])
        path.mkdir(parents=True, exist_ok=True)
        lock = path / 'ingestion.lock'
        handle = lock.open('x')
        handle.close()
        dirty = path / 'ingestion.incomplete'
        dirty.write_text('ingestion in progress; retry ingestion to recover', encoding='utf-8')
        try:
            result = function(*args, **kwargs)
            dirty.unlink()
            return result
        finally:
            lock.unlink()
    return run


def verify_corpus_current(chroma_path, manifest_path):
    """Fail closed when documents change or an ingestion is incomplete."""
    if any((Path(chroma_path) / marker).exists() for marker in ('ingestion.lock','ingestion.incomplete')):
        raise RuntimeError('knowledge base is being updated')
    metadata = load_index_metadata(Path(chroma_path))
    if metadata.get('manifest_sha256') != file_sha256(Path(manifest_path)):
        raise RuntimeError('knowledge base manifest changed; run ingestion')
    root = Path(manifest_path).resolve().parents[1]
    files = metadata.get('corpus_files')
    if not files:
        raise RuntimeError('index lacks corpus revision metadata; run ingestion')
    for relative, expected in files.items():
        if file_sha256(root / relative) != expected:
            raise RuntimeError('knowledge base document changed; run ingestion')


@_index_write_guard
def ingest_manifest_to_chroma(
    manifest_path: Path,
    *,
    root: Path,
    chroma_path: Path,
    embedding_model: str,
    embedding_device: str = "cpu",
    max_characters: int = 1800,
    overlap: int = 240,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Validate five PDFs, replace stale chunks, and idempotently upsert them."""

    manifest = load_manifest(manifest_path)
    errors = validate_manifest(manifest, root=root)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors))
    embeddings = _cached_embeddings(embedding_model, embedding_device)
    collection = _client(chroma_path).get_or_create_collection("insurex_kb")
    total_pages = 0
    chunks: list[dict[str, Any]] = []
    tokenizer = embeddings.passage_tokenizer() if max_tokens else None
    for document in manifest["documents"]:
        corrected_path = document.get("corrected_text_path")
        if not corrected_path:
            raise ValueError(f"active document has no corrected extraction artifact: {document['doc_id']}")
        pages = load_verified_pages(
            root / corrected_path,
            doc_id=document["doc_id"],
            expected_pdf_sha256=document["sha256"],
        )
        total_pages += len(pages)
        for page in pages:
            for chunk in chunk_page_text(
                page,
                max_characters=max_characters,
                overlap=overlap,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            ):
                chunks.append({
                    **chunk,
                    "product_id": document["product_id"],
                    "product_name": document["product_name"],
                    "source_url": document["pdf_url"],
                    "source_page_url": document["source_page_url"],
                })
    texts = [chunk["text"] for chunk in chunks]
    stale_chunks_removed = 0
    current_ids = {str(chunk["chunk_id"]) for chunk in chunks}
    existing = collection.get()
    stale_ids = stale_chunk_ids(existing.get("ids", []), current_ids)
    if stale_ids:
        collection.delete(ids=stale_ids)
        stale_chunks_removed = len(stale_ids)
    if texts:
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in chunks],
            documents=texts,
            embeddings=embeddings.embed_documents(texts),
            metadatas=[
                {
                    "doc_id": chunk["doc_id"],
                    "product_id": chunk["product_id"],
                    "product_name": chunk["product_name"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "source_url": chunk["source_url"],
                    "source_page_url": chunk["source_page_url"],
                    "content_hash": chunk["content_hash"],
                    "text_quality": "unverified_private_use" if has_private_use_glyphs(chunk["text"]) else "verified",
                }
                for chunk in chunks
            ],
        )
    fingerprint, embedding_revision, config = index_fingerprint(
        manifest_path,
        root=root,
        embedding_model=embedding_model,
        max_characters=max_characters,
        overlap=overlap,
        max_tokens=max_tokens,
    )
    actual_ids = collection.get().get("ids", [])
    index_metadata_path(chroma_path).write_text(
        json.dumps(
            {
                "collection_name": "insurex_kb",
                "chunk_count": collection.count(),
                "chunk_ids_sha256": _chunk_ids_hash(actual_ids),
                "index_fingerprint": fingerprint,
                "manifest_sha256": config["manifest_sha256"],
                "embedding_revision": embedding_revision,
                "embedding_manifest_sha256": config["embedding_manifest_sha256"],
                "model_files_fingerprint": config["model_files_fingerprint"],
                "extraction_verification_sha256": config["extraction_verification_sha256"],
                "extraction_revision": config["extraction_revision"],
                "chunk_config": {key: config[key] for key in ("max_characters", "overlap", "max_tokens", "chunking_revision")},
                "corpus_files": {item[key]: file_sha256(root / item[key])
                                 for item in manifest['documents'] for key in ('local_path','corrected_text_path')},
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return {
        "documents": len(manifest["documents"]), "pages": total_pages, "chunks": len(chunks),
        "stale_chunks_removed": stale_chunks_removed, "index_fingerprint": fingerprint,
        "manifest_sha256": config["manifest_sha256"],
        "embedding_revision": embedding_revision,
        "embedding_manifest_sha256": config["embedding_manifest_sha256"],
        "model_files_fingerprint": config["model_files_fingerprint"],
        "chunk_config": {key: config[key] for key in ("max_characters", "overlap", "max_tokens", "chunking_revision")},
    }


def retrieve_from_chroma(
    query: str,
    *,
    chroma_path: Path,
    embedding_model: str,
    embedding_device: str = "cpu",
    product_id: str | None = None,
    product_aliases: dict[str, list[str]] | None = None,
    top_k: int = 5,
    retrieval_mode: str = "semantic",
) -> list[RetrievedChunk]:
    embeddings = _cached_embeddings(embedding_model, embedding_device)
    collection = _client(chroma_path).get_collection("insurex_kb")
    if retrieval_mode not in {"semantic", "hybrid"}:
        raise ValueError("retrieval_mode must be semantic or hybrid")
    kwargs: dict[str, Any] = {"query_embeddings": [embeddings.embed_query(query)], "n_results": min(collection.count(), top_k * 3 if retrieval_mode == "hybrid" else top_k)}
    if product_id:
        kwargs["where"] = {"product_id": product_id}
    else:
        metadata_rows = collection.get(include=["metadatas"]).get("metadatas", [])
        products = sorted({(str(row.get("product_id", "")), str(row.get("product_name", ""))) for row in metadata_rows})
        inferred = infer_product_id_from_query(query, products, product_aliases)
        if inferred:
            kwargs["where"] = {"product_id": inferred}
    result = collection.query(**kwargs)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    semantic = [
        RetrievedChunk(
            doc_id=metadata["doc_id"], product_id=metadata["product_id"],
            product_name=metadata["product_name"], page_number=int(metadata["page_number"]),
            chunk_id=metadata["chunk_id"], text=text, source_url=metadata["source_url"],
            score=1.0 - float(distance),
            text_quality=metadata.get("text_quality", "unknown"),
        )
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
    if retrieval_mode == "semantic":
        return semantic
    data = collection.get(where=kwargs.get("where"), include=["documents", "metadatas"])
    candidates = [RetrievedChunk(doc_id=m['doc_id'], product_id=m['product_id'], product_name=m['product_name'],
                                page_number=int(m['page_number']), chunk_id=m['chunk_id'], text=t,
                                source_url=m['source_url'], score=0.0, text_quality=m.get('text_quality','unknown'))
                  for t,m in zip(data['documents'], data['metadatas'])]
    return hybrid_rank(query, semantic, candidates, top_k)


def hybrid_rank(query, semantic, candidates, top_k=5):
    """BM25 over Thai character fragments + English words, fused with semantic ranks."""
    from insurex.rag.contracts import _normalise_evidence, _query_terms
    terms = _query_terms(query)
    texts = [_normalise_evidence(c.product_name+' '+c.text) for c in candidates]
    average = sum(map(len, texts)) / max(len(texts), 1)
    scores = []
    for chunk, text in zip(candidates, texts):
        score = 0.0
        for term in terms:
            frequency = text.count(term)
            if not frequency:
                continue
            df = sum(term in document for document in texts)
            idf = math.log(1+(len(texts)-df+0.5)/(df+0.5))
            score += idf * frequency * 2.2 / (frequency+1.2*(0.25+0.75*len(text)/max(average,1)))
        if score > 0:
            scores.append((score,chunk))
    lexical = [c for _,c in sorted(scores,key=lambda item:(-item[0],item[1].chunk_id))]
    fused = {}
    by_id = {c.chunk_id:c for c in candidates}
    by_id.update({c.chunk_id:c for c in semantic})
    for weight, ranked in ((0.7,semantic),(0.3,lexical)):
        for rank,chunk in enumerate(ranked,1):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id,0)+weight/(60+rank)
    return [by_id[key] for key in sorted(fused,key=lambda key:(-fused[key],key))[:top_k]]
