"""Build or refresh the local five-PDF Chroma index."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insurex.config import Settings
from insurex.rag.index import ingest_manifest_to_chroma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="remove the local Chroma directory first")
    args = parser.parse_args()
    settings = Settings.from_env()
    model = Path(settings.embedding_model)
    if not model.exists():
        model = Path("rag/models/multilingual-e5-small")
    if args.reset and settings.chroma_path.exists():
        shutil.rmtree(settings.chroma_path)
    before = 0
    if settings.chroma_path.exists():
        try:
            before = chromadb.PersistentClient(path=str(settings.chroma_path)).get_collection("insurex_kb").count()
        except Exception:
            before = 0
    result = ingest_manifest_to_chroma(
        settings.kb_manifest_path,
        root=Path("rag"),
        chroma_path=settings.chroma_path,
        embedding_model=str(model),
        embedding_device=settings.embedding_device,
        max_characters=1800,
        overlap=240,
        # E5's model limit is 512 tokens. Reserve room for the passage prefix
        # and special tokens instead of allowing a boundary overflow.
        max_tokens=400,
    )
    after = chromadb.PersistentClient(path=str(settings.chroma_path)).get_collection("insurex_kb").count()
    print({"before_count": before, **result, "after_count": after, "idempotent_count": before == after if not args.reset else None})


if __name__ == "__main__":
    main()
