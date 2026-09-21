"""Create a review-ready assessment folder without secrets or user runtime data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = Path(r"E:\insurex-assessment-submission-2026-09-21")
TOP_LEVEL_FOLDERS = (
    "analysis",
    "data_modeling",
    "rag",
    "insurex",
    "scripts",
    "tests",
    "docs",
    "assessment",
)
TOP_LEVEL_FILES = (
    "README.md",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "insurex_staff_ai_data_science_test_case.md",
)
EXCLUDED_PARTS = {
    "__pycache__",
    ".pbi",
    ".cache",
    ".ipynb_checkpoints",
    "traces",
}


def selected_files() -> list[Path]:
    files = [ROOT / name for name in TOP_LEVEL_FILES]
    files.append(ROOT / ".streamlit" / "config.toml")
    for folder in TOP_LEVEL_FOLDERS:
        base = ROOT / folder
        if base.exists():
            files.extend(base.rglob("*"))

    selected: list[Path] = []
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if set(relative.parts) & EXCLUDED_PARTS:
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        if path.name == "build_submission.py":
            continue
        if path.suffix in {".pyc", ".db", ".log", ".abf"} or ".sqlite-" in path.name:
            continue
        if path.suffix == ".sqlite":
            allowed = (
                relative.as_posix() == "data_modeling/db/analytics.sqlite"
                or "chroma_db" in relative.parts
            )
            if not allowed:
                continue
        selected.append(path)
    return sorted(set(selected))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_target(target: Path) -> Path:
    resolved = target.expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents or resolved.parent == resolved:
        raise ValueError("submission target must be a separate non-root directory")
    if len(resolved.parts) < 2:
        raise ValueError("submission target is too broad")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    target = validate_target(args.target)
    if target.exists():
        if not args.replace:
            parser.error(f"target already exists: {target}; pass --replace to rebuild it")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    copied: list[dict[str, object]] = []
    for source in selected_files():
        relative = source.relative_to(ROOT)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({
            "path": relative.as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        })

    names = {item["path"] for item in copied}
    required = {
        "README.md",
        "requirements.txt",
        ".env.example",
        "analysis/notebooks/insurex_analysis.ipynb",
        "analysis/powerbi/InsureX_Analysis.pbip",
        "data_modeling/db/analytics.sqlite",
        "rag/ui/app.py",
        "rag/chroma_db/chroma.sqlite3",
        "rag/models/multilingual-e5-small/model.safetensors",
        "rag/knowledge_base/manifest.json",
        "rag/reports/demo/assessment_requirements_current.json",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError("submission folder is missing required files: " + ", ".join(missing))
    forbidden = [
        name for name in names
        if name in {".env", "rag/db/assistant.sqlite", "rag/db/checkpoints.sqlite"}
        or name.startswith("rag/traces/")
    ]
    if forbidden:
        raise RuntimeError("submission folder contains private runtime files: " + ", ".join(forbidden))

    manifest = {
        "format": "directory",
        "source_root": str(ROOT),
        "submission_root": str(target),
        "files": len(copied),
        "bytes": sum(int(item["bytes"]) for item in copied),
        "includes_local_embedding_model": True,
        "includes_chroma_index": True,
        "includes_analytics_demo_sqlite": True,
        "excludes_user_runtime_databases": True,
        "entries": copied,
    }
    (target / "SUBMISSION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
