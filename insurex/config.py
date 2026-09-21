"""Configuration contract for the local Gemini and Azure deployment profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_env: str = "local"
    log_level: str = "INFO"
    database_backend: str = "local"
    database_url: str = ""
    llm_provider: str = "none"
    llm_model: str = ""
    google_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = ""
    azure_openai_auth_mode: str = "api_key"
    llm_timeout_seconds: int = 30
    llm_max_output_tokens: int = 1024
    max_retrieval_retries: int = 1
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_device: str = "cpu"
    chroma_path: Path = Path("rag/chroma_db")
    # `business_db_path` remains as a compatibility alias for older scripts.
    # New code uses the physically separated assistant/analytics databases.
    business_db_path: Path = Path("rag/db/assistant.sqlite")
    assistant_db_path: Path | None = None
    analytics_db_path: Path = Path("data_modeling/db/analytics.sqlite")
    checkpoint_db_path: Path = Path("rag/db/checkpoints.sqlite")
    kb_manifest_path: Path = Path("rag/knowledge_base/manifest.json")
    retrieval_mode: str = "hybrid"
    trace_path: Path = Path("rag/traces")

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            app_env=_setting("APP_ENV", "local"),
            log_level=_setting("LOG_LEVEL", "INFO"),
            database_backend=_setting("DATABASE_BACKEND", "local"),
            database_url=_setting("DATABASE_URL", ""),
            llm_provider=_setting("LLM_PROVIDER", "none"),
            llm_model=_setting("LLM_MODEL", ""),
            google_api_key=_setting("GOOGLE_API_KEY", ""),
            azure_openai_endpoint=_setting("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_api_key=_setting("AZURE_OPENAI_API_KEY", ""),
            azure_openai_api_version=_setting("AZURE_OPENAI_API_VERSION", ""),
            azure_openai_auth_mode=_setting("AZURE_OPENAI_AUTH_MODE", "api_key"),
            llm_timeout_seconds=int(_setting("LLM_TIMEOUT_SECONDS", "30")),
            llm_max_output_tokens=int(_setting("LLM_MAX_OUTPUT_TOKENS", "1024")),
            max_retrieval_retries=int(_setting("MAX_RETRIEVAL_RETRIES", "1")),
            embedding_model=_setting(
                "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
            ),
            embedding_device=_setting("EMBEDDING_DEVICE", "cpu"),
            chroma_path=Path(_setting("CHROMA_PATH", "rag/chroma_db")),
            business_db_path=Path(
                _setting("ASSISTANT_DB_PATH", _setting("BUSINESS_DB_PATH", "rag/db/assistant.sqlite"))
            ),
            assistant_db_path=Path(
                _setting("ASSISTANT_DB_PATH", _setting("BUSINESS_DB_PATH", "rag/db/assistant.sqlite"))
            ),
            analytics_db_path=Path(_setting("ANALYTICS_DB_PATH", "data_modeling/db/analytics.sqlite")),
            checkpoint_db_path=Path(
                _setting("CHECKPOINT_DB_PATH", "rag/db/checkpoints.sqlite")
            ),
            kb_manifest_path=Path(
                _setting("KB_MANIFEST_PATH", "rag/knowledge_base/manifest.json")
            ),
            retrieval_mode=_setting("RETRIEVAL_MODE", "hybrid"),
            trace_path=Path(_setting("TRACE_PATH", "rag/traces")),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.database_backend not in {"local", "postgresql"}:
            errors.append("DATABASE_BACKEND must be local or postgresql")
        if self.database_backend == "postgresql" and not self.database_url:
            errors.append("DATABASE_URL is required for the postgresql backend")
        if self.llm_timeout_seconds <= 0:
            errors.append("LLM_TIMEOUT_SECONDS must be positive")
        if self.llm_max_output_tokens <= 0:
            errors.append("LLM_MAX_OUTPUT_TOKENS must be positive")
        if self.max_retrieval_retries < 0:
            errors.append("MAX_RETRIEVAL_RETRIES must be zero or positive")
        if self.llm_provider != "none" and not self.llm_model:
            errors.append("LLM_MODEL is required when LLM_PROVIDER is enabled")
        if self.llm_provider not in {"none", "google_ai_studio", "azure_openai"}:
            errors.append("LLM_PROVIDER must be none, google_ai_studio, or azure_openai")
        if self.llm_provider == "google_ai_studio" and not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required for google_ai_studio")
        if self.llm_provider == "azure_openai":
            if not self.azure_openai_endpoint:
                errors.append("AZURE_OPENAI_ENDPOINT is required for azure_openai")
            if self.azure_openai_auth_mode not in {"api_key", "entra_id"}:
                errors.append("AZURE_OPENAI_AUTH_MODE must be api_key or entra_id")
            if self.azure_openai_auth_mode == "api_key" and not self.azure_openai_api_key:
                errors.append("AZURE_OPENAI_API_KEY is required for api_key auth")
        return errors


def _load_dotenv() -> None:
    """Load a simple local .env without making python-dotenv mandatory."""

    env_path = Path(os.getenv("INSUREX_ENV_FILE", ".env"))
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _setting(name: str, default: str = "") -> str:
    """Read process environment first, then Streamlit Cloud secrets.

    The import is lazy so core scripts and tests do not require Streamlit just
    to load configuration. Secrets are never logged or included in diagnostics.
    """

    value = os.getenv(name)
    if value not in (None, ""):
        return value
    try:
        import streamlit as st

        secret_value = st.secrets.get(name, default)
        return str(secret_value) if secret_value is not None else default
    except Exception:
        return default
