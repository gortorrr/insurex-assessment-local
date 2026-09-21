"""Lazy LLM provider factory for Gemini local and Azure deployment profiles."""

from __future__ import annotations

from typing import Any


class LlmDependencyError(RuntimeError):
    """Raised when the selected provider SDK is not installed."""


def build_chat_model(settings: Any) -> Any:
    """Build a LangChain chat model without importing provider SDKs offline."""

    if settings.llm_provider == "google_ai_studio":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise LlmDependencyError(
                "Install the local LLM dependencies from requirements.local-llm.txt"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=settings.google_api_key,
            temperature=0,
            max_tokens=settings.llm_max_output_tokens,
            request_timeout=settings.llm_timeout_seconds,
            retries=settings.max_retrieval_retries,
        )

    if settings.llm_provider == "azure_openai":
        if settings.azure_openai_auth_mode != "api_key":
            raise LlmDependencyError(
                "Azure Entra ID model construction is reserved for the managed-identity deployment profile"
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LlmDependencyError(
                "Install the Azure LLM dependencies from requirements.azure.txt"
            ) from exc
        base_url = settings.azure_openai_endpoint.rstrip("/") + "/openai/v1/"
        return ChatOpenAI(
            model=settings.llm_model,
            base_url=base_url,
            api_key=settings.azure_openai_api_key,
            temperature=0,
            max_tokens=settings.llm_max_output_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.max_retrieval_retries,
        )

    if settings.llm_provider == "none":
        raise LlmDependencyError("LLM_PROVIDER=none does not build a live chat model")
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
