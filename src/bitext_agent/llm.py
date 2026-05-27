"""Nebius OpenAI-compatible model factory and best-effort usage helpers."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

from langchain_openai import ChatOpenAI

from bitext_agent.config import Settings
from bitext_agent.settings_store import SettingsStore


def configure_langsmith(settings: Settings) -> None:
    """Set LangSmith environment variables when optional tracing is enabled."""

    os.environ["LANGSMITH_TRACING"] = "true" if settings.langsmith_tracing else "false"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project


def build_chat_model(settings: Settings, model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    """Build a ChatOpenAI client pointed at Nebius Token Factory."""

    return ChatOpenAI(
        model=model_name,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=temperature,
    )


def invoke_with_usage_log(
    store: SettingsStore,
    model_name: str,
    session_id: str,
    user_uuid: str,
    call: Callable[[], Any],
) -> Any:
    """Run an LLM call and record whatever usage metadata LangChain returns."""

    start = time.perf_counter()
    try:
        result = call()
    except Exception:
        store.log_usage(
            model=model_name,
            status="error",
            session_id=session_id,
            user_uuid=user_uuid,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        raise
    usage = getattr(result, "usage_metadata", None) or getattr(result, "response_metadata", {}).get(
        "token_usage", {}
    )
    prompt_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
    completion_tokens = usage.get("output_tokens") or usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    store.log_usage(
        model=model_name,
        status="ok",
        session_id=session_id,
        user_uuid=user_uuid,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
    return result

