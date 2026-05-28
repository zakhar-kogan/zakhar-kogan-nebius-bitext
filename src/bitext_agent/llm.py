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
    usage = _extract_usage_metadata(result)
    prompt_tokens = _int_or_none(usage.get("input_tokens") or usage.get("prompt_tokens"))
    completion_tokens = _int_or_none(usage.get("output_tokens") or usage.get("completion_tokens"))
    total_tokens = _int_or_none(usage.get("total_tokens"))
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    store.log_usage(
        model=model_name,
        status="ok",
        session_id=session_id,
        user_uuid=user_uuid,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=int((time.perf_counter() - start) * 1000),
        raw_usage_metadata=usage or None,
    )
    return result


def _extract_usage_metadata(result: Any) -> dict[str, Any]:
    """Extract LangChain or OpenAI-compatible token usage metadata."""

    candidates: list[Any] = [
        getattr(result, "usage_metadata", None),
        getattr(result, "response_metadata", None),
    ]
    if isinstance(result, dict):
        candidates.extend([result.get("usage_metadata"), result.get("response_metadata"), result])
    for candidate in candidates:
        if not candidate:
            continue
        data = _as_dict(candidate)
        for key in ("token_usage", "usage"):
            nested = _as_dict(data.get(key))
            if nested:
                return nested
        if any(key in data for key in ("input_tokens", "prompt_tokens", "output_tokens", "completion_tokens", "total_tokens")):
            return data
    return {}


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
