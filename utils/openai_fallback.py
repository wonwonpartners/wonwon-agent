from __future__ import annotations

import logging
import os
from typing import Any, Callable, TypeVar

try:
    from openai import RateLimitError
except Exception:  # pragma: no cover - defensive import for test environments
    RateLimitError = None

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_FALLBACK_OPENAI_MODEL = "gpt-4.1-nano"

T = TypeVar("T")


def get_openai_model_name(default: str = DEFAULT_OPENAI_MODEL) -> str:
    value = os.getenv("OPENAI_MODEL", default).strip()
    return value or default


def get_fallback_openai_model_name(
    default: str = DEFAULT_FALLBACK_OPENAI_MODEL,
) -> str:
    primary_model = get_openai_model_name()
    configured = os.getenv("OPENAI_FALLBACK_MODEL", default).strip()
    if configured and configured != primary_model:
        return configured
    if default != primary_model:
        return default
    return primary_model


def build_chat_model(
    *,
    model: str,
    temperature: float = 0,
    api_key: str | None = None,
):
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
    }
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def is_rate_limit_error(exc: Exception) -> bool:
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        return True
    haystack = " ".join(
        [
            exc.__class__.__name__,
            str(exc),
        ]
    ).lower()
    return any(
        token in haystack
        for token in (
            "ratelimit",
            "rate limit",
            "429",
            "too many requests",
        )
    )


def invoke_with_rate_limit_fallback(
    *,
    payload: Any,
    primary_factory: Callable[[], Any],
    fallback_factory: Callable[[], Any],
    logger: logging.Logger,
    operation_name: str,
) -> T:
    primary_runnable = primary_factory()
    try:
        return primary_runnable.invoke(payload)
    except Exception as exc:
        if not is_rate_limit_error(exc):
            raise
        fallback_runnable = fallback_factory()
        logger.warning(
            "[%s] rate limit detected on primary model; retrying with fallback model",
            operation_name,
        )
        return fallback_runnable.invoke(payload)


async def ainvoke_with_rate_limit_fallback(
    *,
    payload: Any,
    primary_factory: Callable[[], Any],
    fallback_factory: Callable[[], Any],
    logger: logging.Logger,
    operation_name: str,
) -> T:
    primary_runnable = primary_factory()
    try:
        return await primary_runnable.ainvoke(payload)
    except Exception as exc:
        if not is_rate_limit_error(exc):
            raise
        fallback_runnable = fallback_factory()
        logger.warning(
            "[%s] rate limit detected on primary model; retrying with fallback model",
            operation_name,
        )
        return await fallback_runnable.ainvoke(payload)
