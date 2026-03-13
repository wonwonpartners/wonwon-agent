from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from utils.openai_fallback import (
    build_chat_model,
    get_fallback_openai_model_name,
    get_openai_model_name,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)

OPENAI_MODEL = get_openai_model_name("gpt-4o-mini")
FALLBACK_OPENAI_MODEL = get_fallback_openai_model_name("gpt-4.1-nano")
NEWS_MAX_RESULTS = 2
WEB_MAX_RESULTS = 2
NEWS_DAYS = 180

KOREAN_NEWS_DOMAINS = [
    "yna.co.kr",
    "newsis.com",
    "mk.co.kr",
    "zdnet.co.kr",
]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value and name == "OPENAI_API_KEY":
        value = os.getenv("OPEN_AI_KEY")
    if not value and name == "TAVILY_API_KEY":
        value = os.getenv("TRAVILY_API_KEY")
    if value:
        return value
    raise RuntimeError(f"{name} 환경변수가 필요합니다.")


@lru_cache(maxsize=1)
def get_chat_model():
    api_key = require_env("OPENAI_API_KEY")
    return build_chat_model(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def get_fallback_chat_model():
    api_key = require_env("OPENAI_API_KEY")
    return build_chat_model(
        model=FALLBACK_OPENAI_MODEL,
        temperature=0,
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def get_news_search_tool():
    api_key = require_env("TAVILY_API_KEY")
    from langchain_tavily import TavilySearch

    return TavilySearch(
        tavily_api_key=api_key,
        topic="news",
        max_results=NEWS_MAX_RESULTS,
        search_depth="basic",
    )


@lru_cache(maxsize=1)
def get_web_search_tool():
    api_key = require_env("TAVILY_API_KEY")
    from langchain_tavily import TavilySearch

    return TavilySearch(
        tavily_api_key=api_key,
        topic="general",
        max_results=WEB_MAX_RESULTS,
        search_depth="basic",
    )
