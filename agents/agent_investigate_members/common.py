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
MAX_RESULTS_PER_QUERY_FAMILY = 2
MAX_TOTAL_SIGNALS = 12
MAX_KEY_MEMBERS = 5
MAX_SEARCH_RESULTS_PER_QUERY = 5

ROLE_TAXONOMY = (
    "robot_hw",
    "robot_sw_ai",
    "control_perception",
    "system_integration",
    "productization_deployment",
    "manufacturing_operations",
    "business_development",
)

QUERY_FAMILY_TERMS: dict[str, tuple[str, ...]] = {
    "ceo_founder": ("CEO", "대표", "창업자"),
    "leadership_team": ("leadership", "team", "경영진", "리더십", "핵심팀", "임원"),
    "executive_roles": ("CTO", "COO", "CPO", "Head", "총괄", "이사", "연구소장", "팀장"),
    "robotics_expertise": ("robotics", "AI", "연구개발", "로봇", "기술 리더"),
    "deployment_business": ("deployment", "product", "operations", "사업", "운영", "제품화"),
}


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
def get_web_search_tool():
    api_key = require_env("TAVILY_API_KEY")
    from langchain_tavily import TavilySearch

    return TavilySearch(
        tavily_api_key=api_key,
        topic="general",
        max_results=MAX_SEARCH_RESULTS_PER_QUERY,
        search_depth="advanced",
        include_raw_content="text",
    )
