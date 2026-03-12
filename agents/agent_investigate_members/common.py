from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_RESULTS_PER_QUERY_FAMILY = 2
MAX_TOTAL_SIGNALS = 12
MAX_KEY_MEMBERS = 5

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
    "leadership_team": ("leadership", "core team", "경영진", "리더십", "핵심팀", "임원"),
    "executive_roles": ("CTO", "COO", "CPO", "Head", "총괄", "이사", "연구소장", "팀장"),
    "robotics_expertise": ("robotics", "ROS", "computer vision", "autonomy", "AI"),
    "deployment_business": ("deployment", "product", "business", "operations", "사업", "영업", "운영"),
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
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=OPENAI_MODEL,
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
        max_results=MAX_RESULTS_PER_QUERY_FAMILY,
        search_depth="basic",
    )
