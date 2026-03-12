from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from agents.agent_find_company.common import MAX_COMPANY_CANDIDATES
from agents.agent_find_company.input import FindCompanySearchInput
from agents.agent_find_company.service import run_search


@tool(args_schema=FindCompanySearchInput, response_format="content_and_artifact")
def find_company(
    query: str = "",
    invest_level: str | None = None,
    employees_min: int | None = None,
    employees_max: int | None = None,
    categories: list[str] | None = None,
    limit: int = MAX_COMPANY_CANDIDATES,
) -> tuple[str, dict[str, Any]]:
    """회사 검색 도구. 자유 텍스트와 구조화된 조건을 함께 사용해 후보를 조회합니다."""
    payload = run_search(
        FindCompanySearchInput(
            query=query,
            invest_level=invest_level,
            employees_min=employees_min,
            employees_max=employees_max,
            categories=categories,
            limit=limit,
        )
    )
    return payload["summary"], payload
