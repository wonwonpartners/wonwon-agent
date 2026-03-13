from __future__ import annotations

import json
import logging
from typing import Any

from agents.agent_find_company.common import (
    MAX_COMPANY_CANDIDATES,
    get_chat_model,
    get_fallback_chat_model,
)
from agents.agent_find_company.input import FindCompanySearchInput
from agents.agent_find_company.prompts import (
    get_search_system_prompt,
    get_selection_system_prompt,
    render_search_user_prompt,
    render_selection_user_prompt,
)
from agents.agent_find_company.result import CompanySelectionResult
from utils.openai_fallback import invoke_with_rate_limit_fallback
from utils.rdb import get_engine
from utils.rdb_queries import search_companies as search_companies_query

logger = logging.getLogger(__name__)


def run_search(
    search_input: FindCompanySearchInput,
    excluded_company_ids: list[str] | None = None,
) -> dict[str, Any]:
    normalized_input = clean_search_input(search_input)
    applied_filters = normalized_input.model_dump()
    normalized_excluded_company_ids = [
        company_id.strip()
        for company_id in (excluded_company_ids or [])
        if isinstance(company_id, str) and company_id.strip()
    ]
    logger.info(
        "[find_company/tool] filters=%s",
        json.dumps(
            {
                key: value
                for key, value in applied_filters.items()
                if value not in (None, "", [])
            },
            ensure_ascii=False,
        ),
    )
    if normalized_excluded_company_ids:
        logger.info(
            "[find_company/tool] excluded_company_ids=%s",
            ", ".join(normalized_excluded_company_ids),
        )

    results = search_companies_query(
        get_engine(),
        query=normalized_input.query,
        limit=normalized_input.limit,
        invest_level=normalized_input.invest_level,
        employees_min=normalized_input.employees_min,
        employees_max=normalized_input.employees_max,
        categories=normalized_input.categories,
        excluded_company_ids=normalized_excluded_company_ids,
    )
    payload = {
        "results": results,
        "applied_filters": {
            **applied_filters,
            "excluded_company_ids": normalized_excluded_company_ids or None,
        },
        "summary": f"{len(results)}개의 회사 후보를 찾았습니다.",
    }
    logger.info(
        "[find_company/tool] result_count=%s candidates=%s",
        len(results),
        ", ".join(
            f"{company.get('company_name')}({company.get('company_id')})"
            for company in results[:5]
        )
        or "-",
    )
    return payload


def parse_search_query(user_query: str) -> FindCompanySearchInput:
    user_message = render_search_user_prompt(user_query)
    logger.info("[find_company/user/search]\n%s", user_message)

    try:
        payload = [
            ("system", get_search_system_prompt()),
            ("user", user_message),
        ]
        planned_input = invoke_with_rate_limit_fallback(
            payload=payload,
            primary_factory=lambda: get_chat_model().with_structured_output(
                FindCompanySearchInput,
                method="json_schema",
            ),
            fallback_factory=lambda: get_fallback_chat_model().with_structured_output(
                FindCompanySearchInput,
                method="json_schema",
            ),
            logger=logger,
            operation_name="find_company.parse_search_query",
        )
    except Exception:
        fallback = default_search_input(user_query)
        logger.exception(
            "[find_company/ai/search] structured search planning failed; using fallback"
        )
        logger.info(
            "[find_company/ai/search] %s",
            json.dumps(fallback.model_dump(), ensure_ascii=False),
        )
        return fallback

    normalized_input = clean_search_input(planned_input, user_query=user_query)
    logger.info(
        "[find_company/ai/search] %s",
        json.dumps(normalized_input.model_dump(), ensure_ascii=False),
    )
    return normalized_input


def pick_company(
    user_query: str,
    candidates: list[dict[str, Any]],
) -> CompanySelectionResult:
    selection_user_message = render_selection_user_prompt(
        user_query,
        [serialize_candidate(candidate) for candidate in candidates],
    )
    logger.info("[find_company/user/select]\n%s", selection_user_message)

    try:
        payload = [
            ("system", get_selection_system_prompt()),
            ("user", selection_user_message),
        ]
        selection = invoke_with_rate_limit_fallback(
            payload=payload,
            primary_factory=lambda: get_chat_model().with_structured_output(
                CompanySelectionResult,
                method="json_schema",
            ),
            fallback_factory=lambda: get_fallback_chat_model().with_structured_output(
                CompanySelectionResult,
                method="json_schema",
            ),
            logger=logger,
            operation_name="find_company.pick_company",
        )
    except Exception:
        fallback = CompanySelectionResult(
            company_id=str(candidates[0].get("company_id", "")),
            reason="모델 응답 생성에 실패해 첫 번째 후보를 선택했습니다.",
        )
        logger.exception(
            "[find_company/ai/select] structured selection failed; using fallback"
        )
        logger.info(
            "[find_company/ai/select] %s",
            json.dumps(fallback.model_dump(), ensure_ascii=False),
        )
        return fallback

    logger.info(
        "[find_company/ai/select] %s",
        json.dumps(selection.model_dump(), ensure_ascii=False),
    )
    return selection


def clean_search_input(
    search_input: FindCompanySearchInput,
    user_query: str = "",
) -> FindCompanySearchInput:
    normalized_categories = [
        category.strip()
        for category in (search_input.categories or [])
        if isinstance(category, str) and category.strip()
    ]
    normalized_query = search_input.query.strip()
    normalized_invest_level = (
        search_input.invest_level.strip() if search_input.invest_level else None
    )
    normalized_limit = max(1, min(search_input.limit, MAX_COMPANY_CANDIDATES))

    if (
        not normalized_query
        and normalized_invest_level is None
        and search_input.employees_min is None
        and search_input.employees_max is None
        and not normalized_categories
        and user_query.strip()
    ):
        normalized_query = user_query.strip()

    return FindCompanySearchInput(
        query=normalized_query,
        invest_level=normalized_invest_level,
        employees_min=search_input.employees_min,
        employees_max=search_input.employees_max,
        categories=normalized_categories or None,
        limit=normalized_limit,
    )


def default_search_input(user_query: str) -> FindCompanySearchInput:
    return FindCompanySearchInput(
        query=user_query.strip(),
        limit=MAX_COMPANY_CANDIDATES,
    )


def serialize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "company_id": candidate.get("company_id"),
        "company_name": candidate.get("company_name"),
        "product_name": candidate.get("product_name"),
        "description": candidate.get("description"),
        "employees": candidate.get("employees"),
        "revenue": candidate.get("revenue"),
        "invest_count": candidate.get("invest_count"),
        "invest_level": candidate.get("invest_level"),
        "hiring": candidate.get("hiring"),
        "categories": candidate.get("categories"),
        "keywords": candidate.get("keywords"),
    }


def find_selected_company(
    company_id: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate.get("company_id") == company_id:
            return candidate
    return None


def format_search_summary(
    selected_company: dict[str, Any],
    reason: str,
    applied_filters: dict[str, Any] | None,
) -> str:
    company_name = selected_company.get("company_name", "알 수 없는 회사")
    company_id = selected_company.get("company_id", "알 수 없음")

    if not applied_filters:
        return f"{company_name} ({company_id})를 선택했습니다. {reason}"

    non_empty_filters = {
        key: value
        for key, value in applied_filters.items()
        if value not in (None, "", [])
    }
    if not non_empty_filters:
        return f"{company_name} ({company_id})를 선택했습니다. {reason}"

    return (
        f"{company_name} ({company_id})를 선택했습니다. {reason} "
        f"적용한 검색 조건: {non_empty_filters}"
    )
