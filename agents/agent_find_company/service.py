from __future__ import annotations

import json
import logging
from typing import Any

from agents.agent_find_company.common import MAX_COMPANY_CANDIDATES, get_chat_model
from agents.agent_find_company.input import FindCompanySearchInput
from agents.agent_find_company.prompts import (
    get_search_system_prompt,
    get_selection_system_prompt,
    render_search_user_prompt,
    render_selection_user_prompt,
)
from agents.agent_find_company.result import CompanySelectionResult
from utils.rdb import get_engine
from utils.rdb_queries import search_companies as search_companies_query

logger = logging.getLogger(__name__)

SELECTION_CANDIDATE_FIELDS = (
    "company_id",
    "company_name",
    "product_name",
    "description",
    "employees",
    "revenue",
    "invest_count",
    "invest_level",
    "hiring",
    "categories",
    "keywords",
)


def to_log_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def compact_filters(filters: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in filters.items()
        if value not in (None, "", [])
    }


def format_candidate_preview(candidates: list[dict[str, Any]], *, limit: int = 5) -> str:
    preview = [
        f"{company.get('company_name')}({company.get('company_id')})"
        for company in candidates[:limit]
    ]
    return ", ".join(preview) or "-"


def run_search(
    search_input: FindCompanySearchInput,
) -> dict[str, Any]:
    normalized_input = clean_search_input(search_input)
    applied_filters = normalized_input.model_dump()
    logger.info(
        "[find_company/tool] filters=%s",
        to_log_json(compact_filters(applied_filters)),
    )

    results = search_companies_query(
        get_engine(),
        query=normalized_input.query,
        limit=normalized_input.limit,
        invest_level=normalized_input.invest_level,
        employees_min=normalized_input.employees_min,
        employees_max=normalized_input.employees_max,
        categories=normalized_input.categories,
    )
    payload = {
        "results": results,
        "applied_filters": applied_filters,
        "summary": f"{len(results)}개의 회사 후보를 찾았습니다.",
    }
    logger.info(
        "[find_company/tool] result_count=%s candidates=%s",
        len(results),
        format_candidate_preview(results),
    )
    return payload


def parse_search_query(user_query: str) -> FindCompanySearchInput:
    user_message = render_search_user_prompt(user_query)
    logger.info("[find_company/user/search]\n%s", user_message)

    try:
        planner = get_chat_model().with_structured_output(
            FindCompanySearchInput,
            method="json_schema",
        )
        planned_input = planner.invoke(
            [
                ("system", get_search_system_prompt()),
                ("user", user_message),
            ]
        )
    except Exception:
        fallback = default_search_input(user_query)
        logger.exception(
            "[find_company/ai/search] structured search planning failed; using fallback"
        )
        logger.info(
            "[find_company/ai/search] %s",
            to_log_json(fallback.model_dump()),
        )
        return fallback

    normalized_input = clean_search_input(planned_input, user_query=user_query)
    logger.info(
        "[find_company/ai/search] %s",
        to_log_json(normalized_input.model_dump()),
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
        selector = get_chat_model().with_structured_output(
            CompanySelectionResult,
            method="json_schema",
        )
        selection = selector.invoke(
            [
                ("system", get_selection_system_prompt()),
                ("user", selection_user_message),
            ]
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
            to_log_json(fallback.model_dump()),
        )
        return fallback

    logger.info(
        "[find_company/ai/select] %s",
        to_log_json(selection.model_dump()),
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
        field_name: candidate.get(field_name)
        for field_name in SELECTION_CANDIDATE_FIELDS
    }


def find_selected_company(
    company_id: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.get("company_id") == company_id
        ),
        None,
    )


def format_search_summary(
    selected_company: dict[str, Any],
    reason: str,
    applied_filters: dict[str, Any] | None,
) -> str:
    company_name = selected_company.get("company_name", "알 수 없는 회사")
    company_id = selected_company.get("company_id", "알 수 없음")

    if not applied_filters:
        return f"{company_name} ({company_id})를 선택했습니다. {reason}"

    non_empty_filters = compact_filters(applied_filters)
    if not non_empty_filters:
        return f"{company_name} ({company_id})를 선택했습니다. {reason}"

    return (
        f"{company_name} ({company_id})를 선택했습니다. {reason} "
        f"적용한 검색 조건: {non_empty_filters}"
    )
