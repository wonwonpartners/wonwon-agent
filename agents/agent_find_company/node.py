from __future__ import annotations

import logging
from typing import Any

from agents.agent_find_company.output import FindCompanyNodeOutput
from agents.agent_find_company.service import (
    find_selected_company,
    format_search_summary,
    parse_search_query,
    pick_company,
    run_search,
)

logger = logging.getLogger(__name__)


def find_company_node(state: dict[str, Any]) -> FindCompanyNodeOutput:
    user_query = str(state.get("user_query", "")).strip()
    if not user_query:
        return {
            "company_search_summary": "회사 검색을 위한 사용자 질의가 없습니다.",
            "company_search_filters": None,
            "selected_company": None,
            "selected_company_reason": "사용자 질의가 비어 있습니다.",
        }

    search_input = parse_search_query(user_query)
    search_payload = run_search(search_input)
    candidates = search_payload["results"]
    applied_filters = search_payload["applied_filters"]

    if not candidates:
        logger.info("[find_company/final] no company matched the search filters")
        return {
            "company_search_summary": "질의에 맞는 회사를 찾지 못했습니다.",
            "company_search_filters": applied_filters,
            "selected_company": None,
            "selected_company_reason": "검색 결과가 없습니다.",
        }

    if len(candidates) == 1:
        selected_company = candidates[0]
        selection_reason = "검색 결과가 1건이라 해당 회사를 바로 선택했습니다."
        logger.info(
            "[find_company/final] selected=%s(%s) reason=%s",
            selected_company.get("company_name"),
            selected_company.get("company_id"),
            selection_reason,
        )
        return {
            "company_search_summary": format_search_summary(
                selected_company,
                selection_reason,
                applied_filters,
            ),
            "company_search_filters": applied_filters,
            "selected_company": selected_company,
            "selected_company_reason": selection_reason,
        }

    selection = pick_company(user_query, candidates)
    selected_company = find_selected_company(selection.company_id, candidates)
    selection_reason = selection.reason

    if selected_company is None:
        selected_company = candidates[0]
        selection_reason = (
            "모델 응답을 후보 목록과 정확히 매칭하지 못해 첫 번째 후보를 선택했습니다."
        )

    logger.info(
        "[find_company/final] selected=%s(%s) reason=%s",
        selected_company.get("company_name"),
        selected_company.get("company_id"),
        selection_reason,
    )
    return {
        "company_search_summary": format_search_summary(
            selected_company,
            selection_reason,
            applied_filters,
        ),
        "company_search_filters": applied_filters,
        "selected_company": selected_company,
        "selected_company_reason": selection_reason,
    }
