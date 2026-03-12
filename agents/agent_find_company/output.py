from __future__ import annotations

from typing import Any, TypedDict


class FindCompanyNodeOutput(TypedDict):
    company_search_summary: str
    company_search_filters: dict[str, Any] | None
    selected_company: dict[str, Any] | None
    selected_company_reason: str
