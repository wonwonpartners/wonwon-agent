from __future__ import annotations

from typing import Any

from sqlalchemy import case, distinct, func, literal, or_, select
from sqlalchemy.engine import Engine

from utils.rdb_queries.common import get_company_tables


def search_companies(
    engine: Engine,
    query: str,
    limit: int = 5,
    schema: str = "public",
) -> list[dict[str, Any]]:
    normalized_query = query.strip()
    if not normalized_query:
        return []

    safe_limit = max(1, min(limit, 20))
    like_query = f"%{normalized_query}%"
    prefix_query = f"{normalized_query}%"

    companies, categories, keywords, company_categories, company_keywords = get_company_tables(
        schema
    )
    joined_tables = (
        companies.outerjoin(
            company_categories,
            company_categories.c.company_id == companies.c.company_id,
        )
        .outerjoin(categories, categories.c.category_id == company_categories.c.category_id)
        .outerjoin(
            company_keywords,
            company_keywords.c.company_id == companies.c.company_id,
        )
        .outerjoin(keywords, keywords.c.keyword_id == company_keywords.c.keyword_id)
    )

    category_names = func.coalesce(
        func.string_agg(distinct(categories.c.category_name), literal(", ")),
        "",
    ).label("categories")
    keyword_names = func.coalesce(
        func.string_agg(distinct(keywords.c.keyword_name), literal(", ")),
        "",
    ).label("keywords")

    stmt = (
        select(
            companies.c.company_id,
            companies.c.company_name,
            companies.c.product_name,
            companies.c.description,
            companies.c.employees,
            companies.c.revenue,
            companies.c.invest_count,
            companies.c.invest_level,
            companies.c.hiring,
            category_names,
            keyword_names,
        )
        .select_from(joined_tables)
        .where(
            or_(
                companies.c.company_name.ilike(like_query),
                func.coalesce(companies.c.product_name, "").ilike(like_query),
                func.coalesce(companies.c.description, "").ilike(like_query),
                func.coalesce(categories.c.category_name, "").ilike(like_query),
                func.coalesce(keywords.c.keyword_name, "").ilike(like_query),
            )
        )
        .group_by(
            companies.c.company_id,
            companies.c.company_name,
            companies.c.product_name,
            companies.c.description,
            companies.c.employees,
            companies.c.revenue,
            companies.c.invest_count,
            companies.c.invest_level,
            companies.c.hiring,
        )
        .order_by(
            case((companies.c.company_name.ilike(prefix_query), 0), else_=1),
            companies.c.invest_count.desc().nulls_last(),
            companies.c.revenue.desc().nulls_last(),
            companies.c.company_name.asc(),
        )
        .limit(safe_limit)
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()

    companies_list: list[dict[str, Any]] = []
    for row in rows:
        company = dict(row)
        company["categories"] = _split_aggregated_values(company.get("categories"))
        company["keywords"] = _split_aggregated_values(company.get("keywords"))
        companies_list.append(company)

    return companies_list


def _split_aggregated_values(value: Any) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return sorted(part for part in value.split(", ") if part)
