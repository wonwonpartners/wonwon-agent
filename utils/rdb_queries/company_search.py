from __future__ import annotations

from typing import Any

from sqlalchemy import and_, case, distinct, func, literal, or_, select
from sqlalchemy.engine import Engine

from utils.rdb_queries.common import get_company_tables


def search_companies(
    engine: Engine,
    query: str = "",
    limit: int = 5,
    schema: str = "public",
    invest_level: str | None = None,
    employees_min: int | None = None,
    employees_max: int | None = None,
    categories: list[str] | None = None,
    excluded_company_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    normalized_query = query.strip()
    normalized_invest_level = invest_level.strip() if invest_level else None
    normalized_categories = [
        category.strip()
        for category in (categories or [])
        if isinstance(category, str) and category.strip()
    ]
    normalized_excluded_company_ids = [
        company_id.strip()
        for company_id in (excluded_company_ids or [])
        if isinstance(company_id, str) and company_id.strip()
    ]

    if (
        not normalized_query
        and normalized_invest_level is None
        and employees_min is None
        and employees_max is None
        and not normalized_categories
    ):
        return []

    safe_limit = max(1, min(limit, 20))

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

    where_clauses = []
    order_by_clauses = []

    if normalized_query:
        like_query = f"%{normalized_query}%"
        prefix_query = f"{normalized_query}%"
        where_clauses.append(
            or_(
                companies.c.company_name.ilike(like_query),
                func.coalesce(companies.c.product_name, "").ilike(like_query),
                func.coalesce(companies.c.description, "").ilike(like_query),
                func.coalesce(categories.c.category_name, "").ilike(like_query),
                func.coalesce(keywords.c.keyword_name, "").ilike(like_query),
            )
        )
        order_by_clauses.append(
            case((companies.c.company_name.ilike(prefix_query), 0), else_=1)
        )

    if normalized_invest_level is not None:
        where_clauses.append(companies.c.invest_level == normalized_invest_level)

    if employees_min is not None:
        where_clauses.append(companies.c.employees.is_not(None))
        where_clauses.append(companies.c.employees >= employees_min)

    if employees_max is not None:
        where_clauses.append(companies.c.employees.is_not(None))
        where_clauses.append(companies.c.employees <= employees_max)

    if normalized_categories:
        where_clauses.append(categories.c.category_name.in_(normalized_categories))
    if normalized_excluded_company_ids:
        where_clauses.append(companies.c.company_id.not_in(normalized_excluded_company_ids))

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
        .where(and_(*where_clauses))
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
        .limit(safe_limit)
    )
    stmt = stmt.order_by(
        *order_by_clauses,
        companies.c.invest_count.desc().nulls_last(),
        companies.c.revenue.desc().nulls_last(),
        companies.c.company_name.asc(),
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
