from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.engine import Engine

from utils.rdb_queries.common import get_company_tables


def fetch_invest_levels(
    engine: Engine,
    schema: str = "public",
) -> list[str]:
    companies, _, _, _, _ = get_company_tables(schema)
    stmt = (
        select(companies.c.invest_level)
        .distinct()
        .where(companies.c.invest_level.is_not(None))
        .where(companies.c.invest_level != "")
        .order_by(companies.c.invest_level)
    )

    with engine.connect() as conn:
        rows = conn.execute(stmt).scalars().all()

    return [row for row in rows if isinstance(row, str) and row.strip()]


def fetch_categories(
    engine: Engine,
    schema: str = "public",
    limit: int | None = None,
) -> list[str]:
    _, categories, _, _, _ = get_company_tables(schema)
    stmt = (
        select(categories.c.category_name)
        .distinct()
        .where(categories.c.category_name.is_not(None))
        .where(categories.c.category_name != "")
        .order_by(categories.c.category_name)
    )
    if limit is not None:
        stmt = stmt.limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(stmt).scalars().all()

    return [row for row in rows if isinstance(row, str) and row.strip()]
