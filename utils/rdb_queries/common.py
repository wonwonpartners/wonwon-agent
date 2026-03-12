from __future__ import annotations

from functools import lru_cache

from sqlalchemy import BIGINT, BOOLEAN, INTEGER, TEXT, VARCHAR, Column, MetaData, Table


@lru_cache(maxsize=8)
def get_company_tables(schema: str) -> tuple[Table, Table, Table, Table, Table]:
    metadata = MetaData()
    companies = Table(
        "companies",
        metadata,
        Column("company_id", VARCHAR(50)),
        Column("company_name", VARCHAR(255)),
        Column("product_name", VARCHAR(255)),
        Column("description", TEXT),
        Column("employees", INTEGER),
        Column("revenue", BIGINT),
        Column("invest_count", INTEGER),
        Column("invest_level", VARCHAR(100)),
        Column("hiring", BOOLEAN),
        schema=schema,
    )
    categories = Table(
        "categories",
        metadata,
        Column("category_id", INTEGER),
        Column("category_name", VARCHAR(255)),
        schema=schema,
    )
    keywords = Table(
        "keywords",
        metadata,
        Column("keyword_id", INTEGER),
        Column("keyword_name", VARCHAR(255)),
        schema=schema,
    )
    company_categories = Table(
        "company_categories",
        metadata,
        Column("company_id", VARCHAR(50)),
        Column("category_id", INTEGER),
        schema=schema,
    )
    company_keywords = Table(
        "company_keywords",
        metadata,
        Column("company_id", VARCHAR(50)),
        Column("keyword_id", INTEGER),
        schema=schema,
    )
    return companies, categories, keywords, company_categories, company_keywords
