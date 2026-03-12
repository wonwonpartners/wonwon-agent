from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from sqlalchemy.engine import Engine

from utils.rdb_queries import fetch_categories, fetch_invest_levels
from utils.rdb import get_engine

_MAX_CATEGORY_VALUES = 20

# INVEST LEVEL 을 순서대로 정렬하기 위함.
_INVEST_LEVEL_ORDER = (
    "seed",
    "pre-A",
    "series A",
    "series B",
    "series C",
    "series D",
    "pre-IPO",
    "IPO",
    "비공개",
)


def generate_taxonomy_prompt(
    engine: Engine | None = None,
    schema: str = "public",
) -> str:
    if engine is not None:
        return _generate_taxonomy_prompt(engine, schema)

    return _generate_taxonomy_prompt_cached(schema)


@lru_cache(maxsize=8)
def _generate_taxonomy_prompt_cached(schema: str) -> str:
    try:
        engine = get_engine()
    except Exception:
        return "투자 단계/카테고리 메타데이터를 로드 실패"

    return _generate_taxonomy_prompt(engine, schema)


def _generate_taxonomy_prompt(engine: Engine, schema: str) -> str:
    try:
        invest_levels = _sort_invest_levels(fetch_invest_levels(engine, schema))
        categories = fetch_categories(engine, schema, limit=_MAX_CATEGORY_VALUES)
    except Exception:
        return "투자 단계/카테고리 메타데이터를 로드 실패"

    invest_level_line = ", ".join(invest_levels) if invest_levels else "값이 없음"
    category_line = ", ".join(categories) if categories else "값이 없음"

    return "\n".join(
        [
            "아래는 검색 정규화에 사용하는 투자 단계 및 카테고리 canonical value 목록입니다.",
            "사용자 표현을 가능한 한 아래 값으로 정규화해서 검색하세요.",
            "",
            f"- companies.invest_level: {invest_level_line}",
            f"- categories.category_name: {category_line}",
        ]
    )


def _sort_invest_levels(values: Iterable[str]) -> list[str]:
    order_map = {value: index for index, value in enumerate(_INVEST_LEVEL_ORDER)}
    unique_values = sorted(set(values), key=lambda value: value.lower())

    return sorted(
        unique_values,
        key=lambda value: (order_map.get(value, len(order_map)), value.lower()),
    )
