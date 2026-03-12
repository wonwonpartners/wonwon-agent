from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from typing import Sequence

from sqlalchemy.engine import Engine

from utils.rdb_queries import fetch_column_comments, fetch_table_comments
from utils.rdb import get_engine


def _normalize_table_names(table_names: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []

    for table_name in table_names:
        candidate = table_name.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)

    return tuple(normalized)


def generate_schema_prompt(
    table_names: Sequence[str],
    engine: Engine | None = None,
    schema: str = "public",
) -> str:
    normalized_table_names = _normalize_table_names(table_names)
    if not normalized_table_names:
        return "테이블 정보 없음."

    if engine is not None:
        return _generate_schema_prompt(normalized_table_names, engine, schema)

    return _generate_schema_prompt_cached(normalized_table_names, schema)


@lru_cache(maxsize=32)
def _generate_schema_prompt_cached(
    table_names: tuple[str, ...],
    schema: str,
) -> str:
    try:
        engine = get_engine()
    except Exception:
        return "스키마 로드 실패"

    return _generate_schema_prompt(table_names, engine, schema)


def _generate_schema_prompt(
    table_names: tuple[str, ...],
    engine: Engine,
    schema: str,
) -> str:
    schema_info: OrderedDict[str, dict[str, object]] = OrderedDict(
        (
            table_name,
            {"description": "설명 없음", "columns": []},
        )
        for table_name in table_names
    )

    try:
        for row in fetch_table_comments(engine, table_names, schema):
            schema_info[row["table_name"]]["description"] = row["table_comment"] or "설명 없음"

        for row in fetch_column_comments(engine, table_names, schema):
            schema_info[row["table_name"]]["columns"].append(
                {
                    "name": row["column_name"],
                    "type": row["data_type"],
                    "comment": row["column_comment"] or "",
                }
            )
    except Exception:
        return "스키마 로드 실패"

    prompt_lines = [
        "아래는 현재 데이터베이스에서 조회 가능한 테이블 스키마 설명이다.",
        "테이블 및 컬럼 comment를 우선적으로 해석해서 검색과 추론에 사용할 것.",
        "",
    ]

    for table_name in table_names:
        table_info = schema_info.get(table_name)
        if table_info is None:
            continue

        prompt_lines.append(f"### Table: `{table_name}`")
        prompt_lines.append(f"- Description: {table_info['description']}")
        prompt_lines.append("- Columns:")

        columns = table_info["columns"]
        if not columns:
            prompt_lines.append("  * 컬럼 정보 없음.")
        else:
            for column in columns:
                comment = f" : {column['comment']}" if column["comment"] else ""
                prompt_lines.append(
                    f"  * `{column['name']}` ({column['type']}){comment}"
                )

        prompt_lines.append("")

    return "\n".join(prompt_lines).strip()
