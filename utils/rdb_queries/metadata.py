from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine


def fetch_table_comments(
    engine: Engine,
    table_names: Sequence[str],
    schema: str = "public",
) -> list[dict[str, Any]]:
    stmt = text(
        """
        SELECT
            tables.table_name,
            COALESCE(pg_catalog.obj_description(cls.oid, 'pg_class'), '') AS table_comment
        FROM information_schema.tables AS tables
        JOIN pg_catalog.pg_class AS cls
            ON cls.relname = tables.table_name
        JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = cls.relnamespace
           AND namespace.nspname = tables.table_schema
        WHERE tables.table_schema = :schema
          AND tables.table_type = 'BASE TABLE'
          AND tables.table_name IN :table_names
        ORDER BY tables.table_name;
        """
    ).bindparams(bindparam("table_names", expanding=True))

    with engine.connect() as conn:
        return list(
            conn.execute(
                stmt,
                {"schema": schema, "table_names": list(table_names)},
            ).mappings()
        )


def fetch_column_comments(
    engine: Engine,
    table_names: Sequence[str],
    schema: str = "public",
) -> list[dict[str, Any]]:
    stmt = text(
        """
        SELECT
            cols.table_name,
            cols.column_name,
            pg_catalog.format_type(attr.atttypid, attr.atttypmod) AS data_type,
            COALESCE(pg_catalog.col_description(cls.oid, attr.attnum), '') AS column_comment
        FROM information_schema.columns AS cols
        JOIN pg_catalog.pg_class AS cls
            ON cls.relname = cols.table_name
        JOIN pg_catalog.pg_namespace AS namespace
            ON namespace.oid = cls.relnamespace
           AND namespace.nspname = cols.table_schema
        JOIN pg_catalog.pg_attribute AS attr
            ON attr.attrelid = cls.oid
           AND attr.attname = cols.column_name
           AND attr.attnum > 0
           AND NOT attr.attisdropped
        WHERE cols.table_schema = :schema
          AND cols.table_name IN :table_names
        ORDER BY cols.table_name, cols.ordinal_position;
        """
    ).bindparams(bindparam("table_names", expanding=True))

    with engine.connect() as conn:
        return list(
            conn.execute(
                stmt,
                {"schema": schema, "table_names": list(table_names)},
            ).mappings()
        )
