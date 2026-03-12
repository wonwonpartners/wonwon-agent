from functools import lru_cache

from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import os

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable '{name}' is required.")
    return value

def build_database_url() -> str:
    user = quote_plus(_get_required_env("POSTGRES_USER"))
    password = quote_plus(_get_required_env("POSTGRES_PASSWORD"))
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = quote_plus(_get_required_env("POSTGRES_DB"))

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"

@lru_cache(maxsize=1)
def get_engine(echo: bool = False) -> Engine:
    return create_engine(build_database_url(), echo=echo, pool_pre_ping=True)
