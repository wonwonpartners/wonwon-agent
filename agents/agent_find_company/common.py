from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

FIND_COMPANY_TABLES = (
    "companies",
    "categories",
    "keywords",
    "company_categories",
    "company_keywords",
)
MAX_COMPANY_CANDIDATES = 8


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
