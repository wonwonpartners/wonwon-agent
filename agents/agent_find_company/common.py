from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from utils.openai_fallback import (
    build_chat_model,
    get_fallback_openai_model_name,
    get_openai_model_name,
)

load_dotenv()

MAX_COMPANY_CANDIDATES = 8
OPENAI_MODEL = get_openai_model_name("gpt-4o-mini")
FALLBACK_OPENAI_MODEL = get_fallback_openai_model_name("gpt-4.1-nano")


@lru_cache(maxsize=1)
def get_chat_model():
    return build_chat_model(
        model=OPENAI_MODEL,
        temperature=0,
    )


@lru_cache(maxsize=1)
def get_fallback_chat_model():
    return build_chat_model(
        model=FALLBACK_OPENAI_MODEL,
        temperature=0,
    )
