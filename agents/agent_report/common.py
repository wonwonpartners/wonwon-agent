from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = PROJECT_ROOT / "outputs" / "reports"

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env", override=False)

REPORT_OPENAI_MODEL = os.getenv("REPORT_OPENAI_MODEL", "gpt-4o")
REPORT_PDF_FONT_PATH = os.getenv("REPORT_PDF_FONT_PATH", "").strip()
DEFAULT_REPORT_FONT_PATH = Path("/Library/Fonts/Arial Unicode.ttf")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value and name == "OPENAI_API_KEY":
        value = os.getenv("OPEN_AI_KEY")
    if value:
        return value
    raise RuntimeError(f"{name} 환경변수가 필요합니다.")


@lru_cache(maxsize=1)
def get_chat_model():
    api_key = require_env("OPENAI_API_KEY")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=REPORT_OPENAI_MODEL,
        temperature=0,
        api_key=api_key,
    )


def resolve_report_font_path() -> Path:
    if REPORT_PDF_FONT_PATH:
        return Path(REPORT_PDF_FONT_PATH).expanduser().resolve()
    return DEFAULT_REPORT_FONT_PATH
