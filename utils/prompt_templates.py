from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_ROOT = PROJECT_ROOT / "prompts"


@lru_cache(maxsize=32)
def load_prompt_template(template_path: str) -> str:
    return (PROMPTS_ROOT / template_path).read_text(encoding="utf-8").strip()


def render_prompt_template(template_path: str, **kwargs: str) -> str:
    return load_prompt_template(template_path).format(**kwargs)
