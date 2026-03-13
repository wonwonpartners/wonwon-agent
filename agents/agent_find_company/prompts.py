from __future__ import annotations

import json
from functools import lru_cache

from utils.prompt_templates import render_prompt_template
from utils.taxonomy_prompt import generate_taxonomy_prompt


@lru_cache(maxsize=1)
def get_search_system_prompt() -> str:
    try:
        taxonomy_prompt = generate_taxonomy_prompt()
    except Exception:
        taxonomy_prompt = "분류 기준 메타데이터를 불러오지 못했습니다."

    return render_prompt_template(
        "agent_find_company/system_prompt.md",
        taxonomy_prompt=taxonomy_prompt,
    )


def render_search_user_prompt(user_query: str) -> str:
    normalized_query = user_query.strip()
    if not normalized_query:
        return ""

    return render_prompt_template(
        "agent_find_company/user_prompt.md",
        user_query=normalized_query,
    )


@lru_cache(maxsize=1)
def get_selection_system_prompt() -> str:
    return render_prompt_template("agent_find_company/selection_system_prompt.md")


def render_selection_user_prompt(
    user_query: str,
    candidates: list[dict[str, object]],
) -> str:
    serialized_candidates = json.dumps(
        candidates,
        ensure_ascii=False,
        indent=2,
    )
    return render_prompt_template(
        "agent_find_company/selection_user_prompt.md",
        user_query=user_query.strip(),
        candidates_json=serialized_candidates,
    )
