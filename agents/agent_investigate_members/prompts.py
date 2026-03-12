from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from utils.prompt_templates import render_prompt_template


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return render_prompt_template("agent_investigate_members/system_prompt.md")


def render_user_prompt(
    company_profile: dict[str, Any],
    signals: list[dict[str, Any]],
) -> str:
    return render_prompt_template(
        "agent_investigate_members/user_prompt.md",
        company_profile_json=json.dumps(
            company_profile,
            ensure_ascii=False,
            indent=2,
        ),
        signals_json=json.dumps(
            signals,
            ensure_ascii=False,
            indent=2,
        ),
    )
