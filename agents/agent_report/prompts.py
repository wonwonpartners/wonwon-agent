from __future__ import annotations

from functools import lru_cache

from utils.prompt_templates import render_prompt_template


@lru_cache(maxsize=1)
def get_system_prompt() -> str:
    return render_prompt_template("agent_report/system_prompt.md")


def render_user_prompt(report_input_text: str) -> str:
    return render_prompt_template(
        "agent_report/user_prompt.md",
        report_input=report_input_text,
    )
